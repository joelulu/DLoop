"""Greedy, batch-one DFlash experiments with variable retained prefixes.

Every draft forward retains the checkpoint's original width. Only the candidate
prefix and its aligned hidden carriers are shortened. No target features are
invented for unverified tokens. Sampling is deliberately unsupported.
"""

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from transformers import DynamicCache


@dataclass(frozen=True)
class PrefixPolicy:
    mode: str = "adaptive_loop"
    rule: str = "prefix_logp"
    threshold: float = -1.0
    loop_gate: float = -1.0
    max_draft_passes: int = 2  # TOTAL forwards, unlike upstream stage_cap
    min_continue_tokens: int = 1
    max_candidates: int = 30
    total_logp_budget: float | None = None
    min_confident_ratio: float = 0.8
    min_confident_tokens: int | None = None
    cut_offset: int = 0

    def __post_init__(self):
        if self.mode not in {"single", "original_loop", "truncate_verify", "adaptive_loop", "gated_prefix_loop"}:
            raise ValueError("unknown decode mode")
        if self.rule not in {"prefix_logp", "token_prob"}:
            raise ValueError("unknown prefix rule")
        if not math.isfinite(self.threshold) or not math.isfinite(self.loop_gate):
            raise ValueError("thresholds must be finite")
        if self.rule == "prefix_logp" and self.threshold > 0:
            raise ValueError("prefix_logp threshold must be <= 0")
        if self.rule == "token_prob" and not 0 < self.threshold <= 1:
            raise ValueError("token probability must be in (0, 1]")
        if min(self.max_draft_passes, self.min_continue_tokens, self.max_candidates) < 1:
            raise ValueError("pass, progress and candidate limits must be positive")
        if not math.isfinite(self.min_confident_ratio) or not 0 < self.min_confident_ratio <= 1:
            raise ValueError("min_confident_ratio must be in (0, 1]")
        if not isinstance(self.cut_offset, int):
            raise ValueError("cut_offset must be an integer")
        if self.min_confident_tokens is not None and (
            not isinstance(self.min_confident_tokens, int) or self.min_confident_tokens < 1
        ):
            raise ValueError("min_confident_tokens must be a positive integer")
        if self.mode == "gated_prefix_loop" and self.total_logp_budget is not None:
            raise ValueError("gated_prefix_loop does not use the legacy total_logp_budget")
        if self.total_logp_budget is not None and (
            not math.isfinite(self.total_logp_budget) or self.total_logp_budget > 0
        ):
            raise ValueError("total_logp_budget must be finite and <= 0")


def select_prefix(token_logp, rule, threshold):
    """Length of a CONTIGUOUS prefix; later confidence cannot skip an error."""
    if rule == "prefix_logp":
        allowed = token_logp.cumsum(-1) >= threshold
    elif rule == "token_prob":
        allowed = token_logp >= math.log(threshold)
    else:
        raise ValueError(rule)
    return int(allowed.to(torch.int64).cumprod(-1).sum().item())


def gated_prefix_decision(token_logp, policy, room, pass_index):
    """Gate on the UNshifted prefix; offset only the boundary for another pass.

    The final pass and failed gates preserve the full proposal. The ratio always
    refers to the checkpoint's full candidate width, never a tail-clipped width.
    Returns (retained length, continue, raw confident length, reason).
    """
    width = token_logp.numel()
    minimum = policy.min_confident_tokens if policy.min_confident_tokens is not None else math.ceil(
        policy.min_confident_ratio * width)
    if minimum > width:
        raise ValueError(f"confident prefix minimum {minimum} exceeds candidate width {width}")
    full = min(width, room)
    if pass_index + 1 >= policy.max_draft_passes:
        return full, False, None, "final_full_block"
    if room <= width:
        return full, False, None, "tail_full_block"
    raw = select_prefix(token_logp, policy.rule, policy.threshold)
    if minimum == width and policy.rule == "prefix_logp":
        # Match original_loop's reduction exactly at the full-width gate,
        # including floating-point borderline cases (sum vs cumsum endpoint).
        raw = width if float(token_logp.sum()) >= policy.threshold else min(raw, width - 1)
    if raw < minimum:
        return full, False, raw, "ratio_gate_failed_full_verify"
    cut = min(width, max(1, raw + policy.cut_offset))
    if cut < policy.min_continue_tokens:
        return full, False, raw, "progress_gate_failed_full_verify"
    return cut, True, raw, "continue_from_shifted_prefix"


def _clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _context(output, model):
    # Same layer_id + 1 convention as extract_context_feature in dflash.py.
    return torch.cat([output.hidden_states[i + 1] for i in model.target_layer_ids], -1)


def _crop_to(cache, length):
    # Negative removal counts work with both the original and newer HF cache
    # APIs; newer versions no longer accept an absolute positive crop length.
    remove = cache.get_seq_length() - length
    if remove < 0:
        raise RuntimeError("cannot extend a cache by cropping")
    if remove:
        cache.crop(-remove)


@torch.inference_mode()
def adaptive_dflash_generate(model, target, input_ids, max_new_tokens, stop_token_ids,
                             policy=None, trace=False):
    """Return capped, EOS-trimmed output and consistently defined decode stats.

    mode='original_loop' uses upstream whole-segment gating and carrier layout;
    the common verifier/timer makes all four policies directly comparable.
    model=None runs the target-only greedy reference through the same verifier.
    Detailed traces are optional and should be disabled for timing runs.
    """
    policy = policy or PrefixPolicy()
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("requires one nonempty prompt (batch size 1)")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    block = model.block_size if model is not None else 1
    if model is not None and block < 2:
        raise ValueError("DFlash checkpoint must have block_size >= 2")
    if model is not None and policy.min_confident_tokens is not None and policy.min_confident_tokens > block - 1:
        raise ValueError("min_confident_tokens exceeds the checkpoint's candidate count")
    stop_ids = set(stop_token_ids or [])
    target_cache, draft_cache = DynamicCache(), DynamicCache()
    positions = torch.arange(prompt_len + max_new_tokens + block, device=device)[None]
    prefill_start = _clock(device)
    out = target(input_ids, position_ids=positions[:, :prompt_len],
                 past_key_values=target_cache, use_cache=True, logits_to_keep=1,
                 output_hidden_states=model is not None)
    generated = out.logits[:, -1:].argmax(-1)
    target_hidden = _context(out, model) if model is not None else None
    ttft = _clock(device) - prefill_start
    decode_start = _clock(device)
    accepted_lengths, committed_lengths, verify_lengths, pass_counts = [], [], [], []
    retained_lengths, round_traces = [], []
    draft_calls = 0
    prefix_counts = dict(gate_checks=0, gate_passes=0, shift_clipped=0,
                         raw_confident_sum=0, applied_cut_sum=0)
    initial_eos = int(generated[0, 0]) in stop_ids

    while generated.shape[1] < max_new_tokens and not initial_eos:
        start = prompt_len + generated.shape[1] - 1  # current known bonus/anchor
        remaining = max_new_tokens - generated.shape[1]
        cap = min(policy.max_candidates, remaining)
        candidates = generated[:, -1:]  # anchor + unverified proposals
        carriers = None
        retained_this_round = []
        decision_trace = []
        cumulative_logp = 0.0
        passes = 0
        stop_reason = "target_only"
        if model is not None:
            for pass_idx in range(policy.max_draft_passes):
                L = candidates.shape[1] - 1
                rows = torch.cat([candidates[:, -1:], torch.full(
                    (1, block - 1), model.mask_token_id, device=device, dtype=torch.long)], 1)
                if pass_idx == 0:
                    hidden = model(
                        target_hidden=target_hidden, noise_embedding=target.model.embed_tokens(rows),
                        position_ids=positions[:, draft_cache.get_seq_length():start + block],
                        past_key_values=draft_cache, use_cache=True, is_causal=False,
                    )[:, -(block - 1):]
                else:
                    # Preserve the upstream carrier mapping, including omission
                    # of the current anchor from the injected hidden stream.
                    hidden = model(
                        target_hidden=target_hidden[:, :0],
                        noise_embedding=target.model.embed_tokens(rows),
                        position_ids=torch.cat([positions[:, start + 1:start + L],
                                                positions[:, start + L:start + L + block]], 1),
                        past_key_values=draft_cache, use_cache=True, is_causal=False,
                        extra_kv_hidden=carriers[:, :L - 1],
                    )[:, -(block - 1):]
                _crop_to(draft_cache, start)
                logits = target.lm_head(hidden).float()
                proposed = logits.argmax(-1)
                logp = logits.log_softmax(-1).gather(-1, proposed[..., None]).squeeze(-1)[0]
                passes += 1
                draft_calls += 1
                room = cap - L
                allow_continue = None
                raw_confident = None
                if policy.mode == "gated_prefix_loop":
                    keep, allow_continue, raw_confident, stop_reason = gated_prefix_decision(
                        logp, policy, room, pass_idx)
                    if raw_confident is not None:
                        prefix_counts["gate_checks"] += 1
                        prefix_counts["raw_confident_sum"] += raw_confident
                        if allow_continue:
                            prefix_counts["gate_passes"] += 1
                            prefix_counts["applied_cut_sum"] += keep
                            prefix_counts["shift_clipped"] += int(keep != raw_confident + policy.cut_offset)
                elif policy.mode in {"single", "original_loop"}:
                    keep = min(block - 1, room)
                else:
                    keep = select_prefix(logp[:room], policy.rule, policy.threshold)
                    if policy.total_logp_budget is not None:
                        keep = min(keep, select_prefix(
                            logp[:room], "prefix_logp", policy.total_logp_budget - cumulative_logp))
                # First pass: if no useful prefix exists, verify the original
                # proposals. Later pass: retain none and verify accumulated ones.
                zero_fallback = keep == 0 and L == 0
                if zero_fallback:
                    keep = min(block - 1, room)
                candidates = torch.cat([candidates, proposed[:, :keep]], 1)
                carriers = hidden[:, :keep] if carriers is None else torch.cat(
                    [carriers, hidden[:, :keep]], 1)
                retained_this_round.append(keep)
                if trace:
                    decision_trace.append(dict(raw_confident=raw_confident, retained=keep,
                                               cut_offset=policy.cut_offset,
                                               allow_continue=allow_continue, reason=stop_reason))
                if trace or policy.total_logp_budget is not None:
                    cumulative_logp += float(logp[:keep].sum())
                if zero_fallback:
                    stop_reason = "zero_prefix_fallback"
                    break
                if policy.mode == "gated_prefix_loop" and not allow_continue:
                    break
                if policy.mode in {"single", "truncate_verify"}:
                    stop_reason = "single_pass"
                    break
                if candidates.shape[1] - 1 >= cap:
                    stop_reason = "candidate_cap"
                    break
                if keep < policy.min_continue_tokens:
                    stop_reason = "insufficient_progress"
                    break
                if policy.mode == "original_loop" and float(logp.sum()) < policy.loop_gate:
                    stop_reason = "whole_block_gate"
                    break
                stop_reason = "pass_cap"

        out = target(candidates, position_ids=positions[:, start:start + candidates.shape[1]],
                     past_key_values=target_cache, use_cache=True,
                     output_hidden_states=model is not None)
        posterior = out.logits.argmax(-1)
        acc = int((candidates[:, 1:] == posterior[:, :-1]).long().cumprod(1).sum())
        additions = torch.cat([candidates[:, 1:acc + 1], posterior[:, acc:acc + 1]], 1)[:, :remaining]
        # Only EOS in tokens actually committed may stop generation. Speculative
        # EOS beyond an earlier rejection must never terminate the output.
        eos_hits = torch.zeros_like(additions, dtype=torch.bool)
        for eos in stop_ids:
            eos_hits |= additions == eos
        hits = eos_hits[0].nonzero()
        stopped = bool(hits.numel())
        if stopped:
            additions = additions[:, :int(hits[0, 0]) + 1]
        effective_acc = min(acc, additions.shape[1])
        generated = torch.cat([generated, additions], 1)
        _crop_to(target_cache, start + acc + 1)
        if model is not None:
            target_hidden = _context(out, model)[:, :acc + 1]
        accepted_lengths.append(effective_acc)
        committed_lengths.append(additions.shape[1])
        verify_lengths.append(candidates.shape[1])  # includes anchor
        pass_counts.append(passes)
        retained_lengths.extend(retained_this_round)
        if trace:
            top = out.logits[0].float().topk(min(2, out.logits.shape[-1]), dim=-1)
            round_traces.append(dict(anchor_position=start, retained=retained_this_round,
                                     prefix_decisions=decision_trace,
                                     output_start=start - prompt_len + 1,
                                     candidate_ids=candidates[0].tolist(),
                                     posterior_top_ids=top.indices.tolist(),
                                     posterior_top_logits=top.values.tolist(),
                                     accepted=effective_acc, committed=additions.shape[1],
                                     verify_width=candidates.shape[1], stop_reason=stop_reason,
                                     second_segment_accepted=max(0, effective_acc - (
                                         retained_this_round[0] if retained_this_round else 0))))
        if stopped:
            break

    decode_seconds = _clock(device) - decode_start
    decode_tokens = generated.shape[1] - 1  # first token belongs to TTFT
    return SimpleNamespace(
        output_ids=torch.cat([input_ids, generated], 1), num_input_tokens=prompt_len,
        num_output_tokens=generated.shape[1], decode_tokens=decode_tokens,
        decode_seconds=decode_seconds, time_to_first_token=ttft,
        time_per_output_token=decode_seconds / max(decode_tokens, 1),
        acceptance_lengths=accepted_lengths, committed_lengths=committed_lengths,
        verify_lengths=verify_lengths, draft_pass_counts=pass_counts,
        retained_lengths=retained_lengths, verify_calls=len(verify_lengths),
        draft_calls=draft_calls, stages_fired=[], round_traces=round_traces,
        prefix_decision_counts=prefix_counts,
    )
