"""Untimed diagnostics for greedy AR / block-verification mismatches.

A mismatch is evidence of different outputs, not by itself proof of either
cache corruption or harmless numerical drift. Keep that distinction explicit.
"""


def first_difference(reference, observed):
    for i, (a, b) in enumerate(zip(reference, observed)):
        if a != b:
            return i
    return min(len(reference), len(observed)) if len(reference) != len(observed) else None


def mismatch_summary(reference, observed, tokenizer=None):
    index = first_difference(reference, observed)
    result = dict(first_difference=index, reference_length=len(reference), observed_length=len(observed))
    if index is None:
        return result
    result.update(reference_token=reference[index] if index < len(reference) else None,
                  observed_token=observed[index] if index < len(observed) else None)
    if tokenizer is not None:
        lo, hi = max(0, index - 12), index + 12
        result.update(reference_text=tokenizer.decode(reference[lo:hi]),
                      observed_text=tokenizer.decode(observed[lo:hi]))
    return result


def _top2(logits):
    top = logits.float().topk(min(2, logits.shape[-1]))
    values = top.values.tolist()
    return dict(ids=top.indices.tolist(), logits=values,
                margin=values[0] - values[1] if len(values) > 1 else None)


def diagnose_mismatch(target, draft, input_ids, reference, observed, policy,
                      max_new_tokens, stop_ids, tokenizer=None):
    import torch
    from transformers import DynamicCache
    from .adaptive_linear_decode import adaptive_dflash_generate

    report = mismatch_summary(reference, observed, tokenizer)
    j = report["first_difference"]
    if j is None:
        return report
    with torch.inference_mode():
        rerun = adaptive_dflash_generate(draft, target, input_ids, max_new_tokens,
                                         stop_ids, policy, trace=True)
        rerun_ids = rerun.output_ids[0, input_ids.shape[1]:].tolist()
        report["rerun_matches_observed"] = rerun_ids == observed
        report["rerun_difference"] = mismatch_summary(reference, rerun_ids, tokenizer)
        report["rounds"] = rerun.round_traces
        if policy.mode == "single":
            try:
                from .linear_decode import dflash_generate
                upstream = dflash_generate(draft, target, input_ids, max_new_tokens, stop_ids,
                                           loop_gate=1.0, stage_cap=0)
                upstream_ids = upstream[0, input_ids.shape[1]:].tolist()
                report["upstream_matches_reference"] = upstream_ids == reference
                report["upstream_matches_observed"] = upstream_ids == observed
                report["upstream_difference"] = mismatch_summary(reference, upstream_ids, tokenizer)
            except Exception as exc:
                report["upstream_diagnostic_error"] = repr(exc)
        for row in rerun.round_traces:
            if row["output_start"] <= j < row["output_start"] + row["committed"]:
                k = j - row["output_start"]
                scores = row["posterior_top_logits"][k]
                report["block_verifier_at_difference"] = dict(
                    round_output_start=row["output_start"], row=k,
                    ids=row["posterior_top_ids"][k], logits=scores,
                    margin=scores[0] - scores[1] if len(scores) > 1 else None)
                break
        # Teacher-force the IDENTICAL shared prefix. Do not compare logits after
        # the first divergence, where the contexts are already different.
        if j < len(reference) and j < len(observed):
            cache = DynamicCache()
            out = target(input_ids, past_key_values=cache, use_cache=True,
                         logits_to_keep=1, output_hidden_states=False)
            for token in reference[:j]:
                token_ids = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
                out = target(token_ids, past_key_values=cache, use_cache=True,
                             logits_to_keep=1, output_hidden_states=False)
            report["ar_shared_prefix_top2"] = _top2(out.logits[0, -1])
            shared = torch.cat([input_ids, torch.tensor([reference[:j]], device=input_ids.device,
                                                       dtype=input_ids.dtype)], 1)
            out = target(shared, use_cache=False, logits_to_keep=1, output_hidden_states=True)
            report["full_shared_prefix_top2"] = _top2(out.logits[0, -1])
    report["interpretation"] = (
        "Inspect the first shared-prefix difference. Small margins and shape-dependent "
        "argmax changes suggest numerical sensitivity but do not prove the absence of a bug. "
        "Eager/FP32 reruns are diagnostics, not performance results."
    )
    return report
