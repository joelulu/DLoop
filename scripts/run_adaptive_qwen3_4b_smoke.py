"""One-process Qwen3-4B prefix-loop screening; run from the repo root.

Example: python scripts/run_adaptive_qwen3_4b_smoke.py --device cuda:0
Models are loaded once. Tuning and confirmation sample IDs are disjoint.
All timed configurations use the same verifier, stopping and timing semantics.
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--draft-model", default="naver-ai/DLoop-DFlash-Qwen3-4B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--experiment", choices=["legacy", "prefix_offsets"], default="legacy")
    p.add_argument("--cut-offsets", type=int, nargs="+", default=[-3, -2, -1, 0, 1, 2, 4])
    p.add_argument("--min-confident-ratio", type=float, default=0.8)
    p.add_argument("--prefix-threshold", type=float, default=-2.0)
    p.add_argument("--target-attention", choices=["sdpa", "eager"], default="sdpa")
    p.add_argument("--model-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--ar-check", choices=["strict", "warn"], default="strict",
                   help="strict diagnoses then aborts; warn is exploratory and marks results non-equivalent")
    p.add_argument("--mismatch-diagnostics", choices=["each", "first", "off"], default="each",
                   help="untimed AR mismatch replay frequency; off still records AR-match counts")
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "mbpp"])
    p.add_argument("--data-file", type=Path, help="optional local JSONL: {turns: [prompt]} or {prompt: ...}")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--smoke-samples", type=int, default=3)
    p.add_argument("--tune-samples", type=int, default=16)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-draft-passes", type=int, default=2)
    p.add_argument("--min-continue-tokens", type=int, default=1)
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--total-logp-budget", type=float)
    p.add_argument("--trace-samples", type=int, default=0, help="separate untimed diagnostic reruns")
    args = p.parse_args()
    for name in ("smoke_samples", "tune_samples", "eval_samples", "max_new_tokens", "repeats"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_new_tokens < 2:
        p.error("--max-new-tokens must be >= 2 to measure decode throughput")
    if args.experiment == "prefix_offsets" and args.total_logp_budget is not None:
        p.error("prefix_offsets does not use --total-logp-budget")
    return args


def experiment_configs(args):
    from eval.adaptive_linear_decode import PrefixPolicy
    common = dict(max_draft_passes=args.max_draft_passes, min_continue_tokens=args.min_continue_tokens,
                  max_candidates=args.max_candidates, total_logp_budget=args.total_logp_budget)
    configs = {"single": PrefixPolicy(mode="single", **common)}
    for gate in (-0.5, -1.0, -2.0):
        configs[f"original_g{gate}"] = PrefixPolicy(mode="original_loop", loop_gate=gate, **common)
    if args.experiment == "prefix_offsets":
        for offset in dict.fromkeys(args.cut_offsets):
            name = f"gated_r{args.min_confident_ratio}_g{args.prefix_threshold}_offset{offset:+d}"
            configs[name] = PrefixPolicy(mode="gated_prefix_loop", threshold=args.prefix_threshold,
                                         min_confident_ratio=args.min_confident_ratio,
                                         cut_offset=offset, **common)
    else:
        for rule, thresholds in (("prefix_logp", (-0.5, -1.0, -2.0)), ("token_prob", (0.8, 0.9, 0.95))):
            for threshold in thresholds:
                for mode in ("truncate_verify", "adaptive_loop"):
                    configs[f"{mode}_{rule}_{threshold}"] = PrefixPolicy(
                        mode=mode, rule=rule, threshold=threshold, **common)
    return configs


def select_confirmation_configs(configs, scores, experiment):
    selected = {"single": "single"}
    modes = ("original_loop",) if experiment == "prefix_offsets" else (
        "original_loop", "truncate_verify", "adaptive_loop")
    for mode in modes:
        candidates = [n for n, p in configs.items() if p.mode == mode]
        selected[mode] = max(candidates, key=lambda n: scores[n]["tok_s"])
    if experiment == "prefix_offsets":
        # Report EVERY offset on the same held-out samples, not only a winner.
        selected.update({name: name for name, p in configs.items() if p.mode == "gated_prefix_loop"})
    return selected


def summarize(rows):
    tokens = sum(r["decode_tokens"] for r in rows)
    seconds = sum(r["decode_seconds"] for r in rows)
    verifies = sum(r["verify_calls"] for r in rows)
    accepted = sum(sum(r["acceptance_lengths"]) for r in rows)
    retained = [n for r in rows for n in r["retained_lengths"]]
    passes = [n for r in rows for n in r.get("draft_pass_counts", [])]
    gate_checks = sum(r.get("prefix_decision_counts", {}).get("gate_checks", 0) for r in rows)
    gate_passes = sum(r.get("prefix_decision_counts", {}).get("gate_passes", 0) for r in rows)
    return dict(decode_tokens=tokens, decode_seconds=seconds,
                second_pass_rate=sum(n > 1 for n in passes) / verifies if verifies and passes else None,
                gate_checks=gate_checks, gate_passes=gate_passes,
                gate_pass_rate=gate_passes / gate_checks if gate_checks else None,
                shift_clipped=sum(r.get("prefix_decision_counts", {}).get("shift_clipped", 0) for r in rows),
                mean_applied_cut=sum(r.get("prefix_decision_counts", {}).get("applied_cut_sum", 0) for r in rows) / gate_passes if gate_passes else None,
                tok_s=tokens / seconds if seconds else 0,
                ms_token=1000 * seconds / tokens if tokens else None,
                verify_calls=verifies, verifies_per_100_tokens=100 * verifies / tokens if tokens else None,
                mean_accepted_draft=accepted / verifies if verifies else None,
                mean_committed=sum(sum(r["committed_lengths"]) for r in rows) / verifies if verifies else None,
                draft_calls=sum(r["draft_calls"] for r in rows),
                mean_retained=sum(retained) / len(retained) if retained else None,
                ar_match=sum(r["ar_match"] for r in rows), samples=len(rows))


def main():
    args = parse_args()
    import torch
    from transformers import AutoTokenizer
    from eval.adaptive_linear_decode import PrefixPolicy, adaptive_dflash_generate
    from eval.benchmark import _encode_turn, _load_models, load_and_process_dataset

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This throughput benchmark requires a CUDA GPU; CPU tests are in tests/.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.model_dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    outdir = args.output_dir or ROOT / "results" / ("adaptive_qwen3_4b_" + time.strftime("%Y%m%d_%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=False)  # never silently overwrite a run
    if args.data_file:
        data = [json.loads(line) for line in args.data_file.read_text().splitlines() if line.strip()]
    else:
        data = load_and_process_dataset(args.dataset)
    order = list(range(len(data)))
    random.Random(args.seed).shuffle(order)
    needed = args.smoke_samples + args.tune_samples + args.eval_samples
    if len(order) < needed:
        raise ValueError(f"Need {needed} distinct samples, got {len(order)}")
    smoke = order[:args.smoke_samples]
    tune = order[args.smoke_samples:args.smoke_samples + args.tune_samples]
    confirm = order[args.smoke_samples + args.tune_samples:needed]
    manifest = dict(vars(args), torch=torch.__version__, gpu=torch.cuda.get_device_name(device),
                    transformers=__import__("transformers").__version__,
                    sample_ids=dict(smoke=smoke, tune=tune, confirm=confirm),
                    timing="decode excludes target prefill/first token; includes draft context prefill",
                    git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (outdir / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2))
    target, draft = _load_models(SimpleNamespace(model=args.model, draft_model=args.draft_model,
                                                        target_attention=args.target_attention, model_dtype=args.model_dtype), "dflash", device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stop_ids = [tokenizer.eos_token_id]
    prompts = {}
    prompt_hashes = {}
    for idx in order[:needed]:
        row = data[idx]
        prompt = row["prompt"] if "prompt" in row else row["turns"][0]
        prompts[idx] = _encode_turn(tokenizer, [{"role": "user", "content": prompt}], device)
        prompt_hashes[idx] = hashlib.sha256(prompt.encode()).hexdigest()
    (outdir / "prompt_hashes.json").write_text(json.dumps(prompt_hashes, indent=2))
    common = dict(max_draft_passes=args.max_draft_passes, min_continue_tokens=args.min_continue_tokens,
                  max_candidates=args.max_candidates, total_logp_budget=args.total_logp_budget)
    configs = experiment_configs(args)
    (outdir / "configs.json").write_text(json.dumps({k: asdict(v) for k, v in configs.items()}, indent=2))

    def generate(idx, policy, length=None, trace=False):
        return adaptive_dflash_generate(draft if policy is not None else None, target, prompts[idx],
                                        length or args.max_new_tokens, stop_ids, policy, trace)

    # Warm AR and every candidate strategy before ANY measured run.
    generate(smoke[0], None, min(64, args.max_new_tokens))
    generate(smoke[0], PrefixPolicy(mode="original_loop", loop_gate=-1e9, **common),
             min(64, args.max_new_tokens))
    for policy in configs.values():
        generate(smoke[0], policy, min(64, args.max_new_tokens))

    reference = {}
    diagnostic_done = set()
    def run(name, policy, ids, phase, repeat=0, trace=False):
        rows = []
        for idx in ids:
            r = generate(idx, policy, trace=trace)
            ids_out = r.output_ids[0, r.num_input_tokens:].tolist()
            if policy is None:
                reference[idx] = ids_out
            matches = reference[idx] == ids_out
            row = dict(sample_id=idx, phase=phase, config=name, repeat=repeat,
                       policy=asdict(policy) if policy else None, output_ids=ids_out,
                       ar_match=matches, decode_tokens=r.decode_tokens, decode_seconds=r.decode_seconds,
                       time_to_first_token=r.time_to_first_token, verify_calls=r.verify_calls,
                       draft_calls=r.draft_calls, acceptance_lengths=r.acceptance_lengths,
                       committed_lengths=r.committed_lengths, verify_lengths=r.verify_lengths,
                       retained_lengths=r.retained_lengths, draft_pass_counts=r.draft_pass_counts,
                       prefix_decision_counts=r.prefix_decision_counts)
            if trace:
                row["round_traces"] = r.round_traces
            with (outdir / ("traces.jsonl" if trace else "responses.jsonl")).open("a") as f:
                f.write(json.dumps(row) + "\n")
            if not matches:
                from eval.ar_diagnostics import diagnose_mismatch, mismatch_summary
                detail = mismatch_summary(reference[idx], ids_out, tokenizer)
                print(f"AR mismatch: {phase}/{name}/sample {idx}: {detail}", flush=True)
                should_diagnose = args.mismatch_diagnostics != "off" and (
                    not diagnostic_done if args.mismatch_diagnostics == "first" else name not in diagnostic_done)
                if should_diagnose:
                    diagnostic_done.add(name)
                    diagnostic_path = outdir / f"mismatch_{phase}_{name}_{idx}.json"
                    try:
                        detail = diagnose_mismatch(target, draft, prompts[idx], reference[idx], ids_out,
                                                   policy, args.max_new_tokens, stop_ids, tokenizer)
                    except Exception as exc:
                        detail["diagnostic_error"] = repr(exc)
                    diagnostic_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False))
                    print(f"Untimed mismatch diagnostic saved: {diagnostic_path}", flush=True)
                if args.ar_check == "strict":
                    raise RuntimeError(f"AR mismatch: {phase}/{name}/sample {idx}; see mismatch_*.json in {outdir}")
            rows.append(row)
        summary = dict(phase=phase, config=name, repeat=repeat,
                       cut_offset=policy.cut_offset if policy and policy.mode == "gated_prefix_loop" else None,
                       min_confident_ratio=policy.min_confident_ratio if policy and policy.mode == "gated_prefix_loop" else None,
                       **summarize(rows))
        with (outdir / ("trace_summaries.jsonl" if trace else "summaries.jsonl")).open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"{phase:8s} {name:45s} {summary['tok_s']:8.2f} tok/s  AR {summary['ar_match']}/{len(rows)}", flush=True)
        return summary

    # Fail early on token mismatch, not after a long timing sweep.
    run("target_ar", None, smoke, "smoke")
    for name, policy in configs.items():
        run(name, policy, smoke, "smoke")
    run("target_ar", None, tune, "tune")
    names = list(configs)
    random.Random(args.seed).shuffle(names)
    scores = {name: run(name, configs[name], tune, "tune") for name in names}
    selected = select_confirmation_configs(configs, scores, args.experiment)
    (outdir / "selected.json").write_text(json.dumps(selected, indent=2))
    for rep in range(args.repeats):
        # Establish AR outputs before shuffled measured strategy runs.
        run("target_ar", None, confirm, "confirm", rep)
        names = list(selected.values())
        random.Random(args.seed + rep).shuffle(names)
        for name in names:
            run(name, configs[name], confirm, "confirm", rep)
    for mode, name in selected.items():
        if args.trace_samples:
            run(name, configs[name], confirm[:args.trace_samples], "diagnose", trace=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_adaptive_results.py"), str(outdir)], check=True)
    print(f"Results: {outdir}")


if __name__ == "__main__":
    main()
