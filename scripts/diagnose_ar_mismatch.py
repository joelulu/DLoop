"""Replay the first saved mismatch without repeating a threshold sweep."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target-attention", choices=["sdpa", "eager"], default=None)
    p.add_argument("--model-dtype", choices=["bfloat16", "float32"], default=None)
    p.add_argument("--saved-only", action="store_true", help="print stored first difference; no GPU/model loading")
    args = p.parse_args()
    from eval.ar_diagnostics import mismatch_summary
    manifest = json.loads((args.result_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in (args.result_dir / "responses.jsonl").read_text().splitlines()]
    mismatch = next((row for row in rows if not row["ar_match"]), None)
    if mismatch is None:
        raise SystemExit("No saved AR mismatch found")
    idx = mismatch["sample_id"]
    reference = next(row["output_ids"] for row in reversed(rows)
                     if row["config"] == "target_ar" and row["sample_id"] == idx
                     and row["phase"] == mismatch["phase"] and row["repeat"] == mismatch["repeat"])
    saved = mismatch_summary(reference, mismatch["output_ids"])
    print(json.dumps(dict(sample_id=idx, config=mismatch["config"], **saved), indent=2), flush=True)
    if args.saved_only:
        return

    import torch
    from transformers import AutoTokenizer
    from eval.adaptive_linear_decode import PrefixPolicy, adaptive_dflash_generate
    from eval.ar_diagnostics import diagnose_mismatch
    from eval.benchmark import _encode_turn, _load_models, load_and_process_dataset
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU replay needs CUDA; --saved-only works without a GPU")
    torch.cuda.set_device(device)
    torch.manual_seed(manifest.get("seed", 42))
    torch.cuda.manual_seed_all(manifest.get("seed", 42))
    dtype = args.model_dtype or manifest.get("model_dtype", "bfloat16")
    attention = args.target_attention or manifest.get("target_attention", "sdpa")
    if dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    load_args = SimpleNamespace(model=manifest["model"], draft_model=manifest["draft_model"],
                                target_attention=attention, model_dtype=dtype)
    target, draft = _load_models(load_args, "dflash", device)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"])
    if manifest.get("data_file"):
        data = [json.loads(line) for line in Path(manifest["data_file"]).read_text().splitlines() if line.strip()]
    else:
        data = load_and_process_dataset(manifest["dataset"])
    prompt = data[idx]["prompt"] if "prompt" in data[idx] else data[idx]["turns"][0]
    input_ids = _encode_turn(tokenizer, [{"role": "user", "content": prompt}], device)
    policy = PrefixPolicy(**mismatch["policy"])
    max_new = manifest["max_new_tokens"]
    stops = [tokenizer.eos_token_id]
    ar = adaptive_dflash_generate(None, target, input_ids, max_new, stops)
    observed = adaptive_dflash_generate(draft, target, input_ids, max_new, stops, policy)
    ar_ids = ar.output_ids[0, input_ids.shape[1]:].tolist()
    observed_ids = observed.output_ids[0, input_ids.shape[1]:].tolist()
    report = diagnose_mismatch(target, draft, input_ids, ar_ids, observed_ids,
                               policy, max_new, stops, tokenizer)
    report.update(sample_id=idx, config=mismatch["config"], target_attention=attention,
                  model_dtype=dtype, saved_difference=saved,
                  ar_matches_saved_reference=ar_ids == reference,
                  saved_mismatch_reproduced=observed_ids == mismatch["output_ids"])
    path = args.result_dir / f"diagnostic_{attention}_{dtype}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "rounds"}, indent=2, ensure_ascii=False))
    print(f"Untimed diagnostic: {path}")


if __name__ == "__main__":
    main()
