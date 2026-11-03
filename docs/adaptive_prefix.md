# Qwen3-4B adaptive-prefix inference experiment

This is an inference-only prototype. It uses the released
`naver-ai/DLoop-DFlash-Qwen3-4B` drafter with `Qwen/Qwen3-4B`.
No Qwen3-4B GPU speedup has been measured by the implementation author.

## Gated boundary-offset sweep (new experiment)

`--experiment prefix_offsets` evaluates all seven requested offsets against the
same held-out prompts. It keeps `single` as a control, tunes the original DLoop
gate on the tuning split, and confirms that baseline plus EVERY offset. Offsets
are not collapsed to a single tuning winner.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_adaptive_qwen3_4b_smoke.py \
  --model /home/jovyan/LMM/lmm_model/Qwen3-4B \
  --draft-model /home/jovyan/LMM/lmm_model/Loop-DFlash-Qwen3-4B \
  --data-file "$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/eval_datasets/gsm8k.jsonl" \
  --experiment prefix_offsets \
  --min-confident-ratio 0.8 --prefix-threshold -2 \
  --cut-offsets -3 -2 -1 0 1 2 4 \
  --max-draft-passes 2 \
  --smoke-samples 1 --tune-samples 4 --eval-samples 16 \
  --max-new-tokens 256 --repeats 1 \
  --ar-check warn --mismatch-diagnostics off
```

For a larger confirmation use `--eval-samples 64 --repeats 3`. You can add
`--trace-samples 2` for separate diagnostic reruns.

The rule is:

1. Find the longest prefix k whose cumulative draft log probability >= -2.
2. Gate continuation using **unshifted** k: k >= ceil(0.8 * 15) = 12.
3. If the gate fails, verify the full original proposal without another draft.
4. If the gate passes, retain clamp(k + offset, 1, 15), use its last token as
   the next anchor, and restart drafting at the following position.
5. The final pass retains its full available proposal for verification. Offsets
   change continuation boundaries, not final verification lengths.

For k=12, offsets -3,-2,-1,0,+1,+2,+4 retain 9,10,11,12,13,14,15 tokens.
The +4 request clips to 15 (actual shift +3). Positive offsets deliberately allow
some tokens beyond the confidence boundary; negative offsets back off from it.
The ratio is not rechecked after shifting. At candidate width 15, ratios .75 and
.8 both require 12 tokens after rounding up, so those two ratios are identical.
When k=15 and offset=0, continuation and the final full block reproduce the
original whole-block DLoop behavior (with the same confidence threshold/limits).
Negative offsets also apply when k=15; positive offsets then clip at 15.
Near the output/candidate limit, a full available proposal is verified directly
instead of lowering the ratio denominator to force an extra short draft.

`comparison.csv` includes `cut_offset`, `min_confident_ratio`, `second_pass_rate`,
`gate_pass_rate`, `mean_applied_cut`, and `shift_clipped`. `responses.jsonl`
records per-request draft-pass counts and gate/clip counters. Detailed traces
record the raw confidence boundary separately from the applied cut.

The older `--experiment legacy` remains available and is the default. It uses
the previous truncation rules, including its final-pass truncation, so it should
not be confused with the new gated experiment.

## Setup

Use the upstream CUDA environment (tested upstream with torch 2.8/2.9).
Keep the upstream-tested Transformers version for the existing model classes:

```bash
git fetch origin
git switch research/adaptive-prefix-loop
python -m pip install -e . 'transformers==5.8.1'
python -m unittest discover -s tests -v
```

The tests run on CPU and require no checkpoint downloads. They include a tiny
randomly initialized real Qwen3 target + real DFlash backbone, as well as
scripted tests forcing 10+8 retention and checking carrier/position alignment.
They are correctness checks, not evidence of Qwen3-4B model quality or speed.

## Fast GPU screening

Run from the repository root. A single GPU holds target and draft; GPU index is
relative to CUDA_VISIBLE_DEVICES. Models are loaded only once.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_adaptive_qwen3_4b_smoke.py \
  --device cuda:0 \
  --smoke-samples 1 --tune-samples 4 --eval-samples 16 \
  --max-new-tokens 256 --repeats 1
```

Larger confirmation (3 smoke + 16 tuning + 64 separate held-out samples,
3 repeats; GSM8K by default):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_adaptive_qwen3_4b_smoke.py \
  --device cuda:0 --trace-samples 2
```

Use local checkpoint directories if already downloaded:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_adaptive_qwen3_4b_smoke.py \
  --model "$HOME/LMM/lmm_model/Qwen3-4B" \
  --draft-model "$HOME/LMM/lmm_model/DLoop-DFlash-Qwen3-4B" \
  --smoke-samples 1 --tune-samples 4 --eval-samples 16 --repeats 1
```

`--dataset math500` and `--dataset mbpp` reuse the upstream prompts. For offline
data use `--data-file /path/data.jsonl`, with either `{"prompt": "..."}` or
`{"turns": ["..."]}` per line. This script evaluates the first turn only.
At least smoke+tune+eval distinct rows are required. Prompts use upstream
`enable_thinking=False`. Default seed is 42.

## Methods and controls

All four modes use the **same checkpoint**, common causal verifier, EOS handling,
and output cap. Every draft forward keeps the checkpoint's original width:
block_size=16 means 1 anchor plus 15 speculative tokens. Candidate count limits
exclude the anchor. `max_draft_passes=2` is the TOTAL number of draft forwards,
equivalent to upstream `stage_cap=1` extra stage.

| Mode | Action |
| --- | --- |
| single | One full draft, then verify |
| original_loop | Whole-block confidence gate, append full blocks, then verify |
| truncate_verify | Keep a confidence-selected prefix, immediately verify |
| adaptive_loop | Keep a confidence-selected prefix, restart at that boundary, repeat, then verify |

`original_loop` reproduces upstream gate and carrier layout, but uses the common
verifier/timer. The original `legacy` code path is unchanged. Candidate/output
caps may shorten the final round in the common path; upstream does not apply the
same final-round cap before verification.

Two prefix rules are swept:

- prefix_logp: longest prefix with cumulative draft log probability >= threshold
  (-0.5, -1, -2).
- token_prob: prefix ending before the first token below threshold
  (0.8, 0.9, 0.95). No skipping uncertain positions.

Original DLoop gates are swept over -0.5, -1, -2 as well. Best settings for each
method are selected on the tuning split ONLY. All selected methods are then
evaluated on the same held-out split in shuffled method order. Target-only AR is
run for each sample, and a token mismatch aborts the run after saving the row.
Do not interpret very small screening splits as a reliable final speedup claim.

First-pass zero retention falls back to verifying the full original proposal.
Later-pass zero retention adds nothing and verifies the accumulated proposals.
`--min-continue-tokens` stops continuation after a short retained segment.
`--max-candidates` limits the accumulated proposal count (default 30).
Optional `--total-logp-budget -2` limits cumulative retained log probability;
the zero-prefix fallback still takes priority so generation always progresses.

Retained token prefixes and hidden carriers are sliced together. The retained
last token is the next stage's unverified anchor; its carrier is omitted exactly
as in upstream DLoop. Draft cache temporaries are discarded back to the verified
context boundary. No target forward occurs between draft passes. Because draft
attention is bidirectional, slicing saved hidden states does NOT recompute them
as if the discarded mask slots had never existed.

## Output and timing

Each run creates a new `results/adaptive_qwen3_4b_TIMESTAMP/` directory:

- `manifest.json`: arguments, versions, GPU, commit SHA, disjoint sample IDs.
- `configs.json`, `selected.json`, `prompt_hashes.json`: reproducibility metadata.
- `responses.jsonl`: per-sample outputs, timings, actual draft/verify counts.
- `summaries.jsonl`: smoke, tuning and held-out summaries (explicit phase labels).
- `comparison.csv`: held-out median throughput, repeat range, gain over original
  DLoop, speedup over AR, verification frequency, accepted drafts, AR matches.
- `traces.jsonl`: optional separate diagnostic reruns with retained segment
  lengths, effective acceptance, second-segment acceptance and stopping reason.

Decode timing excludes target prefill and the first generated token, and includes
the first draft context-prefill cost. Throughput is total **actual decode tokens
divided by total elapsed decode seconds**, not an average of request rates.
Accepted draft tokens exclude the bonus; committed tokens include it. EOS/output
caps are applied before metrics. Verification widths include the anchor.
Every mode includes normal Python control/selection overhead. Detailed traces
are disabled in timing runs; diagnostic reruns never enter comparison.csv.

The main question is whether reduced target calls outweigh extra draft work.
Longer retained blocks or acceptance alone do not establish a speedup.

Rebuild a summary with:

```bash
python scripts/summarize_adaptive_results.py results/your_run_directory
```

## Existing benchmark entry point

The original default remains `--decode-mode legacy`. For the new modes:

```bash
python -m eval.benchmark --model Qwen/Qwen3-4B --dataset gsm8k \
  --max-samples 16 --max-new-tokens 256 --baseline --decode-mode single
python -m eval.benchmark --model Qwen/Qwen3-4B \
  --draft-model naver-ai/DLoop-DFlash-Qwen3-4B --dataset gsm8k \
  --max-samples 16 --max-new-tokens 256 --decode-mode adaptive_loop \
  --prefix-rule prefix_logp --prefix-threshold -1 --max-draft-passes 2
```

New modes use a separate baseline-cache namespace so upstream timing conventions
are not silently mixed. The standalone screening script above is the preferred
path for threshold selection and matched, repeated comparisons. Random sampling,
tree verification, DSpark, Domino and batched serving are outside this prototype.

## Diagnosing an AR mismatch

Strict AR comparison remains the default. On a mismatch the runner now prints
the first differing token/context and writes an untimed `mismatch_*.json` report
before stopping. A greedy block-verification output can differ from incremental
BF16 AR because the numerical execution shapes differ, but that possibility
must not be used to assume away a cache or indexing bug.

Replay an existing failed run (the original responses and manifest suffice):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_ar_mismatch.py results/your_run_directory
```

It compares the first shared-prefix divergence, records verifier top-2 logits,
replays the common prefix with token-by-token AR and full-prefix evaluation,
and checks the upstream decoder when the failing mode is `single`. The report
also states whether the saved failure reproduces. No diagnostic timings are
included in the performance comparison.

Optional numerical sensitivity checks, each using the same saved sample:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_ar_mismatch.py results/your_run_directory \
  --target-attention eager
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_ar_mismatch.py results/your_run_directory \
  --target-attention eager --model-dtype float32
```

FP32 applies to both target and draft and disables TF32; it needs more memory.
These are diagnostic runs, not equivalent-backend speed comparisons.
`--saved-only` prints stored token IDs/lengths without loading any models.
The main runner also accepts `--target-attention` and `--model-dtype`.

For explicitly exploratory measurements, `--ar-check warn` continues after
mismatches while preserving AR-match counts and diagnostic reports. If held-out
outputs differ, `comparison_valid=false` and the lossless speedup columns are
suppressed; only separately labeled exploratory throughput ratios are reported.
