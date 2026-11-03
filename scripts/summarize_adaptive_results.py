"""Summarize held-out runs; tuning results are never presented as confirmation."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize_directory(directory):
    groups = defaultdict(list)
    for line in (directory / "summaries.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["phase"] == "confirm":
            groups[row["config"]].append(row)
    result = []
    comparison_valid = all(r["ar_match"] == r["samples"] for rows in groups.values() for r in rows)
    baseline = next((statistics.median(r["tok_s"] for r in rows)
                     for name, rows in groups.items() if name.startswith("original_")), None)
    ar = statistics.median(r["tok_s"] for r in groups.get("target_ar", [])) if groups.get("target_ar") else None
    for name, rows in groups.items():
        speed = statistics.median(r["tok_s"] for r in rows)
        def mean(key):
            values = [r[key] for r in rows if r.get(key) is not None]
            return statistics.mean(values) if values else None
        result.append(dict(config=name, cut_offset=rows[0].get("cut_offset"),
                           min_confident_ratio=rows[0].get("min_confident_ratio"),
                           second_pass_rate=mean("second_pass_rate"), gate_pass_rate=mean("gate_pass_rate"),
                           mean_applied_cut=mean("mean_applied_cut"), shift_clipped=mean("shift_clipped"),
                           repeats=len(rows), median_tok_s=speed,
                           min_tok_s=min(r["tok_s"] for r in rows), max_tok_s=max(r["tok_s"] for r in rows),
                           comparison_valid=comparison_valid,
                           speedup_vs_ar=speed / ar if ar and comparison_valid else None,
                           gain_vs_original_pct=100 * (speed / baseline - 1) if baseline and comparison_valid else None,
                           exploratory_gain_vs_original_pct=100 * (speed / baseline - 1) if baseline else None,
                           verifies_per_100_tokens=mean("verifies_per_100_tokens"),
                           mean_accepted_draft=mean("mean_accepted_draft"), mean_committed=mean("mean_committed"),
                           ar_match=sum(r["ar_match"] for r in rows), samples=sum(r["samples"] for r in rows)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    result = summarize_directory(args.directory)
    if not result:
        raise SystemExit("No held-out confirmation results yet.")
    with (args.directory / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    for row in result:
        gain = row["gain_vs_original_pct"]
        gain_text = f"{gain:+.2f}%" if gain is not None else "n/a"
        print(f"{row['config']:45s} {row['median_tok_s']:8.2f} tok/s  vs DLoop {gain_text}")
    if not all(row["comparison_valid"] for row in result):
        print("WARNING: AR mismatches observed. comparison_valid=false; lossless speedup columns suppressed. "
              "Exploratory throughput is not a matched-output speedup claim.")


if __name__ == "__main__":
    main()
