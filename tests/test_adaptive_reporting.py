import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_adaptive_qwen3_4b_smoke import summarize
from scripts.summarize_adaptive_results import summarize_directory


class ReportingTests(unittest.TestCase):
    def test_throughput_is_ratio_of_totals(self):
        base = dict(verify_calls=1, acceptance_lengths=[1], committed_lengths=[2],
                    retained_lengths=[2], draft_calls=1, ar_match=True)
        result = summarize([dict(base, decode_tokens=10, decode_seconds=1),
                            dict(base, decode_tokens=10, decode_seconds=9)])
        self.assertEqual(result["tok_s"], 2)
        self.assertEqual(result["verifies_per_100_tokens"], 10)
        self.assertEqual(result["mean_accepted_draft"], 1)
        self.assertEqual(result["ar_match"], 2)

    def test_confirmation_excludes_tuning_and_uses_median(self):
        rows = []
        for name, speeds in (("target_ar", (5, 5, 5)), ("original_g-1", (10, 12, 11)),
                             ("adaptive_loop_prefix_logp_-1", (12, 15, 13))):
            for speed in speeds:
                rows.append(dict(config=name, phase="confirm", tok_s=speed,
                                 verifies_per_100_tokens=5, mean_accepted_draft=6,
                                 mean_committed=7, ar_match=16, samples=16))
        rows.append(dict(config="adaptive_loop_prefix_logp_-1", phase="tune", tok_s=999))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "summaries.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            result = {r["config"]: r for r in summarize_directory(path)}
        adaptive = result["adaptive_loop_prefix_logp_-1"]
        self.assertEqual(adaptive["median_tok_s"], 13)
        self.assertEqual(adaptive["speedup_vs_ar"], 2.6)
        self.assertAlmostEqual(adaptive["gain_vs_original_pct"], 100 * (13 / 11 - 1))
        self.assertEqual(adaptive["ar_match"], 48)


if __name__ == "__main__":
    unittest.main()
