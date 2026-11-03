import unittest

from eval.ar_diagnostics import diagnose_mismatch, first_difference, mismatch_summary
from eval.adaptive_linear_decode import PrefixPolicy, adaptive_dflash_generate
from test_adaptive_decode import Draft, Target
import torch


class MismatchTests(unittest.TestCase):
    def test_first_difference_and_length_only(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))
        self.assertEqual(first_difference([1, 2], [1, 3]), 1)
        self.assertEqual(first_difference([1, 2], [1]), 1)
        self.assertIsNone(mismatch_summary([1, 2], [1])["observed_token"])

    def test_replay_reports_shared_prefix_and_reproduction(self):
        prompt = torch.tensor([[1, 2, 3]])
        reference = adaptive_dflash_generate(None, Target(), prompt, 12, []).output_ids[0, 3:].tolist()
        observed = reference.copy()
        observed[1] = 8
        report = diagnose_mismatch(Target(), Draft(), prompt, reference, observed,
                                    PrefixPolicy(mode="single"), 12, [])
        self.assertEqual(report["first_difference"], 1)
        self.assertFalse(report["rerun_matches_observed"])
        self.assertIsNone(report["rerun_difference"]["first_difference"])
        self.assertEqual(report["ar_shared_prefix_top2"]["ids"][0], 5)
        self.assertEqual(report["full_shared_prefix_top2"]["ids"][0], 5)
        self.assertEqual(report["block_verifier_at_difference"]["ids"][0], 5)


if __name__ == "__main__":
    unittest.main()
