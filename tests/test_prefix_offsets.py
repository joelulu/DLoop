import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eval.adaptive_linear_decode import PrefixPolicy, gated_prefix_decision, adaptive_dflash_generate
from scripts.run_adaptive_qwen3_4b_smoke import experiment_configs, select_confirmation_configs, parse_args
from test_adaptive_decode import Draft, Target


class OffsetTests(unittest.TestCase):
    def policy(self, **kwargs):
        return replace(PrefixPolicy(mode="gated_prefix_loop", rule="token_prob", threshold=.9), **kwargs)

    def test_seven_offsets_at_twelve(self):
        lp = torch.tensor([-.001] * 12 + [-2.] * 3)
        for offset, expected in ((-3, 9), (-2, 10), (-1, 11), (0, 12), (1, 13), (2, 14), (4, 15)):
            with self.subTest(offset=offset):
                keep, go, raw, _ = gated_prefix_decision(lp, self.policy(cut_offset=offset), 30, 0)
                self.assertEqual((keep, go, raw), (expected, True, 12))

    def test_gate_precedes_offset(self):
        lp = torch.tensor([-.001] * 11 + [-2.] * 4)
        keep, go, raw, _ = gated_prefix_decision(lp, self.policy(cut_offset=4), 30, 0)
        self.assertEqual((keep, go, raw), (15, False, 11))

    def test_ratio_ceil_and_clamp(self):
        lp = torch.tensor([-.001] * 12 + [-2.] * 3)
        for ratio in (.75, .8):
            self.assertTrue(gated_prefix_decision(lp, self.policy(min_confident_ratio=ratio), 30, 0)[1])
        self.assertFalse(gated_prefix_decision(lp, self.policy(min_confident_ratio=.81), 30, 0)[1])
        self.assertEqual(gated_prefix_decision(lp, self.policy(cut_offset=-99), 30, 0)[0], 1)
        self.assertEqual(gated_prefix_decision(lp, self.policy(cut_offset=99), 30, 0)[0], 15)

    def test_final_and_tail_preserve_full(self):
        lp = torch.tensor([-2.] * 15)
        self.assertEqual(gated_prefix_decision(lp, self.policy(cut_offset=-3), 30, 1)[:2], (15, False))
        self.assertEqual(gated_prefix_decision(lp, self.policy(cut_offset=-3), 5, 0)[:2], (5, False))

    def test_failed_gate_matches_single(self):
        prompt = torch.tensor([[1, 2, 3]])
        a = adaptive_dflash_generate(Draft(confident=(10,)), Target(), prompt, 48, [], self.policy(), trace=True)
        b = adaptive_dflash_generate(Draft(confident=(10,)), Target(), prompt, 48, [], PrefixPolicy(mode="single"))
        self.assertTrue(torch.equal(a.output_ids, b.output_ids))
        self.assertEqual(a.verify_lengths, b.verify_lengths)
        self.assertEqual(a.draft_calls, a.verify_calls)
        self.assertEqual(a.retained_lengths[0], 15)

    def test_real_loop_path_for_each_offset(self):
        prompt = torch.tensor([[1, 2, 3]])
        reference = adaptive_dflash_generate(None, Target(), prompt, 60, []).output_ids
        for offset in (-3, -2, -1, 0, 1, 2, 4):
            with self.subTest(offset=offset):
                r = adaptive_dflash_generate(Draft(confident=(12, 8)), Target(), prompt, 60, [],
                                             self.policy(cut_offset=offset), trace=True)
                cut = min(15, 12 + offset)
                self.assertEqual(r.round_traces[0]["retained"], [cut, 15])
                self.assertTrue(torch.equal(r.output_ids, reference))
                self.assertEqual(r.round_traces[0]["prefix_decisions"][0]["raw_confident"], 12)
                self.assertEqual(r.prefix_decision_counts["shift_clipped"] > 0, offset == 4)

    def test_full_confidence_offset_zero_matches_original(self):
        prompt = torch.tensor([[1, 2, 3]])
        a = adaptive_dflash_generate(Draft(confident=(15,)), Target(), prompt, 70, [], self.policy())
        b = adaptive_dflash_generate(Draft(confident=(15,)), Target(), prompt, 70, [],
                                     PrefixPolicy(mode="original_loop", loop_gate=-2.))
        self.assertTrue(torch.equal(a.output_ids, b.output_ids))
        self.assertEqual(a.retained_lengths, b.retained_lengths)
        self.assertEqual(a.verify_lengths, b.verify_lengths)

    def test_validation(self):
        for kwargs in ({"min_confident_ratio": 0}, {"min_confident_ratio": 1.1},
                       {"cut_offset": .5}, {"total_logp_budget": -2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.policy(**kwargs)

    def test_all_offsets_reach_confirmation(self):
        args = SimpleNamespace(experiment="prefix_offsets", max_draft_passes=2, min_continue_tokens=1,
                               max_candidates=30, total_logp_budget=None, min_confident_ratio=.8,
                               prefix_threshold=-2., cut_offsets=[-3, -2, -1, 0, 1, 2, 4])
        configs = experiment_configs(args)
        self.assertEqual(len(configs), 11)  # single + 3 original gates + 7 offsets
        scores = {name: {"tok_s": i} for i, name in enumerate(configs)}
        selected = select_confirmation_configs(configs, scores, "prefix_offsets")
        offsets = [configs[name].cut_offset for name in selected.values()
                   if configs[name].mode == "gated_prefix_loop"]
        self.assertEqual(offsets, args.cut_offsets)
        self.assertEqual(len(selected), 9)

    def test_cli_parses_negative_offsets(self):
        with patch("sys.argv", ["runner", "--experiment", "prefix_offsets", "--cut-offsets",
                                "-3", "-2", "-1", "0", "1", "2", "4", "--ar-check", "warn",
                                "--mismatch-diagnostics", "off"]):
            args = parse_args()
        self.assertEqual(args.cut_offsets, [-3, -2, -1, 0, 1, 2, 4])
        self.assertEqual(args.mismatch_diagnostics, "off")


if __name__ == "__main__":
    unittest.main()
