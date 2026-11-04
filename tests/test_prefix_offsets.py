import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eval.adaptive_linear_decode import PrefixPolicy, gated_prefix_decision, adaptive_dflash_generate
from scripts.run_adaptive_qwen3_4b_smoke import experiment_configs, select_confirmation_configs, parse_args, full_width_parity
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

    def test_integer_gates_cut_at_actual_prefix(self):
        for minimum in (15, 13, 12):
            for confident in (11, 12, 13, 14, 15):
                with self.subTest(minimum=minimum, confident=confident):
                    lp = torch.tensor([-.001] * confident + [-3.] * (15 - confident))
                    policy = PrefixPolicy(mode="gated_prefix_loop", threshold=-2.,
                                          min_confident_tokens=minimum, cut_offset=0)
                    keep, go, raw, _ = gated_prefix_decision(lp, policy, 30, 0)
                    self.assertEqual(raw, confident)
                    self.assertEqual(go, confident >= minimum)
                    self.assertEqual(keep, confident if go else 15)

    def test_integer_full_width_matches_original_with_passing_and_failing_gates(self):
        prompt = torch.tensor([[1, 2, 3]])
        for confident in ((12, 15), (13, 15), (15, 15)):
            with self.subTest(confident=confident):
                a = adaptive_dflash_generate(Draft(confident=confident), Target(), prompt, 100, [],
                    PrefixPolicy(mode="gated_prefix_loop", threshold=-2., min_confident_tokens=15))
                b = adaptive_dflash_generate(Draft(confident=confident), Target(), prompt, 100, [],
                    PrefixPolicy(mode="original_loop", loop_gate=-2.))
                self.assertTrue(torch.equal(a.output_ids, b.output_ids))
                for field in ("retained_lengths", "verify_lengths", "draft_pass_counts",
                              "draft_calls", "verify_calls", "acceptance_lengths"):
                    self.assertEqual(getattr(a, field), getattr(b, field), field)

    def test_integer_gate_validation(self):
        for minimum in (0, -1, 12.5):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                PrefixPolicy(min_confident_tokens=minimum)
        with self.assertRaises(ValueError):
            gated_prefix_decision(torch.zeros(15), self.policy(min_confident_tokens=16), 30, 0)

    def test_all_integer_gates_reach_confirmation_with_same_baseline(self):
        with patch("sys.argv", ["runner", "--experiment", "prefix_lengths"]):
            args = parse_args()
        configs = experiment_configs(args)
        self.assertEqual(len(configs), 5)
        original = [p for p in configs.values() if p.mode == "original_loop"]
        self.assertEqual([p.loop_gate for p in original], [-2.])
        selected = select_confirmation_configs(configs, {n: {"tok_s": i} for i, n in enumerate(configs)},
                                               "prefix_lengths")
        gated = [configs[n] for n in selected.values() if configs[n].mode == "gated_prefix_loop"]
        self.assertEqual([p.min_confident_tokens for p in gated], [15, 13, 12])
        self.assertTrue(all(p.cut_offset == 0 and p.threshold == -2. for p in gated))

    def test_parity_reporting_ignores_timing_but_detects_schedule_changes(self):
        original = dict(phase="confirm", repeat=0, sample_id=1, config="original_g-2.0",
                        policy=dict(mode="original_loop", loop_gate=-2.), decode_seconds=1.,
                        output_ids=[1, 2], draft_calls=2, verify_calls=1, verify_lengths=[31],
                        retained_lengths=[15, 15], draft_pass_counts=[2], acceptance_lengths=[2])
        gated = dict(original, config="gated_n15", decode_seconds=2.,
                     policy=dict(mode="gated_prefix_loop", min_confident_tokens=15, threshold=-2.))
        self.assertTrue(full_width_parity([gated, original], 15)[0]["matches"])
        gated["verify_lengths"] = [16]
        self.assertEqual(full_width_parity([gated, original], 15)[0]["differing_fields"], ["verify_lengths"])


if __name__ == "__main__":
    unittest.main()
