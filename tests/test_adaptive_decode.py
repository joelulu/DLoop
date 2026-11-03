"""CPU integration tests with deterministic causal target and scripted drafter.

The target enforces cache continuity. The drafter validates carrier positions
and uses low confidence at known bad slots, so truncation really changes output.
Run: python -m unittest discover -s tests -v
"""

import importlib.util
import importlib
import math
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

import torch

from eval.adaptive_linear_decode import PrefixPolicy, adaptive_dflash_generate, select_prefix


VOCAB = 64


class Target(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(embed_tokens=lambda ids: ids[..., None].float())
        self.verify_widths = []

    def lm_head(self, hidden):
        logits = torch.zeros((*hidden.shape[:2], VOCAB))
        return logits.scatter_(-1, hidden[..., :1].long(), hidden[..., 1:2])

    def forward(self, ids, position_ids, past_key_values, use_cache,
                logits_to_keep=None, output_hidden_states=False):
        n = ids.shape[1]
        assert past_key_values.get_seq_length() == int(position_ids[0, 0])
        kv = torch.zeros(1, 1, n, 1)
        past_key_values.update(kv, kv, 0)
        hidden = torch.stack([((ids + 1) % VOCAB).float(), torch.full_like(ids, 12).float(),
                              position_ids.float()], -1)
        logits = self.lm_head(hidden)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        self.verify_widths.append(n)
        return SimpleNamespace(logits=logits, hidden_states=(hidden, hidden))


class Draft(torch.nn.Module):
    target_layer_ids = [0]
    mask_token_id = 63

    def __init__(self, confident=(10, 8), block_size=16, corrupt=True):
        super().__init__()
        self.confident = confident
        self.block_size = block_size
        self.calls = 0
        self.corrupt = corrupt

    def forward(self, target_hidden, noise_embedding, position_ids, past_key_values,
                use_cache, is_causal, extra_kv_hidden=None):
        n = noise_embedding.shape[1]
        carrier_n = 0 if extra_kv_hidden is None else extra_kv_hidden.shape[1]
        assert position_ids.shape[1] == target_hidden.shape[1] + carrier_n + n
        if carrier_n:
            assert torch.equal(extra_kv_hidden[..., 2], position_ids[:, :carrier_n].float())
        rows_pos = position_ids[:, -n:]
        assert torch.equal(rows_pos[0], torch.arange(int(rows_pos[0, 0]), int(rows_pos[0, 0]) + n))
        kv = torch.zeros(1, 1, position_ids.shape[1], 1)
        past_key_values.update(kv, kv, 0)
        anchor = int(noise_embedding[0, 0, 0])
        predicted = (torch.arange(n)[None] + anchor) % VOCAB
        confidence = torch.full((1, n), 12.0)
        trusted = self.confident[self.calls % len(self.confident)]
        if trusted < n - 1:
            confidence[:, trusted + 1:] = 1
            if self.corrupt:
                predicted[:, trusted + 1:] = (predicted[:, trusted + 1:] + 3) % VOCAB
        self.calls += 1
        return torch.stack([predicted.float(), confidence, rows_pos.float()], -1)


class PrefixTests(unittest.TestCase):
    def test_probability_stops_at_first_bad_slot(self):
        lp = torch.tensor([math.log(.99), math.log(.6), math.log(.999)])
        self.assertEqual(select_prefix(lp, "token_prob", .9), 1)

    def test_cumulative_prefix(self):
        lp = torch.tensor([math.log(.99), math.log(.98), math.log(.97), math.log(.6)])
        self.assertEqual(select_prefix(lp, "prefix_logp", math.log(.9)), 3)

    def test_empty_and_zero(self):
        self.assertEqual(select_prefix(torch.tensor([]), "prefix_logp", -1), 0)
        self.assertEqual(select_prefix(torch.tensor([-2.]), "prefix_logp", -1), 0)

    def test_validation(self):
        for kwargs in ({"threshold": 1}, {"max_draft_passes": 0}, {"max_candidates": 0},
                       {"threshold": float("nan")}, {"rule": "token_prob", "threshold": 0},
                       {"total_logp_budget": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PrefixPolicy(**kwargs)


class DecodeTests(unittest.TestCase):
    prompt = torch.tensor([[1, 2, 3]])

    def policy(self, **kwargs):
        return replace(PrefixPolicy(rule="token_prob", threshold=.9), **kwargs)

    def decode(self, draft=None, policy=None, length=40, stops=None):
        return adaptive_dflash_generate(draft, Target(), self.prompt, length, stops or [], policy, trace=True)

    def test_ten_plus_eight(self):
        r = self.decode(Draft(), self.policy())
        self.assertEqual(r.round_traces[0]["retained"], [10, 8])
        self.assertEqual(r.verify_lengths[0], 19)  # anchor + 18 candidates
        self.assertEqual(r.round_traces[0]["second_segment_accepted"], 8)
        self.assertTrue(torch.equal(r.output_ids, self.decode().output_ids))

    def test_all_modes_match_ar(self):
        ref = self.decode().output_ids
        for mode in ("single", "original_loop", "truncate_verify", "adaptive_loop"):
            with self.subTest(mode=mode):
                r = self.decode(Draft(), self.policy(mode=mode))
                self.assertTrue(torch.equal(r.output_ids, ref))
                self.assertEqual(r.decode_tokens, 39)
                self.assertEqual(sum(r.committed_lengths), r.decode_tokens)

    def test_zero_prefix_first_pass_fallback(self):
        r = self.decode(Draft(confident=(0,)), self.policy())
        self.assertEqual(r.round_traces[0]["stop_reason"], "zero_prefix_fallback")
        self.assertEqual(r.draft_calls, r.verify_calls)
        self.assertTrue(torch.equal(r.output_ids, self.decode().output_ids))

    def test_zero_second_pass(self):
        r = self.decode(Draft(confident=(10, 0)), self.policy())
        self.assertEqual(r.round_traces[0]["retained"], [10, 0])
        self.assertTrue(torch.equal(r.output_ids, self.decode().output_ids))

    def test_three_passes_and_carrier_positions(self):
        r = self.decode(Draft(confident=(4, 5, 6)), self.policy(max_draft_passes=3))
        self.assertEqual(r.round_traces[0]["retained"], [4, 5, 6])
        self.assertTrue(torch.equal(r.output_ids, self.decode().output_ids))

    def test_one_token_progress_has_empty_carrier(self):
        r = self.decode(Draft(confident=(1,)), self.policy(), length=16)
        self.assertEqual(r.round_traces[0]["retained"], [1, 1])
        self.assertTrue(torch.equal(r.output_ids, self.decode(length=16).output_ids))

    def test_minimum_progress(self):
        r = self.decode(Draft(confident=(1,)), self.policy(min_continue_tokens=2))
        self.assertEqual(r.draft_calls, r.verify_calls)

    def test_risk_budget(self):
        r = self.decode(Draft(), self.policy(total_logp_budget=-.0001))
        self.assertTrue(torch.equal(r.output_ids, self.decode().output_ids))

    def test_candidate_cap(self):
        r = self.decode(Draft(), self.policy(max_candidates=12))
        self.assertEqual(r.round_traces[0]["retained"], [10, 2])
        self.assertLessEqual(max(r.verify_lengths), 13)

    def test_lengths_and_eos(self):
        for length in (1, 2, 3, 10, 17, 25):
            for eos in (None, [4], [5], [13], [22]):
                with self.subTest(length=length, eos=eos):
                    r = self.decode(Draft(), self.policy(), length, eos)
                    ref = self.decode(length=length, stops=eos)
                    self.assertTrue(torch.equal(r.output_ids, ref.output_ids))
                    self.assertLessEqual(r.num_output_tokens, length)

    def test_trace_does_not_change_output(self):
        a = self.decode(Draft(), self.policy())
        b = adaptive_dflash_generate(Draft(), Target(), self.prompt, 40, [], self.policy(), trace=False)
        self.assertTrue(torch.equal(a.output_ids, b.output_ids))
        self.assertEqual(b.round_traces, [])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            adaptive_dflash_generate(Draft(), Target(), self.prompt.repeat(2, 1), 4, [])

    def test_original_loop_matches_upstream_proposals(self):
        # Load upstream decoder without importing SpecForge's training stack.
        stub = SimpleNamespace(extract_context_feature=lambda hs, ids: torch.cat([hs[i + 1] for i in ids], -1))
        path = Path(__file__).resolve().parents[1] / "eval/linear_decode.py"
        spec = importlib.util.spec_from_file_location("upstream_decoder", path)
        upstream = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"specforge.modeling.draft.dflash": stub}):
            spec.loader.exec_module(upstream)
        upstream._cuda_time = time.perf_counter
        for gate in (-1e9, -1., 1.):
            target1, target2 = Target(), Target()
            draft1, draft2 = Draft(confident=(15,)), Draft(confident=(15,))
            a = upstream.dflash_generate(draft1, target1, self.prompt, 70, [], loop_gate=gate, stage_cap=1)
            b = adaptive_dflash_generate(draft2, target2, self.prompt, 70, [],
                                        self.policy(mode="original_loop", loop_gate=gate))
            self.assertTrue(torch.equal(a, b.output_ids))
            self.assertEqual(target1.verify_widths[:-1], target2.verify_widths[:-1])


class TinyQwenTests(unittest.TestCase):
    def test_real_qwen_and_dflash_cache_path(self):
        # Load the actual drafter and its registry under a lightweight package,
        # without importing optional training/distributed dependencies.
        directory = Path(__file__).resolve().parents[1] / "specforge/modeling/draft"
        package = ModuleType("_test_dflash_package")
        package.__path__ = [str(directory)]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + ".dflash")
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(42)
        cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=256)
        cfg._attn_implementation = "sdpa"
        target = Qwen3ForCausalLM(cfg).eval()
        dc = Qwen3Config(**cfg.to_dict())
        dc.num_hidden_layers = 1
        dc.num_target_layers = 2
        dc.block_size = 8
        dc.dflash_config = {"target_layer_ids": [0], "mask_token_id": 63}
        dc._attn_implementation = "sdpa"
        draft = module.DFlashDraftModel(dc).eval()
        prompt = torch.tensor([[1, 2, 3]])
        ref = adaptive_dflash_generate(None, target, prompt, 12, []).output_ids
        # Force partial retention to exercise the REAL bidirectional attention,
        # carrier RoPE positions, and HF target/draft KV updates on CPU.
        with patch("eval.adaptive_linear_decode.select_prefix", side_effect=lambda lp, *_: min(3, lp.numel())):
            result = adaptive_dflash_generate(draft, target, prompt, 12, [],
                                              PrefixPolicy(max_candidates=10), trace=True)
        self.assertTrue(torch.equal(ref, result.output_ids))
        self.assertEqual(result.round_traces[0]["retained"], [3, 3])


if __name__ == "__main__":
    unittest.main()
