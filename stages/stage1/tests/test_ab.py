"""CPU checks for FAIL recovery and pending-token cache boundaries."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from stages.stage1.scripts import run_ab as ab
from veriserve_research.inference import Session


class Cache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = min(self.length, length)


class FakeSession(Session):
    def forward(self, ids):
        if self.cache is None:
            self.cache = Cache()
        self.cache.length += len(ids)
        self.logits = torch.zeros(10)
        assert self.cache.length == len(self.ids)


class CacheRecoveryTests(unittest.TestCase):
    def generator(self, arm):
        return ab.ResidentGenerator(SimpleNamespace(model=None, tokenizer=None, device="cpu"), arm)

    @patch.object(ab, "cuda_now", return_value=0.0)
    @patch.object(ab, "Session", FakeSession)
    def test_fail_crops_A_but_recomputes_B(self, clock):
        a, b = self.generator("A"), self.generator("B")
        for generator in (a, b):
            generator._prefill([1, 2, 3, 4, 5])
            generator.restore = ([1, 2, 3], [8, 9])
            generator._prefill([1, 2, 3, 8, 9])
            self.assertEqual(generator.session.ids, [1, 2, 3, 8, 9])
            self.assertEqual(generator.session.cache.get_seq_length(), 5)
        self.assertEqual(a.recoveries[0]["forward_tokens"], 2)
        self.assertEqual(b.recoveries[0]["forward_tokens"], 5)

    @patch.object(ab, "cuda_now", return_value=0.0)
    @patch.object(ab, "Session", FakeSession)
    def test_missing_last_sampled_token_is_fed_before_feedback(self, clock):
        generator = self.generator("A")
        generator._prefill([1, 2, 3])
        generator.session.ids.append(4)  # sampled but absent from KV
        generator.restore = ([1, 2, 3, 4], [8])
        generator._prefill([1, 2, 3, 4, 8])
        self.assertEqual(generator.recoveries[0]["forward_tokens"], 2)
        self.assertEqual(generator.session.cache.get_seq_length(), 5)

    @patch.object(ab, "cuda_now", return_value=0.0)
    @patch.object(ab, "Session", FakeSession)
    def test_pass_keeps_same_cache_and_only_feeds_pending(self, clock):
        for arm in ("A", "B"):
            generator = self.generator(arm)
            generator._prefill([1, 2, 3])
            cache = generator.session.cache
            generator.session.ids.append(4)
            generator._prefill([1, 2, 3, 4])
            self.assertIs(generator.session.cache, cache)
            self.assertEqual(generator.actual_forward_tokens, 4)
            with self.assertRaisesRegex(RuntimeError, "context changed"):
                generator._prefill([1, 2, 7])


if __name__ == "__main__":
    unittest.main()
