"""Run with the vLLM environment; uses CPU tensors and no model."""
import unittest
from types import SimpleNamespace
import torch
from stages.stage3.verification_benefit_logits import RestoredPresence, pending_history


class RestoredPresenceTest(unittest.TestCase):
    def test_worker_processor_name_loads_in_installed_vllm(self):
        from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns
        from stages.stage3.verification_benefit_worker import PROCESSOR
        self.assertEqual(_load_logitsprocs_by_fqcns([PROCESSOR]), [RestoredPresence])

    def test_old_and_new_tokens_are_penalized_only_once(self):
        generated = [2, 4, 4]
        self.assertEqual(pending_history([1, 1, 2, 3], generated), [1, 3])
        processor = RestoredPresence(None, torch.device('cpu'), False)
        params = SimpleNamespace(extra_args={'restored_presence_history': [1, 1, 2, 3]}, presence_penalty=1.5)
        processor.update_state(SimpleNamespace(removed=[], moved=[], added=[(0, params, [9], generated)]))
        logits = processor.apply(torch.zeros((1, 10)))
        # vLLM's native presence penalty subsequently handles generated token IDs.
        logits[0, sorted(set(generated))] -= params.presence_penalty
        self.assertEqual(logits[0].tolist(), [0., -1.5, -1.5, -1.5, -1.5, 0., 0., 0., 0., 0.])
        generated.append(3)
        logits = processor.apply(torch.zeros((1, 10)))
        logits[0, sorted(set(generated))] -= params.presence_penalty
        self.assertEqual(logits[0, 3].item(), -1.5)


if __name__ == '__main__':
    unittest.main()
