"""vLLM processor: preserve presence history across an original-token prefill."""
from vllm.v1.sample.logits_processor.interface import LogitsProcessor


def pending_history(history, generated):
    """The built-in penalty already charges current-request output tokens."""
    return sorted(set(history) - set(generated))


class RestoredPresence(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        self.requests = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, _prompt, output in batch_update.added:
            extra = params.extra_args or {}
            self.requests[index] = (extra.get('restored_presence_history', []), output,
                                    params.presence_penalty)
        for src, dst, direction in batch_update.moved:
            if direction.name == 'SWAP':
                self.requests[src], self.requests[dst] = self.requests.get(dst), self.requests.get(src)
            else:
                self.requests[dst] = self.requests.pop(src, None)
            self.requests = {i: x for i, x in self.requests.items() if x is not None}

    def apply(self, logits):
        for index, (history, output, penalty) in self.requests.items():
            if history and penalty:
                tokens = pending_history(history, output)
                if tokens:
                    logits[index, tokens] -= penalty
        return logits
