"""CPU-only check for streamed delta accounting, windows and time-budget failure."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

api = runpy.run_path(str(Path(__file__).with_name('benchmark.py')))
sys.modules['vllm'] = NS(SamplingParams=lambda **kwargs: kwargs)
sys.modules['vllm.sampling_params'] = NS(RequestOutputKind=NS(DELTA='delta'))


class Engine:
    def __init__(self):
        self.chunks = iter([1, 8191, 8192, 16384, 32768])
        self.remaining = 5

    def add_request(self, key, prompt, params):
        assert params['output_kind'] == 'delta' and not params['ignore_eos']
        self.key = key

    def has_unfinished_requests(self):
        return self.remaining > 0

    def step(self):
        count = next(self.chunks)
        self.remaining -= 1
        return [NS(request_id=self.key, finished=self.remaining == 0,
                   outputs=[NS(token_ids=[7] * count, finish_reason='length')])]

    def abort_request(self, keys):
        self.aborted = keys
        self.remaining = 0


cfg = dict(temperature=1, top_p=.95, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1)
records = [dict(unique_id='fixed', prompt_ids=[1, 2], generation_seed=123)]
engine = Engine()
llm = NS(llm_engine=engine, get_metrics=lambda: [NS(name='vllm:num_preemptions', value=0)])
with patch('time.perf_counter', side_effect=range(7)):
    result = api['stream'](llm, records, 65536, cfg, [248046], 'check', float('inf'))
assert result['tokens'] == 65536 and result['seconds'] == 6
assert len(result['rows'][0]['decode_windows']) == 4
assert result['rows'][0]['decode_windows'][0]['tokens_per_second'] == 8191
assert result['rows'][0]['decode_windows'][-1]['tokens_per_second'] == 32768
assert result['preemptions'] == 0
engine = Engine()
llm.llm_engine = engine
try:
    api['stream'](llm, records, 65536, cfg, [248046], 'timeout', 0)
except TimeoutError:
    assert engine.aborted == ['timeout-0']
else:
    raise AssertionError('Expired benchmark did not abort')
print('PASS: delta counts, all four windows, preemption count and timeout; no GPU inference.')
