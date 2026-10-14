"""Compare actual vLLM penalty/filter probabilities with the frozen HF semantics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from transformers import TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
from vllm.model_executor.layers.utils import apply_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from veriserve_research.artifacts.io import atomic_json,now
rows=[]
for seed in range(20):
 torch.manual_seed(seed);scores=torch.randn(1,1024,device='cuda')
 prompt=torch.tensor([[0,2,3]],device='cuda');suffix=torch.tensor([[1,1,4]],device='cuda')
 original=scores.clone();original[:,suffix.unique()]-=1.5
 penalized=apply_penalties(scores.clone(),prompt,suffix,torch.tensor([1.5],device='cuda'),torch.tensor([0.],device='cuda'),torch.tensor([1.],device='cuda'))
 assert torch.equal(original,penalized)
 ids=torch.cat([prompt,suffix],dim=1)
 expected=TopPLogitsWarper(.95)(ids,TopKLogitsWarper(20)(ids,TemperatureLogitsWarper(1.)(ids,original)))
 actual=apply_top_k_top_p_pytorch(penalized,torch.tensor([20],device='cuda'),torch.tensor([.95],device='cuda'))
 masks=torch.equal(torch.isfinite(expected),torch.isfinite(actual))
 error=float((expected.softmax(-1)-actual.softmax(-1)).abs().max())
 assert masks and error<=1e-7
 rows.append(dict(seed=seed,penalty_exact=True,support_equal=masks,max_probability_abs_error=error))
atomic_json(Path(__file__).parent/'sampling.json',dict(status='PASS',cases=rows,updated=now(),
 semantics='presence once on generated suffix only; prompt excluded; repetition=1; frequency=0; temperature=1 then top_k=20 then top_p=.95; min_p=0 inactive',
 rng='Engine/kernel/PyTorch runtimes differ; bitwise cross-engine generation is not assumed'))
print('SAMPLING PASS',len(rows),flush=True)
