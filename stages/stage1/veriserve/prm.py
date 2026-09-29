"""Adapter for Qwen2.5-Math-PRM-7B process reward inference.

The reward model scores steps; it does not generate explanations. The JSON and
feedback strings below are deterministic adapters and are identified as such.
"""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .model import cuda_now


STEP_RE = re.compile(r"(?m)^Step\s+(\d+)\s*:")
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def split_steps(text: str, start: int = 1) -> list[tuple[int, str]]:
    matches = list(STEP_RE.finditer(text))
    if not matches:
        value = text.strip()
        return [(start, value)] if value else []
    steps = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            steps.append((int(match.group(1)), body))
    return steps


class ProcessRewardModel:
    def __init__(self, model_path: str | Path, threshold: float = 0.5):
        self.model_path = str(model_path)
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True,
                                                       trust_remote_code=True)
        self.separator_id = self.tokenizer.convert_tokens_to_ids("<extra_0>")
        if not isinstance(self.separator_id, int) or self.separator_id == self.tokenizer.unk_token_id:
            raise RuntimeError("PRM tokenizer lacks <extra_0> step separator")
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.bfloat16,
                                          bnb_4bit_use_double_quant=True)
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True,
            quantization_config=quantization,
        ).eval()
        self.device = next(self.model.parameters()).device
        if self.device.type != "cuda":
            raise RuntimeError("PRM is not entirely on CUDA")

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def input_ids(self, question: str, accepted: str, new: str) -> tuple[list[int], list[int], int]:
        previous = split_steps(accepted)
        following = split_steps(new, start=previous[-1][0] + 1 if previous else 1)
        all_steps = previous + following
        if not all_steps:
            return [], [], 0
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<extra_0>".join(body for _, body in all_steps) + "<extra_0>"},
        ]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                 add_generation_prompt=False)
        positions = [index for index, token in enumerate(ids) if token == self.separator_id]
        if len(positions) != len(all_steps):
            raise RuntimeError(f"Expected {len(all_steps)} PRM step tokens, got {len(positions)}")
        return ids, [number for number, _ in all_steps], len(previous)

    @torch.inference_mode()
    def verify(self, question: str, accepted: str, new: str) -> tuple[str, int, int, float]:
        ids, numbers, prior_count = self.input_ids(question, accepted, new)
        if not ids or len(numbers) == prior_count:
            value: dict[str, Any] = {"verdict": "UNCERTAIN", "first_error_step": None,
                                     "diagnosis": "No complete step to score.", "hint": ""}
            return json.dumps(value), 0, 0, 0.0
        start = cuda_now()
        inputs = torch.tensor([ids], device=self.device)
        output = self.model(input_ids=inputs, use_cache=False)
        positions = [index for index, token in enumerate(ids) if token == self.separator_id]
        scores = torch.softmax(output.logits[0, positions].float(), dim=-1)[:, 1].tolist()
        seconds = cuda_now() - start
        failing = next((index for index, score in enumerate(scores) if score < self.threshold), None)
        if failing is None:
            value = {"verdict": "PASS", "first_error_step": None,
                     "diagnosis": "", "hint": ""}
        elif failing < prior_count:
            value = {"verdict": "UNCERTAIN", "first_error_step": None,
                     "diagnosis": "An earlier accepted step received a low process score.",
                     "hint": ""}
        else:
            number = numbers[failing]
            value = {"verdict": "FAIL", "first_error_step": number,
                     "diagnosis": f"Step {number} has a low process score ({scores[failing]:.3f}).",
                     "hint": f"Recheck the reasoning and arithmetic in Step {number} from the accepted prefix."}
        value["step_scores"] = [{"step": number, "score": round(score, 6)}
                                for number, score in zip(numbers, scores)]
        value["verifier_output_kind"] = "deterministic_adapter_from_process_reward"
        return json.dumps(value, ensure_ascii=False), len(ids), 0, seconds

    @torch.inference_mode()
    def profile(self, ids: list[int]) -> float:
        start = cuda_now()
        self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
        return cuda_now() - start
