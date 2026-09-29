from __future__ import annotations

import gc
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

from .data import predicted_answer
from .state import PathState


GENERATOR_TEMPLATE = """Solve the problem step by step.
Use Step 1:, Step 2:, and so on. Put a blank line after every step.
End with Final answer: <number>.
If feedback is inserted, preserve the accepted prefix and continue as instructed.

Problem: {question}"""

VERIFIER_TEMPLATE = """Check the NEW segment of this solution, using the problem and accepted prefix as context.
The accepted prefix was accepted by a previous check; that does not prove it is correct.
An unfinished but valid new segment is not an error.
If the suspected error is only in the accepted prefix, return UNCERTAIN.
Return exactly one raw JSON object, with no Markdown fence or surrounding prose.
Include verdict (PASS, FAIL, or UNCERTAIN), first_error_step (integer or null), diagnosis (string), and hint (string).
Use an empty string for a diagnosis or hint that does not apply; never use null for those fields.
Do not provide a complete new solution.

Problem: {question}
Previously accepted steps: {accepted_steps}
New steps to check: {new_steps}"""

STEP_RE = re.compile(r"(?m)^Step\s+(\d+)\s*:")
FINAL_RE = re.compile(r"(?im)^Final answer:\s*[^\n]+\n")


def cuda_now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@dataclass
class GenerationResult:
    kept_ids: list[int]
    pending_ids: list[int]
    reason: str
    answer: str | None
    current_step: int
    prefill_seconds: float
    decode_seconds: float
    rng_state: list[int]


class LanguageModel:
    def __init__(self, model_path: str | Path, quantization: str | None = None):
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        options: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": {"": 0},
                                   "low_cpu_mem_usage": True}
        if quantization == "nf4":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **options).eval()
        self.device = next(self.model.parameters()).device
        if self.device.type != "cuda":
            raise RuntimeError("Model is not entirely on CUDA")

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def prompt_ids(self, prompt: str) -> list[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True,
        )

    def generator_prompt(self, question: str) -> list[int]:
        return self.prompt_ids(GENERATOR_TEMPLATE.format(question=question))

    def text_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.inference_mode()
    def _prefill(self, ids: list[int]) -> tuple[DynamicCache, torch.Tensor, float]:
        if not ids:
            raise ValueError("Cannot prefill an empty context")
        cache = DynamicCache(config=self.model.config)
        start = cuda_now()
        output = self.model(input_ids=torch.tensor([ids], device=self.device),
                            past_key_values=cache, use_cache=True)
        elapsed = cuda_now() - start
        return cache, output.logits[:, -1, :], elapsed

    @torch.inference_mode()
    def _next_logits(self, token: int, cache: DynamicCache) -> torch.Tensor:
        output = self.model(input_ids=torch.tensor([[token]], device=self.device),
                            past_key_values=cache, use_cache=True)
        return output.logits[:, -1, :]

    @staticmethod
    def _sample(logits: torch.Tensor, generator: torch.Generator,
                temperature: float, top_p: float) -> int:
        scores = logits[0].float() / temperature
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        excluded = cumulative - probs > top_p
        probs[excluded] = 0
        probs /= probs.sum()
        selected = torch.multinomial(probs, 1, generator=generator)
        return int(sorted_indices[selected].item())

    def _split_before_marker(self, old_ids: list[int], new_ids: list[int],
                             marker_offset: int) -> tuple[list[int], list[int]]:
        for index in range(len(new_ids) + 1):
            decoded = self.decode(old_ids + new_ids[:index])
            if len(decoded) > marker_offset:
                return new_ids[:max(0, index - 1)], new_ids[max(0, index - 1):]
        return new_ids, []

    @torch.inference_mode()
    def generate_path(self, state: PathState, config: dict[str, Any]) -> GenerationResult:
        state.merge_pending()
        remaining = min(state.budget.generated_tokens - state.generated_tokens,
                        state.budget.context_tokens - len(state.context_ids))
        if remaining <= 0:
            raise ValueError("No generation budget")
        interval = None if state.policy == "unchecked" else (
            10**9 if state.policy == "endpoint" else int(state.policy.removeprefix("k")))
        if interval != 10**9 and interval is not None:
            remaining = min(remaining, int(config["generation"]["max_step_tokens"]) * (interval + 1))
        cache, logits, prefill_seconds = self._prefill(state.context_ids)
        initial_segment_ids = state.context_ids[state.segment_start:]
        initial_segment_text = self.decode(initial_segment_ids)
        original_completed = max(0, len(STEP_RE.findall(initial_segment_text)) - 1)
        generator = torch.Generator(device=self.device)
        if state.rng_state is None:
            generator.manual_seed(state.seed)
        else:
            generator.set_state(torch.tensor(state.rng_state, dtype=torch.uint8))
        generated: list[int] = []
        reason = "GEN_BUDGET"
        answer = None
        current_step = state.checkpoint_step
        pending: list[int] = []
        decode_start = cuda_now()
        for _ in range(remaining):
            token = self._sample(logits, generator,
                                 float(config["generation"]["temperature"]),
                                 float(config["generation"]["top_p"]))
            generated.append(token)
            if token == self.tokenizer.eos_token_id:
                segment_text = self.decode(initial_segment_ids + generated)
                answer = predicted_answer(segment_text)
                reason = "FINAL" if answer is not None else "EOS"
                break
            segment_text = self.decode(initial_segment_ids + generated)
            matches = list(STEP_RE.finditer(segment_text))
            completed = max(0, len(matches) - 1) - original_completed
            if FINAL_RE.search(segment_text) and predicted_answer(segment_text) is not None:
                reason = "FINAL"
                answer = predicted_answer(segment_text)
                if matches:
                    current_step = int(matches[-1].group(1))
                break
            if interval is not None and completed >= interval and matches:
                reason = "STEP"
                current_step = int(matches[-2].group(1)) if len(matches) >= 2 else state.checkpoint_step
                kept, pending = self._split_before_marker(initial_segment_ids, generated,
                                                          matches[-1].start())
                generated = kept
                break
            logits = self._next_logits(token, cache)
        else:
            if len(state.context_ids) + len(generated) >= state.budget.context_tokens:
                reason = "CONTEXT_LIMIT"
            elif state.generated_tokens + len(generated) >= state.budget.generated_tokens:
                reason = "GEN_BUDGET"
            else:
                reason = "FORMAT_ERROR"
        decode_seconds = cuda_now() - decode_start
        return GenerationResult(generated, pending, reason, answer, current_step,
                                prefill_seconds, decode_seconds, generator.get_state().tolist())

    @torch.inference_mode()
    def generate_verification(self, prompt: str, max_tokens: int) -> tuple[str, int, int, float]:
        ids = self.prompt_ids(prompt)
        cache, logits, prefill_seconds = self._prefill(ids)
        generated: list[int] = []
        start = cuda_now()
        for _ in range(max_tokens):
            token = int(torch.argmax(logits, dim=-1).item())
            if token == self.tokenizer.eos_token_id:
                break
            generated.append(token)
            logits = self._next_logits(token, cache)
        seconds = prefill_seconds + (cuda_now() - start)
        return self.decode(generated), len(ids), len(generated), seconds


def verification_prompt(question: str, accepted: str, new: str) -> str:
    return VERIFIER_TEMPLATE.format(question=question, accepted_steps=accepted,
                                    new_steps=new)


def parse_verdict(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("verdict") not in {"PASS", "FAIL", "UNCERTAIN"}:
        return None
    if not all(key in value for key in ("first_error_step", "diagnosis", "hint")):
        return None
    if value["first_error_step"] is not None and not isinstance(value["first_error_step"], int):
        return None
    if not isinstance(value["diagnosis"], str) or not isinstance(value["hint"], str):
        return None
    return value


def feedback_text(diagnosis: str, hint: str, checkpoint_step: int) -> str:
    next_step = checkpoint_step + 1
    return ("\n[Verification feedback]\n" + diagnosis.strip() + "\n" + hint.strip()
            + "\n[Continue]\n"
            + f"Keep the accepted Steps 1–{checkpoint_step}. The later attempt has been discarded.\n"
            + f"Continue again from Step {next_step}.\n")
