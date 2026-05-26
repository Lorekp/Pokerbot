"""LLM teacher wrapper for PokerBench distillation."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL_ID = "YiPz/llama3-8b-pokerbench-sft"


@dataclass
class LLMTeacherConfig:
    model_id: str = DEFAULT_MODEL_ID
    load_in_4bit: bool = True
    batch_size: int = 4
    max_new_tokens: int = 16
    temperature: float = 0.1


class LLMTeacher:
    """Wraps YiPz/llama3-8b-pokerbench-sft for batched spot inference."""

    def __init__(self, config: LLMTeacherConfig | None = None) -> None:
        self.config = config or LLMTeacherConfig()
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("Install llm extras: pip install -e '.[llm]'") from exc

        quant_config = None
        if self.config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        kwargs: dict = {"device_map": "auto", "trust_remote_code": True}
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
        self._model.eval()

    def predict_batch(self, instructions: list[str]) -> list[str]:
        self.load()
        assert self._tokenizer is not None and self._model is not None

        import torch

        prompts = []
        for instr in instructions:
            text = instr.strip()
            if not text.endswith(":"):
                if "optimal action is" not in text.lower():
                    text = text + "\nYour optimal action is:"
            prompts.append(text)

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        results: list[str] = []
        for i, seq in enumerate(out):
            prompt_len = inputs["input_ids"][i].shape[0]
            gen = self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            results.append(gen.strip().split("\n")[0].strip().lower())
        return results


def parse_llm_action(text: str) -> tuple[int | None, str]:
    """Return (action_index, raw_text) or (None, raw) on failure."""
    from pokerbot.engine.actions import action_to_index, parse_pokerbench_output

    text = text.strip().lower()
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            action = parse_pokerbench_output(line)
            return action_to_index(action), line
        except ValueError:
            continue
    return None, text
