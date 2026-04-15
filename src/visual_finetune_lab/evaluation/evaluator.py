from __future__ import annotations

import importlib.util
import os
import torch
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import nltk
import structlog
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, pipeline

logger = structlog.get_logger()

nltk.download("punkt_tab", quiet=True)

# Capture the real find_spec before anything can replace it.
_REAL_FIND_SPEC = importlib.util.find_spec

# transformers >=5.x removed several DynamicCache methods that Phi-3.5-vision's
# custom forward still calls. Restore all missing ones at import time.
def _patch_dynamic_cache() -> None:
    from transformers import DynamicCache
    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod  # type: ignore[misc]
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (keys, values) in enumerate(past_key_values):
                    cache.update(keys, values, layer_idx)
            return cache
        DynamicCache.from_legacy_cache = from_legacy_cache

    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:  # type: ignore[misc]
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "get_max_length"):
        def get_max_length(self) -> None:  # type: ignore[misc]
            return None
        DynamicCache.get_max_length = get_max_length  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "to_legacy_cache"):
        def to_legacy_cache(self):  # type: ignore[misc]
            return tuple(
                (self.key_cache[i], self.value_cache[i])
                for i in range(len(self.key_cache))
            )
        DynamicCache.to_legacy_cache = to_legacy_cache  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length(0))  # type: ignore[attr-defined]

_patch_dynamic_cache()


@dataclass
class EvalResult:
    question: str
    reference: str
    prediction: str
    bleu: float
    rouge_l: float
    llm_judge_score: float   # 1-5 scale from GPT-4o
    llm_judge_reason: str


LLM_JUDGE_PROMPT = """You are evaluating the quality of an AI model's answer.

Question: {question}
Reference answer: {reference}
Model answer: {prediction}

Rate the model answer on a scale from 1 to 5:
1 = completely wrong or hallucinated
3 = partially correct, missing key details
5 = accurate, complete, matches the reference

Return JSON: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""


class ModelEvaluator:
    """
    Evaluates a fine-tuned vision model on a held-out JSONL dataset using:
      - BLEU score (n-gram overlap with reference)
      - ROUGE-L (longest common subsequence)
      - LLM-as-judge (GPT-4o scores each prediction 1-5)

    All results are logged to MLflow via the active run.

    Example:
        evaluator = ModelEvaluator(checkpoint_path=Path("checkpoints/best"))
        results = evaluator.evaluate(test_samples)
        evaluator.print_summary(results)
    """

    def __init__(self, checkpoint_path: Path, judge_model: str = "gpt-4o"):
        self.checkpoint_path = checkpoint_path
        self.judge_model = judge_model
        self._pipe = None
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evaluate(self, samples: list[dict]) -> list[EvalResult]:
        pipe = self._get_pipeline()
        results = []
        for sample in samples:
            prediction = self._predict(pipe, sample["question"], sample.get("image_path"))
            bleu = self._bleu(sample["answer"], prediction)
            rouge_l = self._rouge_l(sample["answer"], prediction)
            judge_score, judge_reason = self._llm_judge(
                sample["question"], sample["answer"], prediction
            )
            results.append(EvalResult(
                question=sample["question"],
                reference=sample["answer"],
                prediction=prediction,
                bleu=bleu,
                rouge_l=rouge_l,
                llm_judge_score=judge_score,
                llm_judge_reason=judge_reason,
            ))
            logger.info(
                "sample_evaluated",
                bleu=round(bleu, 3),
                rouge_l=round(rouge_l, 3),
                llm_judge=judge_score,
            )
        return results

    def print_summary(self, results: list[EvalResult]) -> None:
        n = len(results)
        avg_bleu = sum(r.bleu for r in results) / n
        avg_rouge = sum(r.rouge_l for r in results) / n
        avg_judge = sum(r.llm_judge_score for r in results) / n
        print(f"\n{'='*50}")
        print(f"Evaluation summary ({n} samples)")
        print(f"  BLEU:       {avg_bleu:.3f}")
        print(f"  ROUGE-L:    {avg_rouge:.3f}")
        print(f"  LLM Judge:  {avg_judge:.2f} / 5")
        print(f"{'='*50}\n")

    # ------------------------------------------------------------------ #

    @staticmethod
    @contextmanager
    def _hide_flash_attn():
        def _patched(name, *args, **kwargs):
            if name == "flash_attn":
                return None
            return _REAL_FIND_SPEC(name, *args, **kwargs)
        importlib.util.find_spec = _patched
        try:
            yield
        finally:
            importlib.util.find_spec = _REAL_FIND_SPEC

    def _get_pipeline(self):
        if self._pipe is None:
            processor = AutoProcessor.from_pretrained(
                self.checkpoint_path, trust_remote_code=True
            )
            # Get the original base model ID from the PEFT adapter config,
            # not from tokenizer.name_or_path which points to the local checkpoint.
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(self.checkpoint_path)
            base_model_id = peft_config.base_model_name_or_path
            model_config = AutoConfig.from_pretrained(
                base_model_id, trust_remote_code=True
            )
            model_config._attn_implementation = "eager"
            model_config._attn_implementation_autoset = False
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            with self._hide_flash_attn():
                base = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    config=model_config,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
            model = PeftModel.from_pretrained(base, self.checkpoint_path)
            model.config.use_cache = False
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=processor.tokenizer,
                max_new_tokens=256,
            )
        return self._pipe

    def _predict(self, pipe, question: str, image_path: str | None) -> str:
        prompt = f"User: {question}\nAssistant:"
        output = pipe(prompt)[0]["generated_text"]
        # Extract assistant portion
        if "Assistant:" in output:
            return output.split("Assistant:")[-1].strip()
        return output.strip()

    def _bleu(self, reference: str, prediction: str) -> float:
        ref_tokens = nltk.word_tokenize(reference.lower())
        pred_tokens = nltk.word_tokenize(prediction.lower())
        sf = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=sf)

    def _rouge_l(self, reference: str, prediction: str) -> float:
        scores = self._rouge.score(reference, prediction)
        return scores["rougeL"].fmeasure

    def _llm_judge(self, question: str, reference: str, prediction: str) -> tuple[float, str]:
        prompt = LLM_JUDGE_PROMPT.format(
            question=question, reference=reference, prediction=prediction
        )
        try:
            response = self._openai.chat.completions.create(
                model=self.judge_model,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            import json
            data = json.loads(response.choices[0].message.content)
            return float(data["score"]), data.get("reason", "")
        except Exception:
            logger.exception("llm_judge_failed")
            return 0.0, "error"
