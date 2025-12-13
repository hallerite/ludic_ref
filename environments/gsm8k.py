from __future__ import annotations

import re
from typing import Optional

from ludic.envs.dataset_qa_env import DatasetQAEnv, Sample


def gsm8k_target_parser(text: str) -> str:
    """
    Normalize GSM8K-style ground-truth answers:
      - strip whitespace
      - take text after '####' (required)
      - drop commas and grab the last integer token
    """
    cleaned = text.strip()
    if "####" not in cleaned:
        raise ValueError("Expected GSM8K answer to contain '####'.")
    cleaned = cleaned.split("####")[-1].strip()

    cleaned = cleaned.replace(",", "").strip()
    int_tokens = re.findall(r"-?\d+", cleaned)
    if not int_tokens:
        raise ValueError("Could not find a final integer after '####'.")
    return int_tokens[-1].strip()


class GSM8KEnv(DatasetQAEnv):
    """
    Convenience wrapper for GSM8K-style QA.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful math tutor. Think step-by-step. When you are ready, place the final numeric answer inside \\boxed{...}."
    )

    def __init__(
        self,
        sample: Sample,
        *,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        try:
            from math_verify import verify as mv  # type: ignore
        except Exception as e:
            raise SystemExit(
                "GSM8KEnv requires 'math-verify' (import name: math_verify) for grading. "
                "Install with: uv pip install math-verify"
            ) from e

        def _verifier(pred: str, target: str) -> bool:
            return bool(mv(pred, target))

        super().__init__(
            sample=sample,
            prompt_key="question",
            answer_key="answer",
            system_prompt=system_prompt,
            target_parser=gsm8k_target_parser,
            verifier=_verifier,
        )
