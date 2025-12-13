from __future__ import annotations

from typing import Optional

from ludic.envs.dataset_qa_env import DatasetQAEnv, Sample
from ludic.parsers import extract_last_boxed_content


def math_target_parser(text: str) -> str:
    """
    Normalize MATH-style ground-truth answers:
      - strip whitespace
      - unbox the last \\boxed{...} (supports nested braces)
    """
    cleaned = text.strip()

    boxed = extract_last_boxed_content(cleaned)
    if boxed is None:
        raise ValueError("Expected a final answer in \\boxed{...}.")
    return boxed.strip()


class MATHEnv(DatasetQAEnv):
    """
    Convenience wrapper for MATH-style QA.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful math tutor. Think step by step. "
        "Put your final answer in \\boxed{...}."
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
                "MATHEnv requires 'math-verify' (import name: math_verify) for grading. "
                "Install with: uv pip install math-verify"
            ) from e

        def _verifier(pred: str, target: str) -> bool:
            return bool(mv(pred, target))

        super().__init__(
            sample=sample,
            prompt_key="problem" if "problem" in sample else "question",
            answer_key="solution" if "solution" in sample else "answer",
            system_prompt=system_prompt,
            target_parser=math_target_parser,
            verifier=_verifier,
        )
