from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------
# ParseResult and semantic parser API
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ParseResult:
    """
    Result of a semantic parser.

    - action: parsed/cleaned action, or None if parsing fails
    - reward: parser-level reward (penalty for format errors, etc.)
    - obs: optional synthetic observation the agent receives on failure
    """
    action: Optional[str]
    reward: float
    obs: Optional[str]


Parser = Callable[[str], ParseResult]


def compose_parsers(*parsers: Parser) -> Parser:
    """
    Chain multiple Parser functions left-to-right.

    If any parser fails (action=None), return that failure with
    accumulated reward.
    """
    def _p(raw: str) -> ParseResult:
        current = ParseResult(action=raw, reward=0.0, obs=None)

        for parser in parsers:
            result = parser(current.action)  # type: ignore[arg-type]
            if result.action is None:
                return ParseResult(
                    action=None,
                    reward=current.reward + result.reward,
                    obs=result.obs,
                )
            # success: accumulate reward
            current = ParseResult(
                action=result.action,
                reward=current.reward + result.reward,
                obs=None,
            )

        return current
    return _p


# ---------------------------------------------------------------------
# Strict CoT <think>...</think> prefix parser
# ---------------------------------------------------------------------

def cot_prefix_parser(
    raw: str,
    *,
    success_reward: float = 0.1,
    error_reward: float = -1.0,
) -> ParseResult:
    """
    STRICT CoT prefix parser.

    Required:
        <think> ... </think> ANSWER

    Output:
        action = ANSWER

    Rewards:
        Defaults to +0.1 on success and -1.0 on failure; override via keyword
        args or functools.partial for custom parser instances.
    """
    try:
        pattern = re.compile(
            r"^\s*<think>(.*?)</think>\s*(.+)$",
            flags=re.DOTALL | re.IGNORECASE,
        )
        m = pattern.match(raw)
        if not m:
            raise ValueError("Expected '<think>...</think>' prefix followed by answer.")

        answer = m.group(2).strip()
        if not answer:
            raise ValueError("Missing answer after </think>.")

        return ParseResult(action=answer, reward=success_reward, obs=None)

    except Exception as e:
        return ParseResult(
            action=None,
            reward=error_reward,
            obs=f"Invalid CoT structure: {e}",
        )


# ---------------------------------------------------------------------
# Strict XML <move>...</move> parser
# ---------------------------------------------------------------------

def xml_move_parser(
    raw: str,
    *,
    success_reward: float = 0.1,
    error_reward: float = -1.0,
) -> ParseResult:
    """
    STRICT parser for <move>...</move>.

    Rewards:
        Defaults to +0.1 on success and -1.0 on failure; override via keyword
        args or functools.partial for custom parser instances.
    """
    try:
        m = re.search(r"<move>(.*?)</move>", raw, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            raise ValueError("Expected <move>...</move>.")

        inner = m.group(1).strip()
        if not inner:
            raise ValueError("Empty <move> tag.")

        return ParseResult(action=inner, reward=success_reward, obs=None)

    except Exception as e:
        return ParseResult(
            action=None,
            reward=error_reward,
            obs=f"Invalid action format: {e}",
        )


# ---------------------------------------------------------------------
# Strict \boxed{...} answer parser
# ---------------------------------------------------------------------

def extract_last_boxed_content(raw: str) -> Optional[str]:
    """
    Extract the content of the last LaTeX \\boxed{...} occurrence.

    Supports nested braces inside the boxed content (e.g. \\boxed{\\frac{1}{2}}).
    Returns None if no well-formed \\boxed{...} is found.
    """
    matches = list(re.finditer(r"\\boxed\s*\{", raw))
    if not matches:
        return None

    def _parse_braced(start_brace_idx: int) -> Optional[str]:
        if start_brace_idx >= len(raw) or raw[start_brace_idx] != "{":
            return None

        depth = 0
        i = start_brace_idx
        while i < len(raw):
            ch = raw[i]
            prev = raw[i - 1] if i > 0 else ""

            if ch == "{" and prev != "\\":
                depth += 1
            elif ch == "}" and prev != "\\":
                depth -= 1
                if depth == 0:
                    return raw[start_brace_idx + 1 : i]
            i += 1
        return None

    # Prefer the last occurrence (the model may include intermediate boxes).
    for m in reversed(matches):
        inner = _parse_braced(m.end() - 1)
        if inner is not None:
            return inner
    return None


def boxed_parser(
    raw: str,
    *,
    success_reward: float = 0.1,
    error_reward: float = -1.0,
) -> ParseResult:
    """
    STRICT parser that requires the final answer to appear inside \\boxed{...}.

    Rewards:
        Defaults to +0.1 on success and -1.0 on failure; override via keyword
        args or functools.partial for custom parser instances.
    """
    try:
        inner = extract_last_boxed_content(raw)
        if inner is None:
            raise ValueError("Expected \\boxed{...} with the final answer.")

        inner = inner.strip()
        if not inner:
            raise ValueError("Empty \\boxed{} content.")

        # Positive intrinsic reward for good formatting
        return ParseResult(action=inner, reward=success_reward, obs=None)

    except Exception as e:
        return ParseResult(
            action=None,
            reward=error_reward,
            obs=f"Invalid boxed answer: {e}",
        )
