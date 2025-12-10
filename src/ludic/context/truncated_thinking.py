from __future__ import annotations
import re
from typing import List, Pattern

from ludic.context.full_dialog import FullDialog
from ludic.types import Message

class TruncatedThinkingContext(FullDialog):
    """
    A context strategy that preserves full history in memory, but
    collapses <think>...</think> blocks in the prompt ONLY if
    they follow the strict CoT format:
    
        <think>...</think> [ANSWER/ACTION]
        
    If the message contains only thoughts (no answer), or malformed tags,
    it is left untouched so the model (and you) can see the full state.
    """

    def __init__(
        self, 
        system_prompt: str | None = None,
        placeholder: str = "... [TRUNCATED THOUGHTS] ...",
    ) -> None:
        super().__init__(system_prompt=system_prompt)
        self.placeholder = placeholder
        
        # STRICT PATTERN EXPLANATION:
        # ^\s* -> Start of string (ignoring leading whitespace)
        # (<think>)     -> Capture Group 1: Opening tag
        # (.*?)         -> Capture Group 2: The thought content (non-greedy)
        # (</think>)    -> Capture Group 3: Closing tag
        # \s* -> Optional whitespace between tag and answer
        # (.+)          -> Capture Group 4: THE ANSWER (Must be present and non-empty!)
        # $             -> End of string
        self.strict_pattern: Pattern = re.compile(
            r"^(\s*<think>)(.*?)(</think>\s*)(.+)$", 
            flags=re.DOTALL | re.IGNORECASE
        )

    def on_before_act(self) -> List[Message]:
        raw_history = super().on_before_act()
        sanitized_history: List[Message] = []

        for msg in raw_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                
                # We attempt to match the STRICT pattern.
                # If the regex doesn't match (e.g. missing answer, broken tags),
                # m will be None and we skip the truncation block.
                m = self.strict_pattern.match(content)
                
                if m:
                    # Reconstruct: <think> + placeholder + </think> + answer
                    # Group 1: "<think>" (with potential leading whitespace)
                    # Group 3: "</think>" (with potential trailing whitespace)
                    # Group 4: The Answer
                    new_content = f"{m.group(1)}{self.placeholder}{m.group(3)}{m.group(4)}"
                    
                    new_msg = dict(msg)
                    new_msg["content"] = new_content
                    sanitized_history.append(new_msg)
                else:
                    # Format wasn't exactly right -> Keep raw content
                    sanitized_history.append(msg)
            else:
                sanitized_history.append(msg)

        return sanitized_history