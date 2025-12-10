from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ludic.types import Message, Observation, Info, ChatResponse

class ContextStrategy(ABC):
    """
    Owns the chat transcript & how it's exposed to the agent at each step.
    Natively supports System, User, Assistant, and Tool messages.
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        """
        Initializes the context, optionally with a default system prompt.
        
        Args:
            system_prompt: The default system prompt to use if no override
                           is provided during reset.
        """
        self._messages: List[Message] = []
        # Store the default prompt
        self._default_system_prompt = system_prompt
        
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})


    # ---- lifecycle ------------------------------------------------------
    def reset(self, *, system_prompt: Optional[str] = None) -> None:
        """
        Resets the context memory.
        
        Args:
            system_prompt: An optional override prompt (e.g., from an Env).
                           If provided, it is used. If None, the
                           default prompt (from __init__) is used.
        """
        self._messages = []
        
        # Priority: 1. reset() override, 2. instance default
        prompt_to_use = system_prompt or self._default_system_prompt
        
        if prompt_to_use:
            self._messages.append({"role": "system", "content": prompt_to_use})

    # ---- capabilities ---------------------------------------------------
    @property
    def supports_tools(self) -> bool:
        """
        Does this context strategy support tool_calls and tool messages?
        
        Defaults to True for the base class since _messages is a standard list.
        Subclasses can override this to False if they strictly forbid it.
        """
        return True

    # ---- event hooks ----------------------------------------------------
    @abstractmethod
    def on_env_reset(self, obs: Observation, info: Info) -> None:
        """Called after env.reset(). Typically appends a user message."""

    @abstractmethod
    def on_before_act(self) -> List[Message]:
        """Return messages to send to the agent for the *next* act() call."""

    @abstractmethod
    def on_after_act(self, assistant_response: ChatResponse) -> None:
        """Record assistant output (and optionally transform it)."""

    @abstractmethod
    def on_after_step(self, next_obs: Observation, info: Info) -> None:
        """Record the new observation (usually as a user message)."""

    # ---- tool hooks (native support) ------------------------------------
    
    def add_assistant_step(self, content: Optional[str], tool_calls: Optional[List[Dict[str, Any]]]) -> None:
        """
        Record an assistant turn that might contain thoughts (content) AND/OR tool calls.
        This is used by agents (like ReActAgent) that perform intermediate reasoning steps.
        """
        msg = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> None:
        """
        Record the output of a tool execution.
        Standard format for OpenAI/vLLM tool interactions.
        """
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })

    # ---- convenience ----------------------------------------------------
    @property
    def messages(self) -> List[Message]:
        return list(self._messages)