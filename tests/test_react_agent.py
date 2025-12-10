from __future__ import annotations
import json
import pytest
from typing import Any, List, Dict, Optional, Tuple

from ludic.agents.react_agent import ReActAgent
from ludic.context.full_dialog import FullDialog
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.sampling import SamplingConfig
from ludic.types import Message
from ludic.parsers import xml_move_parser, ParseResult

# ---------------------------------------------------------------------
# Mocks & Helpers
# ---------------------------------------------------------------------

class ReplayMockClient(ChatClient):
    """
    A mock client that replays a sequence of pre-defined responses.
    It simulates the vLLM raw_response structure required by ReActAgent.
    """
    def __init__(self, steps: List[Dict[str, Any]]):
        # Steps is a list of dicts representing the "message" content
        # e.g. [{"content": "...", "tool_calls": [...]}, ...]
        self.steps = steps
        self.call_count = 0
        self.last_sampling_args: Optional[SamplingConfig] = None
        self.last_messages: Optional[List[Message]] = None

    async def complete(
        self,
        *,
        model: str,
        messages: List[Message],
        sampling: SamplingConfig,
        **kwargs,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        
        if self.call_count >= len(self.steps):
            raise RuntimeError("ReplayMockClient exhausted all steps")

        step_data = self.steps[self.call_count]
        self.call_count += 1
        
        self.last_sampling_args = sampling
        self.last_messages = messages

        # Construct ChatResponse (High level)
        text = step_data.get("content") or ""
        
        # Construct the raw vLLM/OpenAI structure (Low level)
        # ReActAgent inspects info["raw_response"]["choices"][0]["message"]
        raw_response = {
            "choices": [
                {
                    "message": step_data,
                    "finish_reason": "tool_calls" if step_data.get("tool_calls") else "stop"
                }
            ]
        }

        resp = ChatResponse(text=text)
        return resp, {"raw_response": raw_response}

    def sync_weights(self, *args, **kwargs):
        pass

def calculator_tool(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

def weather_tool(location: str) -> str:
    """Gets weather."""
    return f"Sunny in {location}"

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_react_agent_happy_path_single_tool():
    """
    Scenario:
    1. Agent Thinks -> Calls Calculator
    2. Tool executes -> Returns result
    3. Agent Thinks -> Returns Final Answer
    """
    
    # Step 1: Model wants to call tool
    step_1 = {
        "content": "I need to calculate 2+2.",
        "tool_calls": [
            {
                "id": "call_123",
                "function": {
                    "name": "calculator_tool",
                    "arguments": json.dumps({"a": 2, "b": 2})
                }
            }
        ]
    }
    
    # Step 2: Model sees result "4" and answers
    step_2 = {
        "content": "The answer is <move>4</move>",
        "tool_calls": None
    }

    client = ReplayMockClient([step_1, step_2])
    
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_move_parser,
        tools=[calculator_tool]
    )

    parse_result, raw_text, _ = await agent.act(sampling_args={})

    # Assert Final Output
    assert parse_result.action == "4"
    assert raw_text == "The answer is <move>4</move>"
    
    # Assert Context State
    # 0: User (Env reset - assumed handled by protocol, but we called act directly so context had prompts)
    # Actually, let's check the messages sent on the FINAL call
    final_history = client.last_messages
    
    # We expect: 
    # ... System/User prompt ...
    # Assistant: "I need to calculate..." + tool_calls
    # Tool: "4"
    
    assert final_history[-2]["role"] == "assistant"
    assert final_history[-2]["tool_calls"][0]["function"]["name"] == "calculator_tool"
    
    assert final_history[-1]["role"] == "tool"
    assert final_history[-1]["content"] == "4"
    assert final_history[-1]["tool_call_id"] == "call_123"


@pytest.mark.asyncio
async def test_react_agent_shot_clock_fallback():
    """
    Scenario:
    - max_react_steps = 2
    - Step 1: Tool Call
    - Step 2: (Limit reached) -> Agent should FORCE final text, strip tools.
    """
    
    # Step 1: Tool Call
    step_1 = {
        "content": "Thinking...",
        "tool_calls": [
            {
                "id": "call_A",
                "function": {"name": "weather_tool", "arguments": '{"location": "NYC"}'}
            }
        ]
    }
    
    # Step 2: Model tries to call tool again? 
    # NO, the Agent logic should prevent the model from seeing tools in args.
    # The mock will simulate the model reacting to the "You must output now" prompt.
    step_2 = {
        "content": "Okay, fine. <move>Sunny</move>",
        "tool_calls": None
    }

    client = ReplayMockClient([step_1, step_2])
    
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_move_parser,
        tools=[weather_tool],
        max_react_steps=2 # Strict limit
    )

    result, _, _ = await agent.act(sampling_args={})

    # 1. Did it finish?
    assert result.action == "Sunny"

    # 2. Was the "Shot Clock" logic triggered?
    # On the last call, "tools" should NOT be in extras
    last_args = client.last_sampling_args
    assert "tools" not in (last_args.extras or {})
    assert "tool_choice" not in (last_args.extras or {})

    # 3. Was the forced message injected?
    last_msgs = client.last_messages
    assert last_msgs[-1]["role"] == "user"
    assert "exhausted your reasoning steps" in last_msgs[-1]["content"]


@pytest.mark.asyncio
async def test_react_agent_handles_bad_tool_calls():
    """
    Scenario: Model tries to call a non-existent tool or passes bad JSON.
    Agent should record the error in context and let model try again.
    """
    
    # Step 1: Call missing tool
    step_1 = {
        "content": "Trying weird tool",
        "tool_calls": [{
            "id": "call_bad",
            "function": {"name": "ghost_tool", "arguments": "{}"}
        }]
    }
    
    # Step 2: Model sees error, fixes it
    step_2 = {
        "content": "Oops. <move>Fixed</move>",
        "tool_calls": None
    }

    client = ReplayMockClient([step_1, step_2])
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_move_parser,
        tools=[calculator_tool] # ghost_tool not here
    )

    await agent.act(sampling_args={})

    # Check that the error was added to context
    # Messages sent during Step 2 should include the Tool output from Step 1
    msgs = client.last_messages
    tool_msg = msgs[-1]
    
    assert tool_msg["role"] == "tool"
    assert "Error: Tool ghost_tool not found" in tool_msg["content"]


@pytest.mark.asyncio
async def test_react_agent_requires_tool_supported_context():
    """
    Ensures the agent raises TypeError if init'd with a context 
    that doesn't support tools.
    """
    
    class DumbDialog(FullDialog):
        @property
        def supports_tools(self) -> bool:
            return False

    with pytest.raises(TypeError, match="requires a context with supports_tools=True"):
        ReActAgent(
            client=ReplayMockClient([]),
            model="mock",
            ctx=DumbDialog(), # <-- The culprit
            parser=xml_move_parser,
            tools=[]
        )