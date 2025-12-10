import pytest
from ludic.context.truncated_thinking import TruncatedThinkingContext

def test_truncates_valid_format():
    """
    Test that a standard <think> ... </think> Answer format is correctly truncated.
    """
    ctx = TruncatedThinkingContext(placeholder="[GONE]")
    
    # Simulate adding an assistant message to history
    ctx._messages.append({
        "role": "assistant", 
        "content": "<think>This is a long reasoning chain.</think> This is the answer."
    })

    # Get the view sent to the model
    view = ctx.on_before_act()
    
    assert len(view) == 1
    # Check that content was replaced
    assert view[0]["content"] == "<think>[GONE]</think> This is the answer."

def test_truncates_multiline():
    """Test that it handles newlines inside the think block correctly."""
    ctx = TruncatedThinkingContext(placeholder="[GONE]")
    
    multiline_content = """<think>
    Step 1: Analyze.
    Step 2: Solve.
    </think>
    The Answer."""
    
    ctx._messages.append({"role": "assistant", "content": multiline_content})
    view = ctx.on_before_act()

    # We expect the tags to be preserved, but inner content replaced
    expected = "<think>[GONE]</think>\n    The Answer."
    assert view[0]["content"] == expected

def test_preserves_missing_answer():
    """
    If the model stopped generating after </think> (or crashed), 
    we must NOT truncate, so we can see the hanging state.
    """
    ctx = TruncatedThinkingContext()
    
    # No answer after the closing tag
    bad_content = "<think>I am thinking...</think>"
    
    ctx._messages.append({"role": "assistant", "content": bad_content})
    view = ctx.on_before_act()

    # Should remain identical
    assert view[0]["content"] == bad_content

def test_preserves_content_before_tag():
    """
    If there is text BEFORE <think>, it's not a standard CoT prefix.
    It should be preserved.
    """
    ctx = TruncatedThinkingContext()
    
    content = "Wait! <think>I need to think.</think> Answer."
    ctx._messages.append({"role": "assistant", "content": content})
    view = ctx.on_before_act()

    assert view[0]["content"] == content

def test_preserves_malformed_tags():
    """Test handling of broken XML tags."""
    ctx = TruncatedThinkingContext()
    
    # Missing closing slash
    case1 = "<think>Thinking... <think> Answer"
    # Missing opening tag
    case2 = "Thinking...</think> Answer"
    
    ctx._messages.append({"role": "assistant", "content": case1})
    ctx._messages.append({"role": "assistant", "content": case2})
    
    view = ctx.on_before_act()
    
    assert view[0]["content"] == case1
    assert view[1]["content"] == case2

def test_ignores_user_and_system_messages():
    """Ensure non-assistant messages are never touched."""
    ctx = TruncatedThinkingContext()
    
    user_msg = {"role": "user", "content": "<think>Don't touch me</think> Answer"}
    sys_msg = {"role": "system", "content": "<think>System rules</think> Answer"}
    
    ctx._messages.extend([user_msg, sys_msg])
    view = ctx.on_before_act()

    assert view[0] == user_msg
    assert view[1] == sys_msg

def test_whitespace_tolerance():
    """Test that leading/trailing whitespace around tags is handled."""
    ctx = TruncatedThinkingContext(placeholder="[GONE]")
    
    # Spaces before <think> and newlines before Answer
    content = "   <think>Thinking</think> \n\n Answer"
    
    ctx._messages.append({"role": "assistant", "content": content})
    view = ctx.on_before_act()
    
    # The regex ^(\s*<think>) captures the leading spaces, preserving them.
    # (</think>\s*) captures the trailing spaces/newlines, also preserving format.
    assert "[GONE]" in view[0]["content"]
    assert "   <think>" in view[0]["content"] 
    assert "Answer" in view[0]["content"]

def test_multiple_think_blocks():
    """
    Edge case: If there are two think blocks, the regex (non-greedy) 
    should match the FIRST one as the prefix.
    """
    ctx = TruncatedThinkingContext(placeholder="[GONE]")
    
    content = "<think>First thought</think> <think>Second thought</think> Answer"
    ctx._messages.append({"role": "assistant", "content": content})
    view = ctx.on_before_act()

    # Because strict format requires `(.*?)` (non-greedy) followed by `(</think>\s*)`,
    # it matches the first block. The rest (`<think>Second...`) becomes part of the "Answer" group.
    # This is acceptable behavior for a "Prefix Truncator".
    expected = "<think>[GONE]</think> <think>Second thought</think> Answer"
    assert view[0]["content"] == expected