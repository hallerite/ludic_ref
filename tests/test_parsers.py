from functools import partial

import pytest

from ludic.parsers import (
    ParseResult,
    cot_prefix_parser,
    boxed_parser,
    xml_move_parser,
    compose_parsers,
)


# ---------------------------------------------------------------------
# cot_prefix_parser tests
# ---------------------------------------------------------------------

def test_cot_prefix_parser_success():
    raw = "<think>reasoning</think>  ANSWER"
    r = cot_prefix_parser(raw)

    assert isinstance(r, ParseResult)
    assert r.action == "ANSWER"
    assert r.reward == pytest.approx(0.1)
    assert r.obs is None


def test_cot_prefix_parser_allows_whitespace_newlines():
    raw = """
        <think>
            foo
        </think>

        Final answer here
    """
    r = cot_prefix_parser(raw)

    assert r.action == "Final answer here"
    assert r.reward == pytest.approx(0.1)
    assert r.obs is None


def test_cot_prefix_parser_fails_without_think_prefix():
    raw = "no think tag at all"
    r = cot_prefix_parser(raw)

    assert r.action is None
    assert r.reward < 0.0  # penalized
    assert "Invalid CoT" in r.obs


def test_cot_prefix_parser_fails_on_empty_answer():
    raw = "<think>abc</think>   "
    r = cot_prefix_parser(raw)

    assert r.action is None
    assert r.reward < 0.0
    assert "Missing answer" in r.obs


def test_cot_prefix_parser_custom_rewards():
    parser = partial(cot_prefix_parser, success_reward=0.25, error_reward=-0.25)

    success = parser("<think>foo</think> bar")
    assert success.reward == pytest.approx(0.25)

    failure = parser("invalid format")
    assert failure.reward == pytest.approx(-0.25)


# ---------------------------------------------------------------------
# xml_move_parser tests
# ---------------------------------------------------------------------

def test_xml_move_parser_success():
    raw = "<move>  A1  </move>"
    r = xml_move_parser(raw)

    assert r.action == "A1"
    assert r.reward == pytest.approx(0.1)
    assert r.obs is None


def test_xml_move_parser_is_case_insensitive():
    raw = "<MoVe> b3 </MoVe>"
    r = xml_move_parser(raw)

    assert r.action == "b3"
    assert r.reward == pytest.approx(0.1)


def test_xml_move_parser_fails_without_move_tag():
    raw = "A1"
    r = xml_move_parser(raw)

    assert r.action is None
    assert r.reward < 0.0
    assert "Invalid action format" in r.obs


def test_xml_move_parser_fails_on_empty_move():
    raw = "<move></move>"
    r = xml_move_parser(raw)

    assert r.action is None
    assert r.reward < 0.0
    assert "Empty" in r.obs or "empty" in r.obs


def test_xml_move_parser_custom_penalty():
    parser = partial(xml_move_parser, error_reward=-0.5)
    r = parser("bad move format")

    assert r.action is None
    assert r.reward == pytest.approx(-0.5)


# ---------------------------------------------------------------------
# compose_parsers tests
# ---------------------------------------------------------------------

def test_compose_parsers_success_chain():
    parser = compose_parsers(cot_prefix_parser, xml_move_parser)

    raw = "<think>blah</think> <move> C2 </move>"
    r = parser(raw)

    assert r.action == "C2"
    assert r.reward == pytest.approx(0.2)
    assert r.obs is None


def test_compose_parsers_stops_on_first_failure():
    parser = compose_parsers(cot_prefix_parser, xml_move_parser)

    # fails at CoT parser, so xml_move_parser is never called
    raw = "not a cot structure"
    r = parser(raw)

    assert r.action is None
    assert r.reward < 0.0
    assert "Invalid CoT" in (r.obs or "")


def test_compose_parsers_fails_on_second_parser():
    parser = compose_parsers(cot_prefix_parser, xml_move_parser)

    # valid CoT but invalid move
    raw = "<think>ok</think> not a move tag"
    r = parser(raw)

    assert r.action is None
    assert r.reward < 0.0  # accumulated
    assert "Invalid action format" in (r.obs or "")


def test_compose_parsers_accumulates_rewards():
    # Fake parsers to test reward accumulation explicitly

    def p1(x: str) -> ParseResult:
        return ParseResult(action=x + "X", reward=0.5, obs=None)

    def p2(x: str) -> ParseResult:
        return ParseResult(action=x + "Y", reward=1.0, obs=None)

    parser = compose_parsers(p1, p2)

    r = parser("A")

    assert r.action == "AXY"
    assert r.reward == pytest.approx(1.5)
    assert r.obs is None


def test_compose_parsers_failure_reward_accumulates():
    # First parser succeeds, second fails
    def good(x: str) -> ParseResult:
        return ParseResult(action=x, reward=0.3, obs=None)

    def bad(x: str) -> ParseResult:
        return ParseResult(action=None, reward=-2.0, obs="bad here")

    parser = compose_parsers(good, bad)
    r = parser("START")

    assert r.action is None
    assert r.reward == pytest.approx(0.3 - 2.0)
    assert r.obs == "bad here"


# ---------------------------------------------------------------------
# boxed_parser tests
# ---------------------------------------------------------------------

def test_boxed_parser_extracts_simple_box():
    r = boxed_parser("final: \\boxed{42}")
    assert r.action == "42"
    assert r.reward == pytest.approx(0.1)
    assert r.obs is None


def test_boxed_parser_supports_nested_braces():
    r = boxed_parser("answer \\boxed{\\frac{1}{2}}")
    assert r.action == "\\frac{1}{2}"
    assert r.reward == pytest.approx(0.1)


def test_boxed_parser_uses_last_box():
    r = boxed_parser("intermediate \\boxed{0} final \\boxed{1}")
    assert r.action == "1"


def test_boxed_parser_fails_without_box():
    r = boxed_parser("no box here")
    assert r.action is None
    assert r.reward < 0.0
