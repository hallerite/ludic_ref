import os
import signal
import subprocess
import sys
import time
from typing import Tuple

import pytest
import requests
from requests import ConnectionError as RequestsConnectionError

from ludic.inference.vllm_client import VLLMChatClient
from ludic.inference.sampling import (
    get_default_sampling_config,
    SamplingConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# ---------------------------------------------------------------------------
# Server fixture: launch ludic.inference.vllm_server end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def vllm_server() -> Tuple[str, int]:
    """
    Start the ludic.inference.vllm_server in a subprocess and wait for /health.

    Uses:
        python -m ludic.inference.vllm_server \
            --model Qwen/Qwen/Qwen2.5-0.5B-Instruct \
            --gpu_memory_utilization 0.7 \
            --max-model-len 4096 \
            --max-num-seqs 4 \
            --max-num-batched-tokens 4096 \
            --enforce-eager

    Returns (host, port) for the running server.
    """
    host = os.getenv("VLLM_HOST", "127.0.0.1")
    port = int(os.getenv("VLLM_PORT", "8000"))

    env = os.environ.copy()
    # Best-effort determinism knobs for V1; online is still not strictly reproducible.
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    cmd = [
        sys.executable,
        "-m",
        "ludic.inference.vllm_server",
        "--model",
        os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        "--host",
        host,
        "--port",
        str(port),
        "--gpu_memory_utilization",
        "0.7",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "4096",
        "--enforce-eager",
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for /health to return 200 or time out
    health_url = f"http://{host}:{port}/health"
    deadline = time.time() + 180.0  # 3 minutes max

    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=2.0)
            if r.status_code == 200:
                break
        except Exception as e:  # noqa: BLE001
            last_err = e
        # Check if process died
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"vLLM server exited early with code {proc.returncode}\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        time.sleep(2.0)
    else:
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=10)
        raise RuntimeError(
            f"vLLM server failed to become healthy at {health_url}\n"
            f"Last error: {last_err}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    # Yield host, port to tests
    yield host, port

    # Teardown: try graceful shutdown, then kill
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Client + model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def vllm_host_port(vllm_server: Tuple[str, int]) -> Tuple[str, int]:
    return vllm_server


@pytest.fixture(scope="session")
def vllm_model_name() -> str:
    return os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="session")
def vllm_client(vllm_host_port: Tuple[str, int]) -> VLLMChatClient:
    host, port = vllm_host_port
    try:
        return VLLMChatClient(
            host=host,
            port=port,
            connection_timeout_s=5.0,
            enable_weight_updates=False,
        )
    except RequestsConnectionError:
        pytest.skip(f"vLLM server not reachable at {host}:{port}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vllm_client_completion_roundtrip(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    Basic completion sanity check against the live vLLM server.
    """
    sampling = get_default_sampling_config()

    messages = [
        {"role": "system", "content": "You are a test assistant. Answer very briefly."},
        {"role": "user", "content": "Reply with the single word 'pong'."},
    ]

    resp, info = await vllm_client.complete(
        model=vllm_model_name,
        messages=messages,
        sampling=sampling,
    )

    assert isinstance(resp.text, str)
    assert resp.text.strip() != ""

    used_args = info.get("used_args", {})
    assert used_args.get("temperature") == sampling.temperature
    assert used_args.get("max_tokens") == sampling.max_tokens

    assert "raw_response" in info
    assert isinstance(info["raw_response"], dict)


@pytest.mark.asyncio
async def test_vllm_client_same_seed_is_deterministic(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    With a fixed seed and deterministic sampling settings, we expect
    fully deterministic outputs: same text, same token IDs, same
    prompt token IDs.
    """
    base = get_default_sampling_config()
    seed_value = 1234

    sampling_with_seed = SamplingConfig(
        seed=seed_value,
        temperature=0.0,
        max_tokens=base.max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=base.stop,
        extras=base.extras,
    )

    messages = [
        {"role": "system", "content": "You are deterministic for a fixed seed."},
        {"role": "user", "content": "Say a short random-looking sentence."},
    ]

    results = []
    for _ in range(3):
        resp, _ = await vllm_client.complete(
            model=vllm_model_name,
            messages=messages,
            sampling=sampling_with_seed,
            return_token_ids=True,
        )
        assert resp.text.strip() != ""
        results.append(resp)

    r0 = results[0]
    for r in results[1:]:
        assert r.text == r0.text
        assert r.token_ids == r0.token_ids
        assert r.prompt_token_ids == r0.prompt_token_ids




@pytest.mark.asyncio
async def test_vllm_client_returns_token_ids_and_detok(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    Ensure return_token_ids=True yields token IDs + prompt token IDs,
    and detokenize them for debug visibility.
    """

    sampling = get_default_sampling_config()

    messages = [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Say 'hello'."},
    ]

    resp, _ = await vllm_client.complete(
        model=vllm_model_name,
        messages=messages,
        sampling=sampling,
        return_token_ids=True,
    )

    assert resp.text.strip() != ""
    assert resp.token_ids is not None
    assert resp.prompt_token_ids is not None

@pytest.mark.asyncio
async def test_vllm_global_think_processor_triggers_at_very_small_max_think(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    Stronger sanity check: with interrupt_thinking=1, the logits processor
    should immediately start forcing the '</think>' sequence, so the closing
    tag should appear right at or very near the beginning of the completion.
    """
    from transformers import AutoTokenizer

    interrupt_thinking = 1  # trigger almost immediately

    base = get_default_sampling_config()
    sampling = SamplingConfig(
        seed=123,
        temperature=base.temperature,
        max_tokens=base.max_tokens,
        top_p=base.top_p,
        frequency_penalty=base.frequency_penalty,
        presence_penalty=base.presence_penalty,
        stop=base.stop,
        extras=base.extras,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a chain-of-thought model. Always wrap your reasoning "
                "between <think> and </think>, then give the final answer."
            ),
        },
        {
            "role": "user",
            "content": "Explain briefly why 2 + 2 = 4, then answer.",
        },
    ]

    resp, _ = await vllm_client.complete(
        model=vllm_model_name,
        messages=messages,
        sampling=sampling,
        interrupt_thinking=interrupt_thinking,
        return_token_ids=True,
    )

    assert resp.text.strip() != ""
    assert resp.token_ids is not None
    completion_ids = resp.token_ids

    tokenizer = AutoTokenizer.from_pretrained(
        vllm_model_name,
        trust_remote_code=True,
    )
    closing_ids = tokenizer.encode("</think>", add_special_tokens=False)
    assert closing_ids, "Expected non-empty tokenization for '</think>'."

    def find_subseq(haystack: list[int], needle: list[int]) -> int:
        if not needle:
            return -1
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i : i + len(needle)] == needle:
                return i
        return -1

    closing_idx = find_subseq(completion_ids, closing_ids)
    assert closing_idx != -1, "Expected '</think>' token sequence in completion."

    # With max_think=1 we expect the processor to fire essentially immediately.
    # Depending on how many tokens the model emits before '</think>', we allow a
    # tiny tolerance, but it should be at or right after the first token.
    assert closing_idx <= 2, (
        f"Expected '</think>' to appear very early (<=2), got index {closing_idx}."
    )
