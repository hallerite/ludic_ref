from __future__ import annotations

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

from ludic.context.full_dialog import FullDialog
from tests._mocks import MockEnv, MockAgent

# ---------------------------------------------------------------------------
# Server fixture: launch ludic.inference.vllm_server end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vllm_server(request) -> Tuple[str, int]:
    host = os.getenv("VLLM_HOST", "127.0.0.1")
    port = int(os.getenv("VLLM_PORT", "8000"))

    # Check if the test requested specific configuration (e.g. tools enabled)
    # via @pytest.mark.parametrize("vllm_server", [{"enable_tools": True}], indirect=True)
    config = getattr(request, "param", {})
    enable_tools = config.get("enable_tools", False)

    env = os.environ.copy()
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

    if enable_tools:
        cmd.extend([
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ])

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for /health to return 200 or time out
    health_url = f"http://{host}:{port}/health"
    deadline = time.time() + 180.0

    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=2.0)
            if r.status_code == 200:
                break
        except Exception as e:  # noqa: BLE001
            last_err = e
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

    yield host, port

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


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

@pytest.fixture
def env_registry():
    """
    Registry mapping env kind -> factory. Used by RolloutEngine tests.
    """
    return {
        "mock": lambda **kwargs: MockEnv(**kwargs),
    }


@pytest.fixture
def ctx_registry():
    """
    Registry mapping ctx kind -> factory. Used by RolloutEngine tests.
    """
    return {
        "full_dialog": lambda **kwargs: FullDialog(**kwargs),
    }


@pytest.fixture
def mock_agent():
    """
    Real Agent wired to MockClient (from tests._mocks).
    """
    return MockAgent()