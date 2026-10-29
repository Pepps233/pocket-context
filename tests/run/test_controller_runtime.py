import json
import os
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

import pytest

from pocketcontextagent.run.benchmarks.utils.controller_runtime import RUNTIME_PYTHON, RUNTIME_REQUIREMENTS

_INSTALLED = {item.metadata["Name"].lower(): item.version for item in distributions()}
_PINNED_RUNTIME_AVAILABLE = all(
    _INSTALLED.get(requirement.split("==")[0].split("[")[0]) == requirement.split("==")[1]
    for requirement in RUNTIME_REQUIREMENTS
)


@pytest.mark.skipif(not _PINNED_RUNTIME_AVAILABLE, reason="Requires the optional pinned Python 3.11 controller runtime")
def test_runtime_real_imports_and_pinned_versions_without_network() -> None:
    assert RUNTIME_PYTHON == "3.11"
    assert RUNTIME_REQUIREMENTS == (
        "litellm[mcp]==1.106.2",
        "mcp==2.3.0",
        "httpx2==2.16.0",
        "openai==2.54.0",
        "pydantic==2.14.1",
        "httpx==0.28.1",
        "tenacity==9.2.1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            """
import json
import os
import sys

blocked = []
def audit(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
        blocked.append(event)
        raise RuntimeError("Network forbidden during runtime preflight")
    if event == "socket.bind":
        raise RuntimeError("Socket binding forbidden during runtime preflight")
sys.addaudithook(audit)
os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
from pocketcontextagent.run.benchmarks.utils.controller_runtime import check_runtime

result = check_runtime()
assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
assert not blocked, blocked
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
assert not LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(
    tools=[{"type": "function", "function": {"name": "preflight", "parameters": {"type": "object"}}}]
)
print(json.dumps(result))
""",
        ],
        env={**os.environ, "MSWEA_SILENT_STARTUP": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    report = json.loads(result.stdout)
    assert report["versions"] == {
        requirement.split("==")[0].split("[")[0]: requirement.split("==")[1] for requirement in RUNTIME_REQUIREMENTS
    }
    assert report["inference_calls"] == 0
    assert set(report["imports"]) == {
        "litellm.responses.mcp.chat_completions_handler.acompletion_with_mcp",
        "litellm.responses.mcp.litellm_proxy_mcp_handler.LiteLLM_Proxy_MCP_Handler",
        "litellm.llms.deepseek.chat.transformation.DeepSeekChatConfig",
        "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler",
        "litellm.llms.openai.openai.OpenAIChatCompletion",
        "openai.OpenAI",
        "openai.AsyncOpenAI",
    }


def test_controller_image_installs_pinned_extra_before_workload() -> None:
    pytest.importorskip("daytona")
    from pocketcontextagent.run.benchmarks.benchmark_controller import _runtime_image

    recipe = _runtime_image(Path(__file__).resolve().parents[2]).dockerfile()
    assert "FROM python:3.11-slim-bookworm" in recipe
    assert all(requirement in recipe for requirement in RUNTIME_REQUIREMENTS)
    assert recipe.index("litellm[mcp]==") < recipe.index("WORKDIR /workspace/pocket-controller")


def test_runtime_missing_distribution_keeps_metadata_exception() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            "-c",
            """
import json
import runpy
import sys
from importlib.metadata import PackageNotFoundError

try:
    runpy.run_path(sys.argv[1])["check_runtime"]()
except PackageNotFoundError as error:
    print(json.dumps({"type": type(error).__name__, "distribution": error.name}))
else:
    raise AssertionError("A Python process without site-packages must fail preflight")
""",
            str(Path(__file__).resolve().parents[2] / "pocketcontextagent/run/benchmarks/utils/controller_runtime.py"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert json.loads(result.stdout) == {"type": "PackageNotFoundError", "distribution": "litellm"}
