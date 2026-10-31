"""Pinned controller inference dependencies and an import-only runtime check."""

import os
from importlib import import_module
from importlib.metadata import version

RUNTIME_PYTHON = "3.11"
RUNTIME_REQUIREMENTS = (
    "litellm[mcp]==1.106.2",
    "mcp==2.3.0",
    "httpx2==2.16.0",
    "openai==2.54.0",
    "pydantic==2.14.1",
    "httpx==0.28.1",
    "tenacity==9.2.1",
    "transformers==5.20.2",
    "tokenizers==0.23.3",
    "huggingface-hub==1.33.0",
)
_RUNTIME_IMPORTS = (
    ("litellm.responses.mcp.chat_completions_handler", "acompletion_with_mcp"),
    ("litellm.responses.mcp.litellm_proxy_mcp_handler", "LiteLLM_Proxy_MCP_Handler"),
    ("litellm.llms.deepseek.chat.transformation", "DeepSeekChatConfig"),
    ("litellm.llms.custom_httpx.llm_http_handler", "BaseLLMHTTPHandler"),
    ("litellm.llms.openai.openai", "OpenAIChatCompletion"),
    ("openai", "OpenAI"),
    ("openai", "AsyncOpenAI"),
    ("transformers", "AutoTokenizer"),
)


def check_runtime() -> dict:
    """Fail on dependency drift or broken lazy imports without requesting inference."""
    versions = {}
    for requirement in RUNTIME_REQUIREMENTS:
        distribution, expected = requirement.split("==")
        distribution = distribution.split("[", 1)[0]
        versions[distribution] = version(distribution)
        if versions[distribution] != expected:
            raise ValueError(f"Controller runtime requires {requirement}; installed {versions[distribution]}")

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    for module, attribute in _RUNTIME_IMPORTS:
        if not callable(getattr(import_module(module), attribute)):
            raise TypeError(f"Controller runtime import is not callable: {module}.{attribute}")
    return {
        "versions": versions,
        "imports": [f"{module}.{attribute}" for module, attribute in _RUNTIME_IMPORTS],
        "inference_calls": 0,
    }
