import pytest
from portkey_ai import Portkey

from pocketcontextagent.agents.utils.context_budget import request_tools
from pocketcontextagent.models.portkey_model import PortkeyModel, PortkeyModelConfig
from pocketcontextagent.models.portkey_response_model import PortkeyResponseAPIModel, PortkeyResponseAPIModelConfig
from pocketcontextagent.models.requesty_model import RequestyModel
from tests.agents.test_context_provider import responses_server as responses_server


@pytest.mark.parametrize(("backend",), [("portkey",), ("portkey_responses",), ("requesty",)])  # noqa: PT006
def test_pocket_adapters_send_filtered_tools_without_changing_defaults(responses_server, backend):
    url, requests, outputs = responses_server

    class LocalPortkey(PortkeyModel):
        def __init__(self):
            self.config = PortkeyModelConfig(model_name="gpt-4.1")
            self.client = Portkey(api_key="local-test", base_url=url)

    class LocalPortkeyResponses(PortkeyResponseAPIModel):
        def __init__(self):
            self.config = PortkeyResponseAPIModelConfig(model_name="gpt-4.1")
            self.client = Portkey(api_key="local-test", base_url=url)

    if backend == "requesty":
        model = RequestyModel(model_name="gpt-4.1")
        model._api_key = "local-test"
        model._api_url = url + "/chat/completions"
    else:
        model = LocalPortkeyResponses() if backend == "portkey_responses" else LocalPortkey()
    for compact in (False, True):
        outputs.append(
            {
                "id": "chat-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4.1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
            }
            if backend != "portkey_responses"
            else [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done"}]}]
        )
        model._query([model.format_message(role="user", content="Work")], tools=request_tools(model, compact=compact))
    assert [tool.get("function", tool)["name"] for tool in requests[0]["tools"]] == ["bash", "expand_block"]
    assert [tool.get("function", tool)["name"] for tool in requests[1]["tools"]] == [
        "bash",
        "compact_block",
        "expand_block",
    ]
    assert "tools" not in model.config.model_kwargs
