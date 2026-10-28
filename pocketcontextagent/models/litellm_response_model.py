import logging
import time
from collections.abc import Callable

import litellm

from pocketcontextagent.comparisons.tools import corpus_tools
from pocketcontextagent.exceptions import FormatError
from pocketcontextagent.models import GLOBAL_MODEL_STATS
from pocketcontextagent.models.litellm_model import LitellmModel, LitellmModelConfig
from pocketcontextagent.models.utils.actions_toolcall_response import (
    BASH_TOOL_RESPONSE_API,
    finish_reason_from_responses_api,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)
from pocketcontextagent.models.utils.compact_tool import context_tools, native_request_kwargs
from pocketcontextagent.models.utils.response import format_input_message, has_tool_calls
from pocketcontextagent.models.utils.retry import retry

logger = logging.getLogger("litellm_response_model")


class LitellmResponseModelConfig(LitellmModelConfig):
    pass


class LitellmResponseModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Flatten response objects into their output items for stateless API calls."""
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result

    def get_tools(self) -> list[dict]:
        if hasattr(self.config, "action_regex"):
            return []
        research = (
            corpus_tools(self.config.context_tools, responses_api=True)
            if self.config.research_tools == "corpus"
            else [BASH_TOOL_RESPONSE_API]
        )
        return [*research, *context_tools(self.config.context_tools, responses_api=True)]

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.responses(
                model=self.config.model_name,
                input=messages,
                **native_request_kwargs(
                    tools=self.get_tools(), model_kwargs=self.config.model_kwargs, overrides=kwargs, responses_api=True
                ),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        final_answer = kwargs.pop("final_answer", False)
        allow_text_response = kwargs.pop("allow_text_response", False)
        if final_answer and not hasattr(self.config, "action_regex"):
            kwargs["tool_choice"] = "none"
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        try:
            actions = (
                []
                if final_answer
                or (
                    allow_text_response
                    and not has_tool_calls(response, action_regex=getattr(self.config, "action_regex", ""))
                )
                else self._parse_actions(response)
            )
        except FormatError as e:
            e.messages[0]["extra"].update(cost_output)
            # hasattr guard: litellm.responses() returns a pydantic object, but tests
            # may inject a plain dict; dict(response) is the correct fallback in that case.
            # Inner try: if serialization itself fails, repr() guarantees the key is always set.
            try:
                e.messages[0]["extra"]["response"] = (
                    response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
                )
            except Exception:
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        message["extra"] = {
            "actions": actions,
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _parse_actions(self, response) -> list[dict]:
        return parse_toolcall_actions_response(
            response.get("output", []) if isinstance(response, dict) else response.output,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": finish_reason_from_responses_api(response)},
        )

    def format_message(self, **kwargs) -> dict:
        return format_input_message(**super().format_message(**kwargs))

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
