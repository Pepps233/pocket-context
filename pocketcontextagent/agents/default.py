"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field

from pocketcontextagent import Environment, Model, __version__
from pocketcontextagent.agents.context import ContextManager, SelectionError, relationships
from pocketcontextagent.agents.utils.context_budget import append_text, estimate_tokens, output_token_setting
from pocketcontextagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, Submitted, TimeExceeded
from pocketcontextagent.models.utils.compact_tool import CONTEXT_TOOLS
from pocketcontextagent.models.utils.response import response_text
from pocketcontextagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in pocketcontextagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    context_db_path: Path | None = None
    """SQLite context archive; defaults to a separate database for each run."""
    compact_min_chars: int = Field(default=2000, ge=0)
    compact_min_reduction: int = Field(default=512, ge=0)
    context_token_limit: int = Field(default=0, ge=0)
    """Working context window, including the next response; 0 disables accounting."""
    response_token_limit: int = Field(default=16384, gt=0)
    """Output reservation when model_kwargs does not specify an output-token limit."""
    context_limit_template: str = (
        "Your {{context_token_limit}} token context window has reached 95% of its budget. "
        "Stop searching and conclude the investigation now. Do not call any tools or manage memory. "
        "Review the available evidence and produce your best final answer in the required answer format. "
        "This is your final generation; the episode ends immediately after your response."
    )
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.context = ContextManager(self.config.context_db_path)
        self._context_results: dict[tuple[int, str], list[dict]] = {}
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        self._final_generation = False

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict, origins: list[dict] | None = None) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        for message, origin in zip(messages, origins or [{} for _ in messages], strict=True):
            self.context.ingest(message, **origin)
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.context = ContextManager(self.config.context_db_path)
        self._context_results = {}
        self._final_generation = False
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                # The call was billed before parsing failed, so query() never got to charge it.
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        message = self.query()
        if self._final_generation:
            return self.finish_context_limited_episode(message)
        if getattr(self.env, "is_final_answer", None) and not message.get("extra", {}).get("actions"):
            text = response_text(message)
            if self.env.is_final_answer(text):
                raise Submitted(
                    self.model.format_message(
                        role="exit", content=text, extra={"exit_status": "Submitted", "submission": text}
                    )
                )
            raise FormatError(
                self.model.format_message(
                    role="user",
                    content="Use the available tools or provide a final answer in the required answer format.",
                )
            )
        return self.execute_actions(message)

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        options = {"allow_text_response": True} if getattr(self.env, "is_final_answer", None) else {}
        if self.config.context_token_limit:
            key, allowance = output_token_setting(self.model, self.config.response_token_limit)
            if self.context_tokens(self.context.projection()) + allowance >= self.config.context_token_limit * 0.95:
                self._final_generation = True
                self.add_messages(
                    self.model.format_message(
                        role="user", content=self._render_template(self.config.context_limit_template)
                    )
                )
                options["final_answer"] = True
            remaining = self.config.context_token_limit - self.context_tokens(self.context.projection())
            if remaining <= 0:
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": "Context window exhausted",
                        "extra": {"exit_status": "LimitsExceeded", "submission": "", "context_limit_reached": True},
                    }
                )
            options[key] = min(allowance, remaining)
        self.n_calls += 1
        message = self.model.query(self.context.assemble(self.model), **options)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        if (results := self.dispatch_context_actions(message)) is not None:
            return results
        if update_context := getattr(self.env, "update_context", None):
            update_context(self.context)
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_observations(self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def context_tokens(self, messages: list[dict]) -> int:
        return estimate_tokens(
            self.context._prepare(self.model, messages),
            getattr(self.env, "tokenizer", None),
            text_based=hasattr(self.model.config, "action_regex"),
        )

    def add_observations(self, messages: list[dict], *, origins: list[dict] | None = None) -> list[dict]:
        if messages and self.config.context_token_limit:
            _, allowance = output_token_setting(self.model, self.config.response_token_limit)
            tokens = self.context_tokens(self.context.projection() + messages) + allowance
            latest_result = next(
                (
                    message
                    for message in reversed(messages)
                    if relationships(message)[1] or "raw_output" in message.get("extra", {})
                ),
                messages[-1],
            )
            append_text(latest_result, f"[CURRENT CONTEXT TOKEN: {tokens}]")
        return self.add_messages(*messages, origins=origins)

    def finish_context_limited_episode(self, message: dict) -> list[dict]:
        text = response_text(message)
        validate = getattr(self.env, "is_final_answer", bool)
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=text,
                extra={
                    "exit_status": "Submitted" if validate(text) else "LimitsExceeded",
                    "submission": text,
                    "context_limit_reached": True,
                },
            )
        )

    def dispatch_context_actions(self, message: dict) -> list[dict] | None:
        actions = message.get("extra", {}).get("actions", [])
        if not any(action.get("type") in (*CONTEXT_TOOLS, "rejected") for action in actions):
            return None
        result_key = (id(message), json.dumps(actions, sort_keys=True))
        if result_key in self._context_results:
            return self._context_results[result_key]

        def format_results(response, outputs):
            results = self.model.format_observation_messages(response, outputs, self.get_template_vars())
            for result, output in zip(results, outputs, strict=True):
                if result.get("type") == "function_call_output":
                    result["output"] = output["output"]
                else:
                    result["content"] = output["output"]
            return results

        origins = None
        operation = actions[0].get("type")
        try:
            if len(actions) != 1 or operation not in CONTEXT_TOOLS:
                raise SelectionError(
                    actions[0].get(
                        "error", "Context operations require exactly one call and no other tools. No actions executed."
                    )
                )
            if operation == "expand_block":
                results, origins = self.context.expand(actions[0], message, self.model, format_results)
            else:
                results = self.context.compact(
                    actions[0],
                    message,
                    self.model,
                    format_results,
                    min_chars=self.config.compact_min_chars,
                    min_reduction=self.config.compact_min_reduction,
                )
        except SelectionError as exc:
            results = format_results(
                message,
                [
                    {"output": f"{operation} error: {exc}", "returncode": -1, "exception_info": str(exc)}
                    for _ in actions
                ],
            )
        self._context_results[result_key] = self.add_observations(results, origins=origins)
        return self._context_results[result_key]

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "context": self.context.serialize(),
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
