import copy
import json

import pytest

from benchmarks.utils.metrics import episode_metrics, summarize_results


def test_mixed_provider_usage_compactions_format_errors_and_tool_deduplication():
    chat = {
        "role": "assistant",
        "extra": {
            "actions": [{"command": "search jazz", "tool_call_id": "search-1"}],
            "response": {
                "id": "chat-1",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": 20},
                },
            },
        },
    }
    response = {
        "id": "resp-1",
        "object": "response",
        "usage": {
            "input_tokens": 70,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 0},
        },
        "extra": {"actions": [{"type": "compact_block"}]},
    }
    messages = [
        {"role": "system", "content": "Research"},
        chat,
        copy.deepcopy(chat),
        response,
        copy.deepcopy(response),
        {"role": "tool", "content": "Found evidence"},
        {
            "role": "user",
            "content": "Please fix the response format",
            "extra": {
                "response": {
                    "id": "compaction-error",
                    "object": "response",
                    "usage": {
                        "input_tokens": 95,
                        "output_tokens": 9,
                        "total_tokens": 104,
                        "input_tokens_details": {"cached_tokens": 30},
                    },
                }
            },
        },
    ]
    before = copy.deepcopy(messages)
    metrics = episode_metrics(messages, model_stats={"instance_cost": 0.25, "api_calls": 3}, latency_seconds=4.5)
    assert metrics == {
        "model_calls": 3,
        "calls_with_usage": 3,
        "calls_without_usage": 0,
        "total_input_tokens": 265,
        "total_output_tokens": 21,
        "total_tokens": 286,
        "total_cached_input_tokens": 50,
        "input_cache_hit_rate": 50 / 265,
        "calls_with_input_tokens": 3,
        "calls_with_output_tokens": 3,
        "calls_with_total_tokens": 3,
        "calls_with_cached_input_tokens": 3,
        "peak_input_tokens": 100,
        "peak_total_tokens": 105,
        "tool_call_counts": {"search": 1, "compact_block": 1},
        "total_tool_calls": 2,
        "cost_usd": 0.25,
        "latency_seconds": 4.5,
    }
    assert messages == before
    assert json.loads(json.dumps(metrics)) == metrics


def test_response_id_deduplication_retains_usage_from_later_copies():
    assert episode_metrics(
        [
            {"object": "response", "id": "shared", "usage": None},
            {"extra": {"response": {"id": "shared", "usage": {"input_tokens": 20}}}},
            {"object": "response", "id": "shared", "usage": {"output_tokens": 3}},
        ]
    ) == episode_metrics([{"object": "response", "id": "shared", "usage": {"input_tokens": 20, "output_tokens": 3}}])
    metrics = episode_metrics(
        [
            {"object": "response", "id": "one", "usage": {"input_tokens": 10, "output_tokens": 2}},
            {"object": "response", "id": "two", "usage": {"input_tokens": 10, "output_tokens": 2}},
        ]
    )
    assert metrics["model_calls"] == 2
    assert metrics["total_tokens"] == 24
    enriched = episode_metrics(
        [
            {"object": "response", "id": "shared", "usage": {"input_tokens": 20, "output_tokens": 3}},
            {"object": "response", "id": "shared", "usage": {"total_tokens": 25}},
        ]
    )
    assert enriched["model_calls"] == 1
    assert enriched["total_tokens"] == enriched["peak_total_tokens"] == 25


@pytest.mark.parametrize(
    ("input_tokens", "cached_tokens", "expected"),
    [(100, 0, 0.0), (100, 100, 1.0), (100, None, None), (None, 10, None), (0, 0, None), (100, 101, None)],
)
def test_cache_hit_rate_distinguishes_missing_zero_and_invalid_usage(input_tokens, cached_tokens, expected):
    messages = [
        {
            "object": "response",
            "usage": {"input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": cached_tokens}},
        }
    ]
    metrics = episode_metrics(messages)
    assert metrics["input_cache_hit_rate"] == expected
    assert summarize_results([{"metrics": metrics}])["input_cache_hit_rate"] == expected
    assert episode_metrics(messages, model_stats={"api_calls": 2})["input_cache_hit_rate"] is None
    assert summarize_results([{"metrics": metrics}, {"metrics": {}}])["input_cache_hit_rate"] is None


def test_identical_anonymous_responses_are_distinct_calls_but_references_are_not():
    response = {"object": "chat.completion", "usage": {"prompt_tokens": 8, "completion_tokens": 2}}
    metrics = episode_metrics([response, response, copy.deepcopy(response)])
    assert metrics["model_calls"] == 2
    assert metrics["total_input_tokens"] == 16
    assert metrics["total_tokens"] == 20
    assert metrics["peak_input_tokens"] == 8


@pytest.mark.parametrize(("usage",), [(None,), ({},), ({"cost": 0.2},)])  # noqa: PT006
def test_missing_usage_never_becomes_measured_zero(usage: dict | None):
    metrics = episode_metrics(
        [
            {"object": "response", "id": "measured", "usage": {"input_tokens": 15, "output_tokens": 4}},
            {"object": "response", "id": "unmeasured", "usage": usage},
        ]
    )
    assert metrics["model_calls"] == 2
    assert metrics["calls_with_usage"] == 1
    assert metrics["calls_without_usage"] == 1
    assert metrics["calls_with_input_tokens"] == 1
    for key in ("total_input_tokens", "total_output_tokens", "total_tokens", "peak_input_tokens", "peak_total_tokens"):
        assert metrics[key] is None
    assert metrics["total_cached_input_tokens"] is None
    assert metrics["cost_usd"] is None
    assert metrics["latency_seconds"] is None


def test_partial_usage_preserves_independent_measurements_and_reported_total():
    metrics = episode_metrics(
        [
            {"object": "response", "id": "input-only", "usage": {"input_tokens": 50, "total_tokens": 60}},
            {"object": "response", "id": "both", "usage": {"input_tokens": 20, "output_tokens": 0}},
        ]
    )
    assert metrics["total_input_tokens"] == 70
    assert metrics["total_output_tokens"] is None
    assert metrics["total_tokens"] == 80
    assert metrics["peak_input_tokens"] == 50
    assert metrics["peak_total_tokens"] == 60
    assert metrics["calls_with_output_tokens"] == 1
    assert metrics["calls_with_usage"] == 2


def test_unpersisted_calls_and_unserializable_format_errors_are_missing_usage():
    metrics = episode_metrics(
        [
            {"role": "assistant", "content": "No provider usage persisted", "extra": {"actions": []}},
            {"role": "user", "content": "Format error", "extra": {"response": "<unserializable response>"}},
            {"role": "tool", "content": "observation"},
        ],
        model_stats={"instance_cost": 0.1, "api_calls": 3},
    )
    assert metrics["model_calls"] == 3
    assert metrics["calls_without_usage"] == 3
    assert metrics["total_tokens"] is None
    assert metrics["peak_input_tokens"] is None
    assert metrics["cost_usd"] == 0.1


def test_explicit_zero_usage_cost_and_latency_are_measurements():
    metrics = episode_metrics(
        [
            {
                "object": "response",
                "usage": {"input_tokens": 0, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 0}},
            }
        ],
        model_stats={"instance_cost": 0.0},
        latency_seconds=0.0,
    )
    assert metrics["total_tokens"] == metrics["peak_input_tokens"] == metrics["total_cached_input_tokens"] == 0
    assert metrics["calls_with_usage"] == 1
    summary = summarize_results([{"metrics": metrics, "correct": False, "model_stats": {"instance_cost": 0.0}}])
    assert summary["cost_usd"] == summary["avg_cost_usd_per_task"] == 0
    assert summary["total_latency_seconds"] == summary["avg_latency_seconds_per_task"] == 0
    assert summary["tasks_with_cost"] == summary["tasks_with_latency"] == summary["tasks_with_total_tokens"] == 1
    assert summary["success_rate"] == 0


@pytest.mark.parametrize(
    ("grades", "success_rate", "graded_success_rate", "graded_tasks", "correct_tasks"),
    [
        ([True, False], 0.5, 0.5, 2, 1),
        ([True, True], 1.0, 1.0, 2, 2),
        ([False, False], 0.0, 0.0, 2, 0),
        ([True, None], None, 1.0, 1, 1),
        ([False, "true", 1, 0], None, 0.0, 1, 0),
        ([None, "false", 0, 1], None, None, 0, 0),
    ],
)
def test_success_requires_explicit_boolean_grades_for_all_attempted_tasks(
    grades: list, success_rate: float | None, graded_success_rate: float | None, graded_tasks: int, correct_tasks: int
):
    summary = summarize_results(
        [
            {"status": "completed" if index % 2 == 0 else "LimitsExceeded", "correct": grade}
            for index, grade in enumerate(grades)
        ]
    )
    assert summary["success_rate"] == success_rate
    assert summary["graded_success_rate"] == graded_success_rate
    assert summary["graded_tasks"] == graded_tasks
    assert summary["correct_tasks"] == correct_tasks
    assert summary["incorrect_tasks"] == graded_tasks - correct_tasks
    assert summary["ungraded_tasks"] == len(grades) - graded_tasks
    assert summary["completed"] == (len(grades) + 1) // 2


def test_completed_status_nested_grades_and_numeric_scores_are_not_correctness():
    summary = summarize_results(
        [
            {"status": "completed"},
            {"status": "completed", "metrics": {"correct": True}},
            {"status": "completed", "judge_result": {"correct": True}},
            {"status": "completed", "score": 1.0},
        ]
    )
    assert summary["completed"] == 4
    assert summary["graded_tasks"] == 0
    assert summary["success_rate"] is None
    assert summary["graded_success_rate"] is None


def test_summary_totals_averages_and_tool_counts_include_failed_tasks():
    results = [
        {
            "correct": index == 0,
            "status": "completed" if index == 0 else "LimitsExceeded",
            "metrics": episode_metrics(
                [
                    {
                        "object": "response",
                        "id": f"response-{index}",
                        "usage": {
                            "input_tokens": tokens,
                            "output_tokens": 10,
                            "input_tokens_details": {"cached_tokens": cached},
                        },
                        "extra": {"actions": [{"command": "search jazz"}, {"command": "submit answer"}]},
                    }
                ],
                model_stats={"instance_cost": cost},
                latency_seconds=latency,
            ),
            "model_stats": {"instance_cost": cost},
        }
        for index, (tokens, cached, cost, latency) in enumerate([(100, 25, 0.2, 3), (200, 40, 0.4, 7)])
    ]
    summary = summarize_results(results)
    assert summary["tasks"] == 2
    assert summary["completed"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["total_input_tokens"] == 300
    assert summary["total_output_tokens"] == 20
    assert summary["total_tokens"] == 320
    assert summary["total_cached_input_tokens"] == 65
    assert summary["input_cache_hit_rate"] == pytest.approx(65 / 300)
    assert summary["avg_input_tokens_per_task"] == 150
    assert summary["avg_output_tokens_per_task"] == 10
    assert summary["avg_total_tokens_per_task"] == 160
    assert summary["peak_input_tokens"] == 200
    assert summary["avg_peak_input_tokens"] == 150
    assert summary["cost_usd"] == pytest.approx(0.6)
    assert summary["avg_cost_usd_per_task"] == pytest.approx(0.3)
    assert summary["total_latency_seconds"] == 10
    assert summary["avg_latency_seconds_per_task"] == 5
    assert summary["tasks_with_cost"] == summary["tasks_with_latency"] == summary["tasks_with_total_tokens"] == 2
    assert summary["tool_call_counts"] == {"search": 2, "submit": 2}
    assert summary["avg_tool_calls_per_task"] == 2
    assert summary["avg_research_tool_calls_per_task"] == 1


def test_partial_summary_totals_and_averages_are_unknown_with_explicit_coverage():
    summary = summarize_results(
        [
            {
                "metrics": {
                    "total_input_tokens": 30,
                    "total_output_tokens": 5,
                    "total_tokens": 35,
                    "peak_input_tokens": 20,
                    "latency_seconds": 8,
                    "cost_usd": 999,
                },
                "model_stats": {"instance_cost": 0.4},
            },
            {"metrics": {"total_input_tokens": 10, "cost_usd": 123, "latency_seconds": None}},
        ]
    )
    assert summary["total_input_tokens"] == 40
    assert summary["avg_input_tokens_per_task"] == 20
    assert summary["tasks_with_input_tokens"] == 2
    for key in ("total_output_tokens", "total_tokens", "peak_input_tokens", "cost_usd", "total_latency_seconds"):
        assert summary[key] is None
    for key in (
        "avg_output_tokens_per_task",
        "avg_total_tokens_per_task",
        "avg_peak_input_tokens",
        "avg_cost_usd_per_task",
        "avg_latency_seconds_per_task",
    ):
        assert summary[key] is None
    for key in (
        "tasks_with_output_tokens",
        "tasks_with_total_tokens",
        "tasks_with_peak_input_tokens",
        "tasks_with_cost",
        "tasks_with_latency",
    ):
        assert summary[key] == 1
    assert summary["total_cached_input_tokens"] is None
    assert summary["tasks_with_cached_input_tokens"] == 0
    assert summary["input_cache_hit_rate"] is None


def test_legacy_metrics_derive_total_but_do_not_reinterpret_missing_usage_as_zero():
    legacy = {"total_input_tokens": 10, "total_output_tokens": 2, "peak_input_tokens": 10, "calls_with_usage": 1}
    summary = summarize_results([{"metrics": legacy}])
    assert summary["total_tokens"] == summary["avg_total_tokens_per_task"] == 12
    summary = summarize_results(
        [
            {"metrics": legacy},
            {
                "metrics": {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "peak_input_tokens": 0,
                    "calls_with_usage": 0,
                }
            },
        ]
    )
    assert summary["total_tokens"] is None
    assert summary["total_input_tokens"] is None
    assert summary["peak_input_tokens"] is None
    assert summary["avg_total_tokens_per_task"] is None
    assert summary["tasks_with_total_tokens"] == 1


@pytest.mark.parametrize(("invalid",), [(-1,), (True,), ("1.0",), (float("nan"),), (float("inf"),)])  # noqa: PT006
def test_invalid_measurements_do_not_poison_totals_or_turn_into_zeros(invalid: object):
    metrics = episode_metrics(
        [{"object": "response", "usage": {"input_tokens": invalid, "output_tokens": invalid}}],
        model_stats={"instance_cost": invalid},
        latency_seconds=invalid,
    )
    summary = summarize_results([{"metrics": metrics, "model_stats": {"instance_cost": invalid}}])
    assert metrics["total_input_tokens"] is None
    assert metrics["total_output_tokens"] is None
    assert metrics["calls_with_usage"] == 0
    for key in (
        "total_tokens",
        "cost_usd",
        "avg_cost_usd_per_task",
        "total_latency_seconds",
        "avg_latency_seconds_per_task",
    ):
        assert summary[key] is None
    assert summary["tasks_with_cost"] == summary["tasks_with_latency"] == 0
    assert json.dumps(summary, allow_nan=False)


def test_empty_episode_and_summary_are_unmeasured_not_free_or_successful():
    metrics = episode_metrics([])
    assert metrics["model_calls"] == metrics["calls_with_usage"] == metrics["total_tool_calls"] == 0
    for key in (
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "peak_input_tokens",
        "cost_usd",
        "latency_seconds",
    ):
        assert metrics[key] is None
    summary = summarize_results([])
    assert summary["tasks"] == summary["completed"] == summary["graded_tasks"] == 0
    for key in (
        "success_rate",
        "graded_success_rate",
        "total_tokens",
        "peak_input_tokens",
        "avg_peak_input_tokens",
        "cost_usd",
        "avg_latency_seconds_per_task",
    ):
        assert summary[key] is None
    assert summary["tool_call_counts"] == {}


def test_unavailable_cost_is_not_reported_as_a_free_run():
    from pocketcontextagent.agents.default import DefaultAgent
    from pocketcontextagent.environments.local import LocalEnvironment
    from pocketcontextagent.models.openrouter_model import OpenRouterModel
    from pocketcontextagent.models.test_models import DeterministicModel, make_output

    model = OpenRouterModel(model_name="example/model", cost_tracking="ignore_errors")
    assert model._calculate_cost({"usage": {}}) == {"cost": 0.0, "cost_available": False}
    assert model._calculate_cost({"usage": {"cost": 0.0}}) == {"cost": 0.0, "cost_available": True}
    output = make_output("Finish", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}], cost=0)
    output["extra"]["cost_available"] = False
    agent = DefaultAgent(
        DeterministicModel(outputs=[output]),
        LocalEnvironment(),
        system_template="Research",
        instance_template="{{task}}",
    )
    agent.run("Question")
    stats = agent.serialize()["info"]["model_stats"]
    assert stats["instance_cost"] == 0 and stats["cost_available"] is False
    metrics = episode_metrics(agent.messages, model_stats=stats)
    assert metrics["cost_usd"] is None
    summary = summarize_results([{"metrics": metrics, "model_stats": stats}])
    assert summary["cost_usd"] is None and summary["avg_cost_usd_per_task"] is None
    assert summary["tasks_with_cost"] == 0
