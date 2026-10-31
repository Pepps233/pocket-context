import pytest

from pocketcontextagent.run.benchmarks.pocket_failed_retry import failed_tasks, launch, submit_command


def evidence() -> tuple[dict, dict, list[dict]]:
    ids = [f"task-{number}" for number in range(125)]
    return (
        {"selected_query_ids": ids},
        {
            "methods": [
                {
                    "method": "pocket_context",
                    "summary": {
                        "instances": [
                            {"instance_id": iid, "resolved": number > 1, "evaluated": number != 1}
                            for number, iid in enumerate(reversed(ids))
                        ]
                    },
                }
            ]
        },
        [{"instance_id": iid, "model_patch": "" if iid == "task-123" else "patch"} for iid in ids],
    )


def test_selection_uses_grades_in_cohort_order_and_allocates_every_task(tmp_path):
    assert failed_tasks(*evidence()) == ["task-123", "task-124"]
    argv = submit_command(tmp_path, failed_tasks(*evidence()))
    assert argv[argv.index("--workers-per-method") + 1] == "2"
    assert argv.count("--query-id") == 2
    assert argv[argv.index("--method") + 1] == "pocket_context"
    assert argv[argv.index("--limit") + 1] == "0"


@pytest.mark.parametrize(("damage",), [("duplicate",), ("missing_grade",), ("ungraded_patch",), ("unknown_outcome",)])  # noqa: PT006
def test_refuse_incomplete_or_ambiguous_grading(damage):
    index, grading, results = evidence()
    reports = grading["methods"][0]["summary"]["instances"]
    if damage == "duplicate":
        results[-1] = results[0]
    elif damage == "missing_grade":
        reports.pop()
    elif damage == "ungraded_patch":
        results[123]["model_patch"] = "ungraded patch"
    else:
        reports[0]["resolved"] = None
    with pytest.raises(ValueError):
        failed_tasks(index, grading, results)


def test_launch_refuses_changed_implementation_before_any_resource_action(tmp_path):
    (tmp_path / "plan.json").write_text('{"implementation_sha256":"stale"}')
    with pytest.raises(ValueError, match="Code changed"):
        launch(tmp_path)
    assert not (tmp_path / "launch-intent.json").exists()
    assert not (tmp_path / "controller.json").exists()
