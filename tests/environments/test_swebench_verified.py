import json
import os
import subprocess
import time
from pathlib import Path
from shlex import quote

import pytest

from pocketcontextagent.agents.default import DefaultAgent
from pocketcontextagent.environments.swebench_verified import (
    SUBMIT_MARKER,
    SWEBenchVerifiedEnvironment,
    subprocess_backend,
)
from pocketcontextagent.exceptions import Submitted
from pocketcontextagent.models.test_models import DeterministicModel, make_output


def shell(command: str, cwd: Path) -> str:
    result = subprocess_backend(command, cwd=str(cwd), timeout=10)
    assert result["returncode"] == 0, result["output"]
    return result["output"]


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    shell("git init -q && git config user.email test@example.com && git config user.name Test", repo)
    (repo / "code.py").write_text("value = 1\n")
    (repo / "delete.py").write_text("old = True\n")
    (repo / ".gitignore").write_text("*.cache\n")
    shell("git add . && git commit -qm base", repo)
    return repo


def environment(repo: Path) -> SWEBenchVerifiedEnvironment:
    env = SWEBenchVerifiedEnvironment(
        subprocess_backend, cwd=str(repo), base_commit=shell("git rev-parse HEAD", repo).strip()
    )
    env.prepare(fresh_isolated=True)
    return env


def test_patch_is_complete_worktree_against_base_without_mutating_index(repository, tmp_path):
    env = environment(repository)
    shell("git clone -q . ../pristine", repository)
    (repository / "code.py").write_text("value = 2\n")
    shell("git add code.py && git commit -qm agent_commit", repository)
    (repository / "code.py").write_text("value = 3\n")
    shell("git add code.py", repository)
    (repository / "code.py").write_text("value = 4\n")
    (repository / "delete.py").unlink()
    (repository / "new file.py").write_text("new = True\n")
    (repository / "binary.dat").write_bytes(bytes(range(256)))
    (repository / "build.cache").write_text("ignored output\n")
    (repository / "intended.cache").write_text("explicitly intended\n")
    shell("git add -f intended.cache", repository)
    staging = shell("git diff --cached --binary", repository)
    patch = env.get_patch()
    assert "+value = 4" in patch and "+value = 2" not in patch
    assert "new file.py" in patch and "GIT binary patch" in patch and "deleted file" in patch
    assert "intended.cache" in patch and "build.cache" not in patch
    assert env.finish_patch() == patch
    assert shell("git diff --cached --binary", repository) == staging
    patch_path = tmp_path / "actual.patch"
    patch_path.write_text(patch)
    shell(f"git apply --check {quote(str(patch_path))} && git apply {quote(str(patch_path))}", tmp_path / "pristine")
    assert (tmp_path / "pristine" / "code.py").read_text() == "value = 4\n"
    assert (tmp_path / "pristine" / "new file.py").read_text() == "new = True\n"
    assert (tmp_path / "pristine" / "binary.dat").read_bytes() == bytes(range(256))
    assert not (tmp_path / "pristine" / "delete.py").exists()


def test_patch_copy_preserves_racy_index_detection_and_staged_ignored_files(repository):
    env = environment(repository)
    shell("git config core.trustctime false", repository)
    timestamp = int(time.time()) - 120
    os.utime(repository / "code.py", (timestamp, timestamp))
    shell("git add code.py", repository)
    (repository / "explicit.cache").write_text("intentionally staged\n")
    (repository / "build.cache").write_text("ignored build output\n")
    shell("git add -f explicit.cache", repository)
    index = repository / ".git/index"
    os.utime(index, (timestamp, timestamp))
    original_index, original_mtime = index.read_bytes(), index.stat().st_mtime_ns
    (repository / "code.py").write_text("value = 8\n")
    os.utime(repository / "code.py", (timestamp, timestamp))
    patch = env.capture_patch()
    assert "+value = 8" in patch and "explicit.cache" in patch and "build.cache" not in patch
    assert env.capture_patch() == patch
    assert index.read_bytes() == original_index and index.stat().st_mtime_ns == original_mtime
    assert (repository / "code.py").read_text() == "value = 8\n"


def test_explicit_prepare_only_and_no_clean_outside_owned_repo(repository):
    env = SWEBenchVerifiedEnvironment(subprocess_backend, cwd=str(repository))
    (repository / "code.py").write_text("dirty\n")
    with pytest.raises(ValueError, match="fresh isolated"):
        env.prepare()
    with pytest.raises(RuntimeError, match="prepare"):
        env.execute({"command": "true"})
    assert (repository / "code.py").read_text() == "dirty\n"
    (repository / "keep.txt").write_text("must not be deleted")
    with pytest.raises(ValueError, match="untracked"):
        env.prepare(fresh_isolated=True)
    assert (repository / "keep.txt").read_text() == "must not be deleted"
    (repository / "keep.txt").unlink()
    (repository / "keep.cache").write_text("ignored file must survive")
    env.prepare(fresh_isolated=True)
    assert (repository / "code.py").read_text() == "value = 1\n"
    assert (repository / "keep.cache").read_text() == "ignored file must survive"
    with pytest.raises(ValueError, match="only run once"):
        env.prepare(fresh_isolated=True)
    child = repository / "child"
    child.mkdir()
    with pytest.raises(RuntimeError, match="repository command failed"):
        SWEBenchVerifiedEnvironment(subprocess_backend, cwd=str(child)).prepare(fresh_isolated=True)


def test_prepare_preserves_verified_build_copies_without_polluting_patch(repository):
    (repository / "quoted ' source.py").write_text("original = True\n")
    shell("git add . && git commit -qm source", repository)
    build = repository / "build/lib"
    build.mkdir(parents=True)
    for name in ("code.py", "quoted ' source.py"):
        (build / name).write_bytes((repository / name).read_bytes())
    base = shell("git rev-parse HEAD", repository).strip()
    (repository / "code.py").write_text("value = 999\n")
    env = SWEBenchVerifiedEnvironment(subprocess_backend, cwd=str(repository), base_commit=base)
    env.prepare(fresh_isolated=True)
    assert env.capture_patch() == ""
    assert (build / "code.py").read_text() == (repository / "code.py").read_text() == "value = 1\n"
    (repository / "code.py").write_text("value = 2\n")
    (build / "code.py").write_text("generated = True\n")
    (repository / "new.py").write_text("new = True\n")
    patch = env.capture_patch()
    assert "+value = 2" in patch and "new.py" in patch and "build/lib" not in patch
    assert "info/exclude" not in patch and ".gitignore" not in patch


@pytest.mark.parametrize(
    ("unexpected",),  # noqa: PT006
    [("changed_copy",), ("unknown_copy",), ("outside_build",), ("symlink",)],
)
def test_prepare_rejects_unverified_build_files_without_mutation(repository, unexpected):
    build = repository / "build/lib"
    build.mkdir(parents=True)
    (build / "code.py").write_bytes((repository / "code.py").read_bytes())
    if unexpected == "changed_copy":
        (build / "code.py").write_text("different = True\n")
    elif unexpected == "unknown_copy":
        (build / "unknown.py").write_text("value = 1\n")
    elif unexpected == "outside_build":
        (repository / "keep.txt").write_text("keep\n")
    else:
        (build / "delete.py").symlink_to(repository / "delete.py")
    exclude = (repository / ".git/info/exclude").read_bytes()
    before = shell("git status --porcelain --untracked-files=all", repository)
    with pytest.raises(ValueError, match="untracked"):
        environment(repository)
    assert shell("git status --porcelain --untracked-files=all", repository) == before
    assert (repository / ".git/info/exclude").read_bytes() == exclude
    assert (build / "code.py").exists()


@pytest.mark.parametrize(
    ("command",),  # noqa: PT006
    [
        (f"echo {SUBMIT_MARKER}",),
        (f"echo '{SUBMIT_MARKER}'",),
        (f"printf 'checks passed\\n{SUBMIT_MARKER}'",),
        (f"echo {SUBMIT_MARKER}; echo 'this is prose, not a patch'",),
        (f"printf 'checks passed\\n'; git diff --check && echo {SUBMIT_MARKER}",),
        (f"printf '\\nchecks passed\\r\\n  {SUBMIT_MARKER}  \\r\\n\\n'",),
    ],
)
def test_submission_snapshots_patch_not_agent_prose(repository, command):
    env = environment(repository)
    env.execute({"command": "printf 'value = 9\\n' > code.py"})
    assert env.execute({"command": "echo 'I fixed it.'"})["output"] == "I fixed it.\n"
    assert env.execute({"command": f"echo {SUBMIT_MARKER}; exit 1"})["returncode"] == 1
    with pytest.raises(Submitted) as submitted:
        env.execute({"command": command})
    message = submitted.value.messages[0]
    assert message["extra"]["exit_status"] == "Submitted"
    assert message["extra"]["submission"] == env.capture_patch() == message["content"]
    assert "+value = 9" in message["content"] and "prose" not in message["content"]
    assert "resolved" not in message["extra"]


@pytest.mark.parametrize(
    ("command", "returncode"),
    [
        (f"echo {SUBMIT_MARKER}; exit 1", 1),
        (f"echo checks; echo {SUBMIT_MARKER}; exit 7", 7),
        (f"echo 'example: {SUBMIT_MARKER}'", 0),
        (f"echo '{SUBMIT_MARKER} is the marker'", 0),
        (f"printf 'before\\n{SUBMIT_MARKER}\\nafter\\n'", 0),
        (f"printf 'before\\n\"{SUBMIT_MARKER}\"\\n'", 0),
        ("true", 0),
    ],
)
def test_submission_rejects_failed_commands_embedded_and_interior_markers(repository, command, returncode):
    env = environment(repository)
    (repository / "code.py").write_text("value = 8\n")
    assert env.execute({"command": command})["returncode"] == returncode
    assert "+value = 8" in env.capture_patch()
    assert (repository / "code.py").read_text() == "value = 8\n"


def test_budget_exit_and_agent_run_do_not_reset_worktree(repository, tmp_path):
    env = environment(repository)
    (repository / "code.py").write_text("before_agent = True\n")
    agent = DefaultAgent(
        DeterministicModel(outputs=[make_output("edit", [{"command": "printf 'budget edit\\n' > new.py"}])]),
        env,
        system_template="Use bash; submit explicitly.",
        instance_template="{{task}}",
        step_limit=1,
        cost_limit=0,
        output_path=tmp_path / "trajectory.json",
    )
    info = agent.run("Fix the public issue")
    attempt = env.capture_attempt("owner__project-1", "offline-test", info["exit_status"])
    assert info["exit_status"] == "LimitsExceeded" and info["submission"] == ""
    assert attempt["submitted"] is False and "resolved" not in attempt
    assert "+before_agent = True" in attempt["model_patch"] and "+budget edit" in attempt["model_patch"]


def test_timeout_propagates_but_patch_remains_capturable(repository):
    env = environment(repository)
    with pytest.raises(subprocess.TimeoutExpired):
        env.execute({"command": "printf 'saved = True\\n' > code.py; sleep 5"}, timeout=1)
    assert "+saved = True" in env.capture_attempt("owner__project-1", "offline", "TimeoutExpired")["model_patch"]


def test_backend_object_public_configuration_and_no_inherited_secrets(repository):
    class LocalCommands:
        api_key = "must-not-serialize"

        def execute(self, command: str, *, cwd: str, timeout: int) -> dict:
            return subprocess_backend(command, cwd=cwd, timeout=timeout)

    env = SWEBenchVerifiedEnvironment(LocalCommands(), cwd=str(repository), context_tools="none")
    env.prepare(fresh_isolated=True)
    assert set(env.get_template_vars(secret="not public")) == {
        "research_tools",
        "context_tools",
        "cwd",
        "timeout",
        "base_commit",
    }
    assert "must-not-serialize" not in json.dumps(env.serialize())
    assert "not public" not in json.dumps(env.get_template_vars(secret="not public"))
    assert "API_KEY" not in env.execute({"command": "env"})["output"]
    assert env.execute({"command": "values=(native bash); printf '%s' \"${values[1]}\""})["output"] == "bash"
    assert env.execute({"command": "printf 'ok\\n' >&2; exit 7"}) == {
        "output": "ok\n",
        "returncode": 7,
        "exception_info": "",
    }
