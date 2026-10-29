import json
import subprocess
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


def test_submission_snapshots_patch_not_agent_prose(repository):
    env = environment(repository)
    env.execute({"command": "printf 'value = 9\\n' > code.py"})
    assert env.execute({"command": "echo 'I fixed it.'"})["output"] == "I fixed it.\n"
    assert env.execute({"command": f"echo {SUBMIT_MARKER}; exit 1"})["returncode"] == 1
    assert env.execute({"command": f"echo not_first; echo {SUBMIT_MARKER}"})["returncode"] == 0
    with pytest.raises(Submitted) as submitted:
        env.execute({"command": f"echo {SUBMIT_MARKER}; echo 'this is prose, not a patch'"})
    message = submitted.value.messages[0]
    assert message["extra"]["exit_status"] == "Submitted"
    assert message["extra"]["submission"] == env.capture_patch() == message["content"]
    assert "+value = 9" in message["content"] and "prose" not in message["content"]
    assert "resolved" not in message["extra"]


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
