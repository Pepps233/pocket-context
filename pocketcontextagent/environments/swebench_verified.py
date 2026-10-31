"""Bash-only coding in a caller-owned, fresh SWE-bench instance sandbox.

The agent/model runs outside the sandbox. The runner creates one Daytona sandbox
per task from the official instance image, passes a process.exec adapter here,
then destroys it. Never inject model/Daytona credentials or grading data there.
No Docker daemon or nested Docker is needed for coding; official grading is a
separate trusted process. See run.benchmarks.utils.swebench_verified for evidence.

Call prepare(fresh_isolated=True) BEFORE agent.run(), and capture_attempt() in the
runner's finally block (also for limits/errors). There is deliberately no reset()
that an agent can invoke implicitly. Submitted means only that coding stopped,
not that the official tests passed.
"""

import re
from dataclasses import asdict, dataclass
from shlex import quote
from typing import Protocol

from pocketcontextagent.environments.local import _run
from pocketcontextagent.exceptions import Submitted

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class CommandBackend(Protocol):
    """Return complete, untruncated output; raise on transport/timeouts.

    A Daytona adapter calls sandbox.process.exec(command, cwd=cwd, timeout=timeout)
    and maps result.result/exit_code to output/returncode. It must preserve output
    exactly (including trailing newlines) and must not forward host credentials.
    """

    def __call__(self, command: str, *, cwd: str, timeout: int) -> dict: ...


class ExecutingBackend(Protocol):
    def execute(self, command: str, *, cwd: str, timeout: int) -> dict: ...


def subprocess_backend(command: str, *, cwd: str, timeout: int) -> dict:
    """Actual local subprocess backend for explicitly owned disposable test repos.

    This is NOT an isolation boundary. Production uses the sandbox adapter. The
    small explicit environment, unlike os.environ, cannot inherit API keys.
    """
    result = _run(
        command,
        cwd,
        {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "GIT_CONFIG_GLOBAL": "/dev/null"},
        timeout,
    )
    return {"output": result.stdout, "returncode": result.returncode}


@dataclass
class SWEBenchVerifiedConfig:
    research_tools: str = "bash"
    context_tools: str = "pocket_context"
    cwd: str = "/testbed"
    timeout: int = 120
    base_commit: str | None = None

    def __post_init__(self) -> None:
        if self.research_tools != "bash":
            raise ValueError("SWE-bench Verified requires research_tools='bash'")
        if self.context_tools not in ("pocket_context", "none", "acm"):
            raise ValueError("context_tools must be pocket_context, none, or acm")
        if not self.cwd.startswith("/") or self.timeout <= 0:
            raise ValueError("cwd must be absolute and timeout must be positive")
        if self.base_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", self.base_commit):
            raise ValueError("base_commit must be a full lowercase Git SHA")


class SWEBenchVerifiedEnvironment:
    def __init__(self, command_backend: CommandBackend | ExecutingBackend, **kwargs):
        self.config = SWEBenchVerifiedConfig(**kwargs)
        self.command_backend = command_backend if callable(command_backend) else command_backend.execute
        self.base_commit: str | None = None

    def _run(self, command: str, *, cwd: str = "", timeout: int | None = None) -> dict:
        result = self.command_backend(
            f"bash -c {quote(command)}", cwd=cwd or self.config.cwd, timeout=timeout or self.config.timeout
        )
        if not isinstance(result.get("output"), str) or type(result.get("returncode")) is not int:
            raise TypeError("Command backend must return string output and integer returncode")
        return {"output": result["output"], "returncode": result["returncode"], "exception_info": ""}

    def _checked(self, command: str) -> str:
        result = self._run(command)
        if result["returncode"] != 0:
            raise RuntimeError(f"SWE-bench repository command failed: {result['output']}")
        return result["output"]

    def prepare(self, *, fresh_isolated: bool = False) -> None:
        """Reset tracked files ONLY with the owner's explicit fresh-repo assertion.

        cwd must be the repository root. Verified build/lib copies of tracked
        source are preserved and ignored; other untracked files are rejected.
        No git clean is ever used. This assertion is
        an ownership contract, not a way to discover whether a repo is disposable.
        """
        if fresh_isolated is not True or self.base_commit is not None:
            raise ValueError("prepare requires a fresh isolated repository and may only run once")
        self._checked('test "$(pwd -P)" = "$(git rev-parse --show-toplevel)"')
        base = self._checked(f"git rev-parse --verify {quote((self.config.base_commit or 'HEAD') + '^{commit}')}")
        base = base.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", base):
            raise ValueError("Repository did not resolve a full base commit")
        untracked = self._checked("git ls-files --others --exclude-standard -z").split("\0")[:-1]
        if untracked:
            copies = self._run(
                f"for file in {' '.join(quote(path) for path in untracked)}; do\n"
                'case "$file" in build/lib/*) ;; *) exit 1;; esac\n'
                'test -f "$file" && ! test -L "$file" || exit 1\n'
                f'source=$(git rev-parse --verify "{base}:${{file#build/lib/}}") || exit 1\n'
                'test "$source" = "$(git hash-object --no-filters -- "$file")" || exit 1\n'
                "done"
            )
            if copies["returncode"] != 0:
                raise ValueError("Fresh repository contains untracked files; refusing to delete or submit them")
            self._checked('printf "\\n/build/lib/\\n" >> "$(git rev-parse --git-path info/exclude)"')
        self._checked(f"git reset --hard {base}")
        self.base_commit = base

    def capture_patch(self) -> str:
        """Snapshot the actual tree against BASE, even after agent commits.

        Includes staged/unstaged changes, deletions, binary files and nonignored
        new files (plus explicitly git-added ignored files). Ignored build output
        is omitted. A temporary COPY of the index leaves staging untouched. Final
        prose and an agent-provided diff are never treated as a prediction.
        """
        if self.base_commit is None:
            raise RuntimeError("Call prepare before executing or capturing a patch")
        return self._checked(
            "set -eu\n"
            "index=$(mktemp)\n"
            'trap \'rm -f "$index" "$index.lock"\' EXIT\n'
            'cp -p "$(git rev-parse --git-path index)" "$index"\n'
            'export GIT_INDEX_FILE="$index"\n'
            "git add -A -- .\n"
            f"git -c core.quotePath=true diff --cached --binary --full-index --no-ext-diff --no-textconv "
            f"--no-color --src-prefix=a/ --dst-prefix=b/ {self.base_commit} -- ."
        )

    def finish_patch(self) -> str:
        return self.capture_patch()

    def get_patch(self) -> str:
        return self.capture_patch()

    def capture_attempt(self, instance_id: str, model_name_or_path: str, exit_status: str) -> dict:
        """Always call after an attempt, including budget/error exits; never grade here."""
        return {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": self.capture_patch(),
            "exit_status": exit_status,
            "submitted": exit_status == "Submitted",
        }

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        if self.base_commit is None:
            raise RuntimeError("Call prepare before executing or capturing a patch")
        result = self._run(action["command"], cwd=cwd, timeout=timeout)
        lines = result["output"].strip().splitlines()
        if result["returncode"] == 0 and lines and SUBMIT_MARKER in (lines[0].strip(), lines[-1].strip()):
            patch = self.capture_patch()
            raise Submitted(
                {"role": "exit", "content": patch, "extra": {"exit_status": "Submitted", "submission": patch}}
            )
        return result

    def get_template_vars(self, **kwargs) -> dict:
        # Only public configuration; never merge os.environ or backend attributes.
        return asdict(self.config) | {"base_commit": self.base_commit or self.config.base_commit}

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.get_template_vars(),
                    "environment_type": f"{type(self).__module__}.{type(self).__name__}",
                }
            }
        }
