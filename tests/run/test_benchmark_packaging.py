import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from click import unstyle

from benchmarks.browsecomp_plus import implementation_hash
from benchmarks.browsecomp_plus_daytona import build_wheel


def test_installed_wheel_contains_root_benchmarks_and_preserves_cli_and_hash(tmp_path: Path):
    destination = tmp_path / "wheel"
    destination.mkdir()
    target = tmp_path / "installed"
    with zipfile.ZipFile(build_wheel(Path(__file__).resolve().parents[2], destination)) as wheel:
        assert {
            "benchmarks/benchmark_controller.py",
            "benchmarks/swebench.py",
            "benchmarks/utils/controller_job.py",
            "pocketcontextagent/agents/default.py",
            "pocketcontextagent/config/benchmarks/swebench_verified_context.yaml",
        } <= set(wheel.namelist())
        assert not any(name.startswith("pocketcontextagent/run/benchmarks/") for name in wheel.namelist())
        wheel.extractall(target)
    environment = os.environ | {
        "PYTHONPATH": str(target),
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,benchmarks; from benchmarks.browsecomp_plus import implementation_hash; "
            "from pocketcontextagent.run.benchmarks.browsecomp_plus import implementation_hash as legacy; "
            "assert legacy() == implementation_hash(); "
            "print(json.dumps({'file':benchmarks.__file__,'hash':implementation_hash()}))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(probe.stdout)
    assert Path(report["file"]).is_relative_to(target)
    assert report["hash"] == implementation_hash()
    for module, arguments in (
        ("benchmarks.browsecomp_plus", ["--help"]),
        ("pocketcontextagent.run.utilities.mini_extra", ["benchmark-controller", "--help"]),
        ("pocketcontextagent.run.benchmarks.browsecomp_plus", ["--help"]),
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Usage:" in unstyle(result.stdout)
