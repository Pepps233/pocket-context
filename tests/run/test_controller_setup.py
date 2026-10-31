import io
import json
import stat
import tarfile
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.utils.controller_setup import (
    ControllerConfig,
    extract_artifacts,
    parse_remote_json,
    provider_endpoint,
    sandbox_params,
    write_private_json,
)


def test_remote_json_preserves_payload_amid_combined_runtime_warnings() -> None:
    marker = "PCA_TEST_123:"
    payload = {"ready": True, "text": "line one\nPCA_TEST_123: not another record", "nested": [1, None]}
    output = (
        "[transformers] No model runtime; tokenizers are available.\n"
        + marker
        + json.dumps(payload)
        + "\nJava warning after result\n"
    )
    assert parse_remote_json(output, marker) == payload
    assert parse_remote_json("notice\nPCA_CORPUS_RESULT:null\n", "PCA_CORPUS_RESULT:") is None


@pytest.mark.parametrize(
    ("output", "marker"),
    [("{}", "PCA:"), ("PCA:{}\nPCA:{}\n", "PCA:"), ("PCA:invalid", "PCA:"), ("{}", ""), ("{}", "PCA:\n")],
)
def test_remote_json_rejects_missing_ambiguous_or_invalid_frames(output: str, marker: str) -> None:
    with pytest.raises(ValueError):
        parse_remote_json(output, marker)


def test_controller_defaults_and_real_daytona_params() -> None:
    daytona = pytest.importorskip("daytona")
    assert asdict(ControllerConfig()) == {"cpu": 2, "memory": 8, "disk": 10, "ttl_minutes": 0}
    for config in (ControllerConfig(), ControllerConfig(cpu=4, memory=16, disk=20, ttl_minutes=90)):
        params = sandbox_params("controller-snapshot", "controller-name", config)
        assert isinstance(params, daytona.CreateSandboxFromSnapshotParams)
        assert params.model_dump(exclude_none=True) == {
            "snapshot": "controller-snapshot",
            "name": "controller-name",
            "os_user": "root",
            "public": False,
            "network_block_all": False,
            "ephemeral": False,
            "auto_stop_interval": 0,
            "auto_pause_interval": 0,
            "auto_delete_interval": -1,
            "ttl_minutes": config.ttl_minutes,
            "labels": {"project": "pocket-context", "role": "trusted-controller"},
            "env_vars": {
                "MSWEA_SILENT_STARTUP": "1",
                "MSWEA_GLOBAL_CONFIG_DIR": "/workspace/pocket-controller/config",
                "PYTHONUNBUFFERED": "1",
            },
        }
        assert not {"resources", "cpu", "memory", "disk", "secrets"} & params.model_fields_set


@pytest.mark.parametrize(("field",), [(field,) for field in ("cpu", "memory", "disk")])  # noqa: PT006
@pytest.mark.parametrize(("value",), [(0,), (-1,), (True,), (False,), (2.0,), ("2",), (None,)])  # noqa: PT006
def test_invalid_controller_resources(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        ControllerConfig(**{field: value})


@pytest.mark.parametrize(("value",), [(-1,), (True,), (False,), (1.0,), ("0",), (None,)])  # noqa: PT006
def test_invalid_controller_ttl(value: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        ControllerConfig(ttl_minutes=value)


def test_private_json_creates_private_parents_and_atomically_replaces(tmp_path: Path) -> None:
    original_parent_mode = stat.S_IMODE(tmp_path.stat().st_mode)
    path = tmp_path / "private" / "nested" / "credentials.json"
    value = {"DEEPSEEK_API_KEY": "private-value", "unicode": "\u03bb", "nested": {"items": [1, None]}}
    write_private_json(path, value)
    assert json.loads(path.read_text()) == value
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(tmp_path.stat().st_mode) == original_parent_mode
    old = path.with_name("old.json")
    old.hardlink_to(path)
    path.chmod(0o644)
    write_private_json(path, {"DEEPSEEK_API_KEY": "replacement"})
    assert json.loads(path.read_text()) == {"DEEPSEEK_API_KEY": "replacement"}
    assert json.loads(old.read_text()) == value
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert {item.name for item in path.parent.iterdir()} == {"credentials.json", "old.json"}
    assert not list(tmp_path.rglob(".env"))


def test_private_json_replaces_symlink_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "public.json"
    target.write_text('{"public": true}')
    target.chmod(0o644)
    private = tmp_path / "private.json"
    private.symlink_to(target)
    write_private_json(private, {"API_KEY": "secret"})
    assert not private.is_symlink()
    assert json.loads(private.read_text()) == {"API_KEY": "secret"}
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert target.read_text() == '{"public": true}'
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_private_json_serialization_failure_leaves_original_and_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_private_json(path, {"key": "original"})
    with pytest.raises(TypeError):
        write_private_json(path, {"bad": object()})
    assert json.loads(path.read_text()) == {"key": "original"}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    ("config", "endpoint"),
    [
        ({"model_name": "deepseek/deepseek-flash"}, "https://api.deepseek.com/models"),
        ({"model_name": "meta/muse-spark"}, "https://api.meta.ai/v1/models"),
        ({"model_name": "openrouter/vendor/model"}, "https://openrouter.ai/api/v1/auth/key"),
        ({"model_name": "vendor/model", "model_class": "openrouter"}, "https://openrouter.ai/api/v1/auth/key"),
        ({"model_class": "openrouter_textbased"}, "https://openrouter.ai/api/v1/auth/key"),
        ({"model_name": "meta/model", "model_class": "openrouter_response"}, "https://openrouter.ai/api/v1/auth/key"),
    ],
)
def test_provider_endpoints(config: dict, endpoint: str) -> None:
    assert provider_endpoint(config) == endpoint


@pytest.mark.parametrize(
    ("provider", "base", "suffix"),
    [
        ("deepseek", "https://api.deepseek.com", "/models"),
        ("meta", "https://api.meta.ai/v1", "/models"),
        ("openrouter", "https://openrouter.ai/api/v1", "/auth/key"),
    ],
)
@pytest.mark.parametrize(("trailing_slash",), [("",), ("/",)])  # noqa: PT006
@pytest.mark.parametrize(("nested",), [(True,), (False,)])  # noqa: PT006
def test_official_provider_base_validation(
    provider: str, base: str, suffix: str, trailing_slash: str, nested: bool
) -> None:
    settings = {"api_base": base + trailing_slash}
    assert (
        provider_endpoint({"model_name": provider + "/model", **({"model_kwargs": settings} if nested else settings)})
        == base + suffix
    )


@pytest.mark.parametrize(
    ("base",),  # noqa: PT006
    [
        ("http://api.meta.ai/v1",),
        ("https://api.meta.ai:443/v1",),
        ("https://user@api.meta.ai/v1",),
        ("https://user:password@api.meta.ai/v1",),
        ("https://api.meta.ai@evil.example/v1",),
        ("https://api.meta.ai.evil.example/v1",),
        ("https://evil.example/v1",),
        ("https://api.meta.ai/v1/other",),
        ("https://api.meta.ai/v1//",),
        ("https://api.meta.ai/v1/../v1",),
        ("https://api.meta.ai/v1?key=secret",),
        ("https://api.meta.ai/v1?",),
        ("https://api.meta.ai/v1#fragment",),
        ("https://api.meta.ai/v1#",),
        (" https://api.meta.ai/v1",),
        ("https://api.\nmeta.ai/v1",),
        ("https://api.meta.ai\\@evil.example/v1",),
        ("",),
        (123,),
    ],
)
@pytest.mark.parametrize(("nested",), [(True,), (False,)])  # noqa: PT006
def test_rejects_custom_or_unsafe_provider_base(base: object, nested: bool) -> None:
    settings = {"api_base": base}
    with pytest.raises(ValueError, match="api_base"):
        provider_endpoint({"model_name": "meta/model", **({"model_kwargs": settings} if nested else settings)})


@pytest.mark.parametrize(("config",), [({},), ({"model_name": "custom/model"},), ({"model_name": "gpt-5"},)])  # noqa: PT006
def test_unknown_provider_is_rejected(config: dict) -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        provider_endpoint(config)


def test_both_api_base_locations_are_validated() -> None:
    with pytest.raises(ValueError, match="api_base"):
        provider_endpoint(
            {
                "model_name": "deepseek/model",
                "api_base": "https://evil.example",
                "model_kwargs": {"api_base": "https://api.deepseek.com"},
            }
        )


def write_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for member, contents in members:
            member.size = len(contents)
            bundle.addfile(member, io.BytesIO(contents))


def member(name: str, kind: bytes = tarfile.REGTYPE, link: str = "") -> tarfile.TarInfo:
    entry = tarfile.TarInfo(name)
    entry.type = kind
    entry.linkname = link
    entry.mode = 0o7777
    return entry


@pytest.mark.parametrize(("existing",), [(True,), (False,)])  # noqa: PT006
def test_extracts_real_archive_preserving_bytes(tmp_path: Path, existing: bool) -> None:
    archive, destination = tmp_path / "outputs.tar.gz", tmp_path / "artifacts"
    if existing:
        destination.mkdir()
    outputs = {
        "status.json": b'{"state":"completed","exit_code":0}',
        "run/trajectory.json": b'{"messages":[]}',
        "run/binary.bin": bytes(range(256)) + b"\0\xff\r\n",
        "empty-file": b"",
    }
    write_archive(
        archive,
        [(member(".", tarfile.DIRTYPE), b"")]
        + [(member("./" + name), contents) for name, contents in outputs.items()]
        + [(member("run", tarfile.DIRTYPE), b""), (member("empty-dir", tarfile.DIRTYPE), b"")],
    )
    extract_artifacts(archive, destination)
    assert {
        str(path.relative_to(destination)): path.read_bytes() for path in destination.rglob("*") if path.is_file()
    } == outputs
    assert (destination / "empty-dir").is_dir()
    assert not any(
        path.is_symlink() or path.stat().st_mode & (stat.S_ISUID | stat.S_ISGID) for path in destination.rglob("*")
    )


@pytest.mark.parametrize(
    ("name", "kind", "link"),
    [
        ("../escape", tarfile.REGTYPE, ""),
        ("nested/../../escape", tarfile.REGTYPE, ""),
        ("nested/../escape", tarfile.REGTYPE, ""),
        ("/absolute", tarfile.REGTYPE, ""),
        ("C:/escape", tarfile.REGTYPE, ""),
        ("..\\escape", tarfile.REGTYPE, ""),
        ("linked", tarfile.SYMTYPE, "../outside"),
        ("linked", tarfile.SYMTYPE, "/tmp/outside"),
        ("linked", tarfile.SYMTYPE, "valid.json"),
        ("linked", tarfile.LNKTYPE, "../outside"),
        ("linked", tarfile.LNKTYPE, "valid.json"),
        ("device", tarfile.CHRTYPE, ""),
        ("device", tarfile.BLKTYPE, ""),
        ("pipe", tarfile.FIFOTYPE, ""),
        ("unknown", b"Z", ""),
        (".", tarfile.REGTYPE, ""),
    ],
)
@pytest.mark.parametrize(("existing",), [(True,), (False,)])  # noqa: PT006
def test_rejects_unsafe_members_before_writing_anything(
    tmp_path: Path, name: str, kind: bytes, link: str, existing: bool
) -> None:
    archive, destination = tmp_path / "outputs.tar.gz", tmp_path / "artifacts"
    if existing:
        destination.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    write_archive(archive, [(member("valid.json"), b"valid"), (member(name, kind, link), b"")])
    with pytest.raises(ValueError, match="Unsafe"):
        extract_artifacts(archive, destination)
    assert outside.read_bytes() == b"untouched"
    assert list(destination.iterdir()) == [] if existing else not destination.exists()
    assert {path.name for path in tmp_path.iterdir()} == (
        {"outputs.tar.gz", "outside", "artifacts"} if existing else {"outputs.tar.gz", "outside"}
    )


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([("same", tarfile.REGTYPE), ("same", tarfile.REGTYPE)], "Duplicate"),
        ([("same", tarfile.REGTYPE), ("./same", tarfile.REGTYPE)], "Duplicate"),
        ([("a/b", tarfile.REGTYPE), ("a//b", tarfile.REGTYPE)], "Duplicate"),
        ([("dir", tarfile.DIRTYPE), ("dir/", tarfile.DIRTYPE)], "Duplicate"),
        ([("dir", tarfile.REGTYPE), ("dir", tarfile.DIRTYPE)], "Duplicate"),
        ([("parent", tarfile.REGTYPE), ("parent/child", tarfile.REGTYPE)], "conflict"),
        ([("parent/child", tarfile.REGTYPE), ("parent", tarfile.REGTYPE)], "conflict"),
        ([("parent/child", tarfile.DIRTYPE), ("parent", tarfile.REGTYPE)], "conflict"),
    ],
)
def test_rejects_duplicates_and_path_conflicts_before_writing(
    tmp_path: Path, entries: list[tuple[str, bytes]], message: str
) -> None:
    archive, destination = tmp_path / "outputs.tar.gz", tmp_path / "artifacts"
    write_archive(archive, [(member(name, kind), b"") for name, kind in entries])
    with pytest.raises(ValueError, match=message):
        extract_artifacts(archive, destination)
    assert not destination.exists()


def test_link_escape_followed_by_child_is_rejected_before_writing(tmp_path: Path) -> None:
    archive, destination = tmp_path / "outputs.tar.gz", tmp_path / "artifacts"
    outside = tmp_path / "outside"
    outside.mkdir()
    write_archive(
        archive,
        [(member("escape", tarfile.SYMTYPE, str(outside)), b""), (member("escape/stolen.json"), b"secret")],
    )
    with pytest.raises(ValueError, match="Unsafe"):
        extract_artifacts(archive, destination)
    assert list(outside.iterdir()) == []
    assert not destination.exists()


@pytest.mark.parametrize(("kind",), [("nonempty",), ("file",), ("symlink",), ("broken-symlink",)])  # noqa: PT006
def test_rejects_nonfresh_destination_without_modifying_it(tmp_path: Path, kind: str) -> None:
    archive, destination = tmp_path / "outputs.tar.gz", tmp_path / "artifacts"
    target = tmp_path / "target"
    target.mkdir()
    if kind == "nonempty":
        destination.mkdir()
        (destination / "original").write_bytes(b"preserved")
    elif kind == "file":
        destination.write_bytes(b"preserved")
    else:
        destination.symlink_to(target if kind == "symlink" else tmp_path / "missing", target_is_directory=True)
    write_archive(archive, [(member("new.json"), b"new")])
    with pytest.raises(ValueError, match="destination"):
        extract_artifacts(archive, destination)
    assert list(target.iterdir()) == []
    if kind == "nonempty":
        assert list(destination.iterdir()) == [destination / "original"]
        assert (destination / "original").read_bytes() == b"preserved"
    elif kind == "file":
        assert destination.read_bytes() == b"preserved"
    else:
        assert destination.is_symlink()
