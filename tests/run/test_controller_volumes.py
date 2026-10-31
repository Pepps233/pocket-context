from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.utils.controller_volumes import (
    STORE_ROOT,
    VOLUME_MOUNT,
    _verify_fuse_mountinfo,
    _volume_is_ready,
    archive_environment,
    validate_volume_name,
    verify_volume_mount,
    volume_mount,
)

if TYPE_CHECKING:
    from daytona.common.volume import Volume


@pytest.fixture
def record() -> dict:
    return {
        "id": "d8662458-761a-4ca0-ae66-4ed12990a306",
        "name": "controller-artifacts",
        "mount_path": VOLUME_MOUNT,
        "store_path": STORE_ROOT,
        "created": False,
    }


def sdk_volume(state: str) -> "Volume":
    pytest.importorskip("daytona")
    from daytona.common.volume import Volume
    from daytona_api_client import VolumeDto

    return Volume.from_dto(
        VolumeDto(
            id="d8662458-761a-4ca0-ae66-4ed12990a306",
            name="controller-artifacts",
            organization_id="private-organization",
            state=state,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            error_reason="private backend details",
            additional_properties={"credentials": "private-value"},
        )
    )


def test_real_sdk_volume_mount_and_creation_params(record: dict) -> None:
    daytona = pytest.importorskip("daytona")
    volume = sdk_volume("ready")
    record.update(id=volume.id, name=volume.name)
    mount = volume_mount(record)
    assert isinstance(mount, daytona.VolumeMount)
    assert mount.model_fields_set == {"volume_id", "mount_path"}
    assert mount.volume_id == volume.id
    assert mount.mount_path == "/mnt/pocket-context-artifacts"
    assert mount.subpath is None
    assert mount.additional_properties == {}
    assert mount.model_dump(exclude_unset=True, by_alias=True) == {
        "volumeId": volume.id,
        "mountPath": VOLUME_MOUNT,
    }
    params = daytona.CreateSandboxFromSnapshotParams(snapshot="controller-snapshot", volumes=[mount])
    assert params.volumes == [mount]
    assert "private" not in params.model_dump_json()
    assert "credentials" not in params.model_dump_json()


@pytest.mark.parametrize(
    ("state", "ready"),
    [("ready", True), ("creating", False), ("pending_create", False)],
)
def test_real_sdk_creation_states(state: str, ready: bool) -> None:
    assert _volume_is_ready(sdk_volume(state)) is ready


@pytest.mark.parametrize(
    ("state",),  # noqa: PT006
    [("error",), ("pending_delete",), ("deleting",), ("deleted",), ("unknown_default_open_api",), ("future-state",)],
)
def test_real_sdk_unusable_and_unknown_states_fail_closed(state: str) -> None:
    volume = sdk_volume(state)
    with pytest.raises(RuntimeError, match="cannot become ready") as error:
        _volume_is_ready(volume)
    assert volume.id in str(error.value)
    assert "private" not in str(error.value)
    if state == "future-state":
        assert volume.state.value == "unknown_default_open_api"


@pytest.mark.parametrize(
    ("name",),  # noqa: PT006
    [("a",), ("0",), ("controller-artifacts_01",), ("Controller-01",), ("a" * 64,)],
)
def test_safe_volume_names(name: str) -> None:
    assert validate_volume_name(name) is None


@pytest.mark.parametrize(
    ("name",),  # noqa: PT006
    [
        ("",),
        ("a" * 65,),
        ("../escape",),
        ("/absolute",),
        ("a/b",),
        ("a\\b",),
        ("a.b",),
        ("-leading",),
        ("_leading",),
        ("two words",),
        ("trailing ",),
        ("line\nbreak",),
        ("nul\0byte",),
        ("a;touch-pwned",),
        ("$(command)",),
        ("https://host",),
        ("café",),
        (None,),
        (True,),
        (123,),
    ],
)
def test_unsafe_volume_names(name: object) -> None:
    with pytest.raises(ValueError, match="volume name"):
        validate_volume_name(name)


def test_archive_environment_is_only_public_string_settings(record: dict) -> None:
    assert STORE_ROOT == "/mnt/pocket-context-artifacts/artifacts-v1"
    assert archive_environment(record, "job-01_a") == {
        "MSWEA_ARTIFACT_STORE_ROOT": STORE_ROOT,
        "MSWEA_ARTIFACT_JOB_ID": "job-01_a",
        "MSWEA_ARTIFACT_MIN_FREE_BYTES": "1073741824",
    }
    record["created"] = True
    assert archive_environment(record, "a" * 64, min_free_bytes=123)["MSWEA_ARTIFACT_MIN_FREE_BYTES"] == "123"
    assert archive_environment(record, "0", min_free_bytes=1)["MSWEA_ARTIFACT_MIN_FREE_BYTES"] == "1"
    assert all(isinstance(value, str) for value in archive_environment(record, "job").values())
    assert record["id"] not in str(archive_environment(record, "job"))
    assert record["name"] not in str(archive_environment(record, "job"))


@pytest.mark.parametrize(
    ("jobid",),  # noqa: PT006
    [("",), ("..",), ("../escape",), ("a/b",), ("A",), ("-x",), ("a.b",), ("a\n",), ("a" * 65,), (None,), (1,)],
)
def test_unsafe_archive_job_ids(record: dict, jobid: object) -> None:
    with pytest.raises(ValueError, match="job identifier"):
        archive_environment(record, jobid)


@pytest.mark.parametrize(("value",), [(0,), (-1,), (True,), (False,), (1.0,), ("1",), (None,)])  # noqa: PT006
def test_invalid_archive_free_space(record: dict, value: object) -> None:
    with pytest.raises(ValueError, match="Minimum free bytes"):
        archive_environment(record, "job", min_free_bytes=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mount_path", "/"),
        ("mount_path", "/tmp/artifacts"),
        ("mount_path", VOLUME_MOUNT + "/../escape"),
        ("store_path", "/etc"),
        ("store_path", STORE_ROOT + "/../../escape"),
        ("store_path", "artifacts-v1"),
        ("name", "../escape"),
        ("id", ""),
        ("id", "../escape"),
        ("id", "id\n"),
        ("id", None),
        ("created", "false"),
        ("created", 0),
        ("subpath", "private"),
        ("credentials", {"token": "secret"}),
        ("unknown", "value"),
    ],
)
def test_invalid_or_unknown_record_fields_rejected_by_both_helpers(record: dict, field: str, value: object) -> None:
    pytest.importorskip("daytona")
    record[field] = value
    with pytest.raises(ValueError, match="[Vv]olume"):
        volume_mount(record)
    with pytest.raises(ValueError, match="[Vv]olume"):
        archive_environment(record, "job")


@pytest.mark.parametrize(("field",), [("id",), ("name",), ("mount_path",), ("store_path",), ("created",)])  # noqa: PT006
def test_incomplete_volume_records_are_rejected(record: dict, field: str) -> None:
    del record[field]
    with pytest.raises(ValueError, match="record fields"):
        archive_environment(record, "job")


def test_ordinary_directory_is_not_a_volume_and_no_fallback_is_created(tmp_path: Path) -> None:
    assert not tmp_path.is_mount()
    with pytest.raises(ValueError, match="refusing root-disk storage"):
        verify_volume_mount(tmp_path)
    assert list(tmp_path.iterdir()) == []
    missing = tmp_path / "missing" / "artifacts-v1"
    with pytest.raises(ValueError, match="directory"):
        verify_volume_mount(missing)
    assert not missing.parent.exists()


@pytest.mark.parametrize(("kind",), [("file",), ("symlink",), ("broken-symlink",), ("linked-parent",)])  # noqa: PT006
def test_real_non_directory_and_symlink_mounts_rejected(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "child").mkdir()
    candidate = tmp_path / "mount"
    if kind == "file":
        candidate.write_text("preserve")
    else:
        candidate.symlink_to(target if kind != "broken-symlink" else tmp_path / "absent", target_is_directory=True)
    path = candidate / "child" if kind == "linked-parent" else candidate
    with pytest.raises(ValueError, match="directory without symlink ancestry"):
        verify_volume_mount(path)
    assert list(target.iterdir()) == [target / "child"]
    assert list((target / "child").iterdir()) == []
    assert candidate.read_text() == "preserve" if kind == "file" else candidate.is_symlink()


@pytest.mark.parametrize(
    ("path",),  # noqa: PT006
    [("/",), ("/mnt",), ("/tmp",), ("/proc",), ("/sys",), ("/dev",), ("/etc",), ("/usr",), ("/workspace",)],
)
def test_root_and_system_paths_are_never_managed_artifact_volumes(path: str) -> None:
    with pytest.raises(ValueError, match="[Aa]rtifact volume"):
        verify_volume_mount(Path(path))


@pytest.mark.parametrize(("path",), [(".",), ("mnt/pocket-context-artifacts",), ("../escape",)])  # noqa: PT006
def test_relative_mount_paths_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="absolute"):
        verify_volume_mount(Path(path))


@pytest.mark.parametrize(("filesystem",), [("fuse",), ("fuse.s3fs",), ("fuse.other-provider",)])  # noqa: PT006
def test_mountinfo_accepts_exact_fuse_mount_with_optional_fields(filesystem: str) -> None:
    mountinfo = (
        "1 0 8:1 / / rw - ext4 /dev/root rw\n"
        f"2 1 0:42 / {VOLUME_MOUNT} rw,nosuid shared:4 master:1 - {filesystem} volume rw\n"
        "3 1 0:43 / /other rw - tmpfs tmpfs rw\n"
    )
    assert _verify_fuse_mountinfo(mountinfo, Path(VOLUME_MOUNT)) is None


@pytest.mark.parametrize(("filesystem",), [("tmpfs",), ("ext4",), ("overlay",), ("xfs",)])  # noqa: PT006
def test_mountinfo_rejects_non_fuse_fallbacks(filesystem: str) -> None:
    with pytest.raises(ValueError, match="FUSE mountinfo"):
        _verify_fuse_mountinfo(f"2 1 8:1 / {VOLUME_MOUNT} rw - {filesystem} /dev/root rw", Path(VOLUME_MOUNT))


@pytest.mark.parametrize(
    ("mountinfo",),  # noqa: PT006
    [
        ("",),
        ("1 0 0:1 / / rw - fuse root rw",),
        (f"2 1 0:2 / {VOLUME_MOUNT}/child rw - fuse volume rw",),
        (f"2 1 0:2 / {VOLUME_MOUNT}-other rw - fuse volume rw",),
        (f"2 1 0:2 / {VOLUME_MOUNT} rw fuse volume rw",),
        (f"2 1 0:2 / {VOLUME_MOUNT} rw - fuse",),
        (f"{VOLUME_MOUNT} - fuse volume rw",),
        (f"2 1 0:2 / {VOLUME_MOUNT} rw - fuse volume rw\n3 1 0:3 / {VOLUME_MOUNT} rw - tmpfs tmpfs rw",),
    ],
)
def test_missing_malformed_or_stacked_mountinfo_fails_closed(mountinfo: str) -> None:
    with pytest.raises(ValueError, match="mountinfo"):
        _verify_fuse_mountinfo(mountinfo, Path(VOLUME_MOUNT))


def test_mountinfo_decodes_kernel_octal_path_escapes() -> None:
    assert (
        _verify_fuse_mountinfo(
            r"2 1 0:2 / /mnt/with\040space\011tab\012newline\134slash rw - fuse volume rw",
            Path("/mnt/with space\ttab\nnewline\\slash"),
        )
        is None
    )
