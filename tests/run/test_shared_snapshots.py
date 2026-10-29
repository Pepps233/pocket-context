from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Barrier, Event, Semaphore

import pytest

from pocketcontextagent.run.benchmarks.utils.shared_snapshots import SharedSnapshots

METHODS = {"a", "b", "c", "d"}


@dataclass
class Resources:
    root: Path
    created: list[Path] = field(default_factory=list)
    destroyed: list[Path] = field(default_factory=list)

    def create(self, payload: dict) -> Path:
        resource = self.root / payload["key"]
        resource.mkdir()
        (resource / "image").write_text(payload["key"])
        self.created.append(resource)
        return resource

    def destroy(self, resource: Path) -> None:
        (resource / "image").unlink()
        resource.rmdir()
        self.destroyed.append(resource)


@pytest.fixture
def resources(tmp_path: Path) -> Resources:
    return Resources(tmp_path)


@contextmanager
def workers(pool: SharedSnapshots[Path]) -> Iterator[ThreadPoolExecutor]:
    with ThreadPoolExecutor(max_workers=8) as executor:
        try:
            yield executor
        finally:
            pool.abort(RuntimeError("Test worker shutdown"))


def test_four_methods_share_one_single_flight_resource(resources: Resources) -> None:
    start, leased = Barrier(5), Barrier(5)
    creating, allow_create, release = Event(), Event(), Event()
    shared: list[Path] = []

    def create(payload: dict) -> Path:
        creating.set()
        assert allow_create.wait(5)
        return resources.create(payload)

    pool = SharedSnapshots(METHODS, 1, create, resources.destroy, 2)

    def consume(method: str) -> None:
        start.wait(5)
        with pool.lease(method, "question", {"key": "question"}) as resource:
            shared.append(resource)
            leased.wait(5)
            assert release.wait(5)
            assert (resource / "image").read_text() == "question"

    with pool, workers(pool) as executor:
        futures = [executor.submit(consume, method) for method in METHODS]
        try:
            start.wait(5)
            assert creating.wait(5)
            assert resources.created == []
            allow_create.set()
            leased.wait(5)
            assert len(resources.created) == 1
            assert len(shared) == 4 and all(resource is shared[0] for resource in shared)
            assert resources.destroyed == []
            release.set()
            for future in futures:
                future.result(5)
            assert resources.destroyed == resources.created
        finally:
            allow_create.set()
            release.set()
    assert list(resources.root.iterdir()) == []


def test_capacity_wait_has_no_readiness_deadline_and_waits_for_destroy(resources: Resources) -> None:
    last_leased, release_last, destroying, allow_destroy, next_attempt, next_created = (Event() for _ in range(6))

    def create(payload: dict) -> Path:
        assert list(resources.root.iterdir()) == []
        resource = resources.create(payload)
        if payload["key"] == "next":
            next_created.set()
        return resource

    def destroy(resource: Path) -> None:
        if resource.name == "first":
            destroying.set()
            assert allow_destroy.wait(5)
        resources.destroy(resource)

    pool = SharedSnapshots(METHODS, 1, create, destroy, 0.03)
    with pool, workers(pool) as executor:
        for method in ("a", "b", "c"):
            with pool.lease(method, "first", {"key": "first"}):
                pass

        def last() -> None:
            with pool.lease("d", "first", {}):
                last_leased.set()
                assert release_last.wait(5)

        def following() -> None:
            next_attempt.set()
            with pool.lease("a", "next", {"key": "next"}) as resource:
                assert not (resources.root / "first").exists()
                assert resource.is_dir()

        last_future = executor.submit(last)
        try:
            assert last_leased.wait(5)
            next_future = executor.submit(following)
            assert next_attempt.wait(5)
            assert not next_created.wait(0.15)
            assert not next_future.done()
            release_last.set()
            assert destroying.wait(5)
            assert (resources.root / "first").is_dir()
            assert not next_created.wait(0.15)
            allow_destroy.set()
            last_future.result(5)
            next_future.result(5)
            assert [path.name for path in resources.destroyed] == ["first"]
        finally:
            release_last.set()
            allow_destroy.set()
    assert resources.destroyed == resources.created
    assert list(resources.root.iterdir()) == []


def test_different_keys_have_no_batch_barrier(resources: Resources) -> None:
    with SharedSnapshots(METHODS, 2, resources.create, resources.destroy, 1) as pool:
        with pool.lease("a", "slow", {"key": "slow"}):
            pass
        for method in sorted(METHODS):
            with pool.lease(method, "fast", {"key": "fast"}):
                assert (resources.root / "slow").is_dir()
        assert [path.name for path in resources.destroyed] == ["fast"]
        assert (resources.root / "slow").is_dir()
        for method in ("b", "c", "d"):
            with pool.lease(method, "slow", {}):
                pass
        assert [path.name for path in resources.destroyed] == ["fast", "slow"]
    assert list(resources.root.iterdir()) == []


def test_failing_creator_wakes_readiness_and_capacity_waiters(resources: Resources) -> None:
    creating, fail, same_attempt, next_attempt = (Event() for _ in range(4))

    def create(payload: dict) -> Path:
        creating.set()
        assert fail.wait(5)
        return Path((resources.root / "missing").read_text())

    pool = SharedSnapshots({"a", "b"}, 1, create, resources.destroy, 5, order=["first", "next"])

    def consume(method: str, key: str, attempted: Event) -> None:
        attempted.set()
        with pool.lease(method, key, {}):
            pytest.fail("A failed creation must not yield a resource")

    with pool, workers(pool) as executor:
        creator = executor.submit(consume, "a", "first", Event())
        try:
            assert creating.wait(5)
            same = executor.submit(consume, "b", "first", same_attempt)
            following = executor.submit(consume, "a", "next", next_attempt)
            assert same_attempt.wait(5) and next_attempt.wait(5)
            fail.set()
            with pytest.raises(FileNotFoundError):
                creator.result(5)
            for future in (same, following):
                with pytest.raises(RuntimeError, match="aborted") as failure:
                    future.result(5)
                assert isinstance(failure.value.__cause__, FileNotFoundError)
        finally:
            fail.set()
    assert resources.created == resources.destroyed == []


def test_lease_body_failure_aborts_capacity_waiter_and_cleans_resource(resources: Resources) -> None:
    leased, fail, attempted = Event(), Event(), Event()
    pool = SharedSnapshots(METHODS, 1, resources.create, resources.destroy, 1)

    def failing() -> None:
        with pool.lease("a", "first", {"key": "first"}) as resource:
            leased.set()
            assert fail.wait(5)
            (resource / "missing").read_text()

    def waiting() -> None:
        attempted.set()
        with pool.lease("a", "next", {"key": "next"}):
            pytest.fail("Capacity waiter must abort")

    with pool, workers(pool) as executor:
        first = executor.submit(failing)
        try:
            assert leased.wait(5)
            following = executor.submit(waiting)
            assert attempted.wait(5)
            fail.set()
            with pytest.raises(FileNotFoundError):
                first.result(5)
            with pytest.raises(RuntimeError, match="aborted") as failure:
                following.result(5)
            assert isinstance(failure.value.__cause__, FileNotFoundError)
            assert resources.destroyed == []
        finally:
            fail.set()
    assert resources.destroyed == resources.created
    assert list(resources.root.iterdir()) == []


def test_readiness_timeout_aborts_but_late_returned_resource_is_cleaned(resources: Resources) -> None:
    creating, finish = Event(), Event()

    def create(payload: dict) -> Path:
        creating.set()
        assert finish.wait(5)
        return resources.create(payload)

    pool = SharedSnapshots({"a", "b"}, 1, create, resources.destroy, 0.03)

    def consume() -> None:
        with pool.lease("a", "first", {"key": "first"}):
            pytest.fail("Creator must observe the pool abort after returning")

    with pool, workers(pool) as executor:
        creator = executor.submit(consume)
        try:
            assert creating.wait(5)
            with pytest.raises(TimeoutError, match="creation timed out"):
                with pool.lease("b", "first", {}):
                    pytest.fail("Creation is still pending")
            finish.set()
            with pytest.raises(RuntimeError, match="aborted") as failure:
                creator.result(5)
            assert isinstance(failure.value.__cause__, TimeoutError)
        finally:
            finish.set()
    assert resources.destroyed == resources.created
    assert len(resources.created) == 1
    assert list(resources.root.iterdir()) == []


@pytest.mark.parametrize(("ordered",), [(True,), (False,)])  # noqa: PT006
@pytest.mark.parametrize(("duplicate",), [("active",), ("released",), ("finished",), ("unknown",)])  # noqa: PT006
def test_unknown_and_duplicate_leases_are_rejected(resources: Resources, duplicate: str, ordered: bool) -> None:
    pool = SharedSnapshots({"a", "b"}, 1, resources.create, resources.destroy, 1, order=["first"] if ordered else None)
    with pool:
        if duplicate == "active":
            with pool.lease("a", "first", {"key": "first"}):
                with pytest.raises(ValueError, match="Duplicate"):
                    with pool.lease("a", "first", {}):
                        pass
        else:
            with pool.lease("a", "first", {"key": "first"}):
                pass
            if duplicate == "finished":
                with pool.lease("b", "first", {}):
                    pass
            with pytest.raises(ValueError, match="Unknown|Duplicate|completed"):
                with pool.lease("unknown" if duplicate == "unknown" else "a", "first", {}):
                    pass
    assert len(resources.created) == len(resources.destroyed) == 1
    assert list(resources.root.iterdir()) == []


def test_close_cleans_pending_keys_and_is_idempotent(resources: Resources) -> None:
    pool = SharedSnapshots(METHODS, 2, resources.create, resources.destroy, 1)
    with pool:
        for key in ("first", "second"):
            with pool.lease("a", key, {"key": key}):
                pass
        assert len(list(resources.root.iterdir())) == 2
        assert resources.destroyed == []
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.raise_if_aborted()
    assert len(resources.created) == len(resources.destroyed) == 2
    assert list(resources.root.iterdir()) == []
    with pytest.raises(RuntimeError, match="closed"):
        with pool.lease("b", "first", {}):
            pass


def test_destroy_failure_aborts_capacity_waiter(resources: Resources) -> None:
    destroying, fail, attempted = Event(), Event(), Event()

    def destroy(resource: Path) -> None:
        destroying.set()
        assert fail.wait(5)
        (resource / "missing").unlink()

    pool = SharedSnapshots({"a"}, 1, resources.create, destroy, 1)

    def consume(key: str) -> None:
        attempted.set()
        with pool.lease("a", key, {"key": key}):
            pass

    with workers(pool) as executor:
        first = executor.submit(consume, "first")
        try:
            assert destroying.wait(5)
            attempted.clear()
            following = executor.submit(consume, "next")
            assert attempted.wait(5)
            fail.set()
            with pytest.raises(FileNotFoundError):
                first.result(5)
            with pytest.raises(RuntimeError, match="aborted") as failure:
                following.result(5)
            assert isinstance(failure.value.__cause__, FileNotFoundError)
        finally:
            fail.set()
    with pytest.raises(FileNotFoundError):
        pool.close()
    assert [path.name for path in resources.created] == ["first"]
    resources.destroy(resources.created[0])
    assert list(resources.root.iterdir()) == []


def test_close_attempts_all_destroys_and_preserves_primary_error(resources: Resources) -> None:
    def destroy(resource: Path) -> None:
        resources.destroy(resource)
        if resource.name == "second":
            (resources.root / "missing-cleanup").read_text()

    def run() -> None:
        with SharedSnapshots(METHODS, 2, resources.create, destroy, 1) as pool:
            for key in ("first", "second"):
                with pool.lease("a", key, {"key": key}):
                    pass
            (resources.root / "missing-primary").read_text()

    with pytest.raises(FileNotFoundError, match="missing-primary") as failure:
        run()
    assert isinstance(failure.value.__cause__, FileNotFoundError)
    assert "missing-cleanup" in str(failure.value.__cause__)
    assert len(resources.destroyed) == 2
    assert list(resources.root.iterdir()) == []


def test_queued_creator_rechecks_abort_after_build_semaphore(resources: Resources) -> None:
    build, queued = Semaphore(1), Event()

    def create(payload: dict) -> Path:
        if payload["key"] == "queued":
            queued.set()
            with build:
                pool.raise_if_aborted()
                return resources.create(payload)
        return resources.create(payload)

    pool = SharedSnapshots(METHODS, 2, create, resources.destroy, 1)

    def consume() -> None:
        with pool.lease("a", "queued", {"key": "queued"}):
            pytest.fail("An aborted queued creator must not yield")

    with pool, workers(pool) as executor:
        with build:
            with pytest.raises(FileNotFoundError):  # noqa: PT012
                with pool.lease("a", "first", {"key": "first"}) as resource:
                    following = executor.submit(consume)
                    assert queued.wait(5)
                    (resource / "missing").read_text()
        with pytest.raises(RuntimeError, match="aborted") as failure:
            following.result(5)
        assert isinstance(failure.value.__cause__, FileNotFoundError)
    assert [path.name for path in resources.created] == ["first"]
    assert resources.destroyed == resources.created
    assert list(resources.root.iterdir()) == []


def test_failed_create_partial_resource_remains_owned_by_callers_journal(resources: Resources) -> None:
    def create(payload: dict) -> Path:
        resource = resources.create(payload)
        (resource / "missing").read_text()
        return resource

    with SharedSnapshots(METHODS, 1, create, resources.destroy, 1) as pool:
        with pytest.raises(FileNotFoundError):
            with pool.lease("a", "partial", {"key": "partial"}):
                pass
    assert len(resources.created) == 1
    assert resources.destroyed == []
    assert resources.created[0].is_dir()
    resources.destroy(resources.created[0])
    assert list(resources.root.iterdir()) == []


def test_ordered_admission_earlier_key_wins_even_when_later_waiter_arrives_first(resources: Resources) -> None:
    later_attempt, earlier_leased, release_earlier, later_leased = (Event() for _ in range(4))
    pool = SharedSnapshots(
        {"a", "b"}, 1, resources.create, resources.destroy, 0.02, order=["first", "earlier", "later"]
    )

    def later() -> None:
        later_attempt.set()
        with pool.lease("a", "later", {"key": "later"}):
            later_leased.set()

    def earlier() -> None:
        with pool.lease("a", "earlier", {"key": "earlier"}) as resource:
            assert resource.is_dir()
            earlier_leased.set()
            assert release_earlier.wait(5)

    with pool, workers(pool) as executor:
        with pool.lease("a", "first", {"key": "first"}):
            pass
        later_future = executor.submit(later)
        try:
            assert later_attempt.wait(5)
            assert not later_leased.wait(0.1)
            earlier_future = executor.submit(earlier)
            with pool.lease("b", "first", {}):
                pass
            assert earlier_leased.wait(5)
            assert [path.name for path in resources.created] == ["first", "earlier"]
            assert not later_leased.is_set()
            with pool.lease("b", "earlier", {}) as resource:
                assert resource.name == "earlier"
            release_earlier.set()
            earlier_future.result(5)
            later_future.result(5)
            assert [path.name for path in resources.created] == ["first", "earlier", "later"]
        finally:
            release_earlier.set()
    assert resources.destroyed == resources.created
    assert list(resources.root.iterdir()) == []


def test_order_waits_without_deadline_even_when_capacity_available(resources: Resources) -> None:
    attempted, later_leased = Event(), Event()
    pool = SharedSnapshots({"a"}, 2, resources.create, resources.destroy, 0.02, order=["first", "later"])

    def later() -> None:
        attempted.set()
        with pool.lease("a", "later", {"key": "later"}):
            later_leased.set()

    with pool, workers(pool) as executor:
        following = executor.submit(later)
        assert attempted.wait(5)
        assert not later_leased.wait(0.1)
        assert not following.done()
        assert resources.created == []
        with pool.lease("a", "first", {"key": "first"}):
            assert later_leased.wait(5)
        following.result(5)
    assert {path.name for path in resources.created} == {"first", "later"}
    assert set(resources.destroyed) == set(resources.created)
    assert list(resources.root.iterdir()) == []


@pytest.mark.parametrize(("order",), [([],), (["known"],)])  # noqa: PT006
def test_ordered_pool_rejects_unknown_keys(resources: Resources, order: list[str]) -> None:
    with SharedSnapshots(METHODS, 1, resources.create, resources.destroy, 1, order=order) as pool:
        with pytest.raises(ValueError, match="Unknown snapshot key"):
            with pool.lease("a", "unknown", {"key": "unknown"}):
                pass
    assert resources.created == resources.destroyed == []
    assert list(resources.root.iterdir()) == []


def test_order_rejects_duplicate_keys(resources: Resources) -> None:
    with pytest.raises(ValueError, match="duplicate keys"):
        SharedSnapshots(METHODS, 1, resources.create, resources.destroy, 1, order=["same", "same"])
    assert list(resources.root.iterdir()) == []


@pytest.mark.parametrize(("methods", "capacity", "timeout"), [(set(), 1, 1), ({"a"}, 0, 1), ({"a"}, 1, 0)])
def test_invalid_configuration(resources: Resources, methods: set[str], capacity: int, timeout: float) -> None:
    with pytest.raises(ValueError):
        SharedSnapshots(methods, capacity, resources.create, resources.destroy, timeout)
    assert list(resources.root.iterdir()) == []
