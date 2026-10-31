"""Bounded, single-flight resources shared once by each configured method."""

import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from threading import Condition
from types import TracebackType
from typing import Generic, TypeVar, cast

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    claimed: set[str] = field(default_factory=set)
    released: int = 0
    ready: bool = False
    resource: T | None = None


class SharedSnapshots(Generic[T]):
    """Join all consumer threads before exiting the pool.

    Capacity includes creates and destroys; only readiness waits time out.
    Optional order constrains new-key admission, not leases of existing keys.
    Failed creates retain no resource ownership here: the caller's journal must
    clean up anything created by a callback that never successfully returned.
    """

    def __init__(
        self,
        methods: set[str],
        capacity: int,
        create: Callable[[dict], T],
        destroy: Callable[[T], None],
        timeout: float,
        *,
        order: list[str] | None = None,
    ) -> None:
        if not methods or capacity < 1 or timeout <= 0:
            raise ValueError("Methods, positive capacity and positive timeout are required")
        self._order = None if order is None else {key: index for index, key in enumerate(order)}
        if order is not None and len(set(order)) != len(order):
            raise ValueError("Snapshot order must not contain duplicate keys")
        self._next_index = 0
        self._methods = frozenset(methods)
        self._capacity, self._create, self._destroy, self._timeout = capacity, create, destroy, timeout
        self._condition = Condition()
        self._entries: dict[str, _Entry[T]] = {}
        self._finished: set[str] = set()
        self._error: BaseException | None = None
        self._closed = False

    def __enter__(self) -> "SharedSnapshots[T]":
        with self._condition:
            self._check()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, error: BaseException | None, tb: TracebackType | None
    ) -> None:
        if error is not None:
            self.abort(error)
        try:
            self.close()
        finally:
            cleanup_error = sys.exc_info()[1]
            if error is not None and cleanup_error is not None and cleanup_error is not error:
                raise error.with_traceback(tb) from cleanup_error

    def abort(self, error: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def raise_if_aborted(self) -> None:
        """Recheck after a callback's build semaphore, before external work."""
        with self._condition:
            self._check()

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError("Shared snapshot pool aborted") from self._error
        if self._closed:
            raise RuntimeError("Shared snapshot pool closed")

    @contextmanager
    def lease(self, method: str, key: str, payload: dict) -> Iterator[T]:
        entry: _Entry[T] | None = None
        success = False
        try:
            with self._condition:
                self._check()
                if method not in self._methods:
                    raise ValueError(f"Unknown snapshot method: {method}")
                if self._order is not None and key not in self._order:
                    raise ValueError(f"Unknown snapshot key: {key}")
                while key not in self._entries and (
                    len(self._entries) >= self._capacity
                    or (self._order is not None and self._order[key] != self._next_index)
                ):
                    if key in self._finished:
                        raise ValueError(f"Snapshot key already completed: {key}")
                    self._condition.wait()
                    self._check()
                if key in self._finished:
                    raise ValueError(f"Snapshot key already completed: {key}")
                creator = key not in self._entries
                candidate = self._entries.setdefault(key, _Entry())
                if creator:
                    self._next_index += 1
                if method in candidate.claimed:
                    raise ValueError(f"Duplicate snapshot lease: {method}/{key}")
                candidate.claimed.add(method)
                entry = candidate
                self._condition.notify_all()
            if creator:
                resource = self._create(payload)
                with self._condition:
                    entry.resource, entry.ready = resource, True
                    self._condition.notify_all()
            with self._condition:
                if not self._condition.wait_for(
                    lambda: entry.ready or self._error is not None or self._closed, timeout=self._timeout
                ):
                    raise TimeoutError(f"Snapshot creation timed out: {key}")
                self._check()
            yield cast(T, entry.resource)
            success = True
        finally:
            if not success:
                self.abort(sys.exc_info()[1] or RuntimeError("Snapshot lease interrupted"))
            if entry is not None and entry.ready:
                with self._condition:
                    entry.released += 1
                    destroy = entry.released == len(self._methods)
                if destroy:
                    self._dispose(key, entry)

    def _dispose(self, key: str, entry: _Entry[T]) -> None:
        success = False
        try:
            self._destroy(cast(T, entry.resource))
            success = True
        finally:
            with self._condition:
                if success:
                    del self._entries[key]
                    self._finished.add(key)
                else:
                    self.abort(sys.exc_info()[1] or RuntimeError("Snapshot destruction interrupted"))
                self._condition.notify_all()

    def close(self) -> None:
        """After workers join, destroy all returned resources, including on abort."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            remaining = [(key, entry) for key, entry in self._entries.items() if entry.ready]
        with ExitStack() as cleanup:
            for key, entry in remaining:
                cleanup.callback(self._dispose, key, entry)
