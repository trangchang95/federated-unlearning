"""Cost measurement for the Month 4 unlearning comparison.

Month 4 compares Full Retraining, FedEraser and KD-based FU on cost as well
as on accuracy.  This module only *measures*; it does not train or unlearn.

- ``PhaseTimer`` records wall-clock time (``time.perf_counter``) for named
  phases, so e.g. the unlearning step and the recovery/calibration rounds
  after it can be timed separately and then summed into one total.
- ``speedup`` is ``Time_retrain / Time_FU`` computed from wall-clock seconds.
- ``model_state_size_bytes`` (re-exported from ``algorithms.fedavg``, not
  duplicated) counts dense tensor payload bytes from shapes and dtypes.
- ``storage_bytes`` sums that count over a list of stored model states or
  updates, for later accounting of what a method keeps on the server.

All byte counts are raw dense-tensor bytes (``numel * element_size`` per
tensor).  They do not include serialization headers, compression, or
network protocol overhead.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from numbers import Real

from algorithms.fedavg import ModelState, model_state_size_bytes


__all__ = [
    "PhaseTimer",
    "model_state_size_bytes",
    "speedup",
    "storage_bytes",
]


class PhaseTimer:
    """Accumulate wall-clock seconds per named phase.

    Usage::

        timer = PhaseTimer()
        with timer.phase("unlearning"):
            ...
        for _ in range(recovery_rounds):
            with timer.phase("recovery"):
                ...
        timer.total_seconds  # unlearning + every recovery round

    Entering the same phase name again adds to its total and increments its
    count.  Phases may not be nested: nesting would count the inner time
    twice in ``total_seconds``.  If the timed block raises, the elapsed time
    is still recorded (the time was spent) and the exception propagates.

    ``clock`` defaults to ``time.perf_counter`` and exists so tests can
    inject a deterministic clock.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._seconds: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._active_phase: str | None = None

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time one execution of the ``name`` phase."""
        if not isinstance(name, str) or not name:
            raise ValueError("Phase name must be a non-empty string.")
        if self._active_phase is not None:
            raise RuntimeError(
                f"Cannot start phase {name!r} while phase "
                f"{self._active_phase!r} is running; phases may not be nested."
            )
        self._active_phase = name
        started = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - started
            self._active_phase = None
            if elapsed < 0:
                raise RuntimeError("Clock went backwards while timing a phase.")
            self._seconds[name] = self._seconds.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    def seconds(self, name: str) -> float:
        """Return accumulated seconds for ``name``; unknown names raise."""
        if name not in self._seconds:
            raise KeyError(f"No phase named {name!r} has been recorded.")
        return self._seconds[name]

    def count(self, name: str) -> int:
        """Return how many times ``name`` was entered; unknown names raise."""
        if name not in self._counts:
            raise KeyError(f"No phase named {name!r} has been recorded.")
        return self._counts[name]

    @property
    def phase_seconds(self) -> dict[str, float]:
        """Copy of accumulated seconds per phase, in first-recorded order."""
        return dict(self._seconds)

    @property
    def total_seconds(self) -> float:
        """Sum of every recorded phase."""
        return float(sum(self._seconds.values()))

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable summary for a result file."""
        return {
            "phase_seconds": dict(self._seconds),
            "phase_counts": dict(self._counts),
            "total_seconds": self.total_seconds,
        }


def _validate_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number of seconds.")
    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError(f"{name} must be finite.")
    if seconds <= 0:
        raise ValueError(f"{name} must be positive wall-clock seconds.")
    return seconds


def speedup(time_retrain: float, time_fu: float) -> float:
    """Return ``Time_retrain / Time_FU`` from wall-clock seconds.

    ``time_fu`` must include every round the method needs before its model
    is usable (e.g. recovery or calibration rounds), otherwise the ratio
    overstates the method's advantage.  A value above 1 means the FU method
    finished faster than full retraining.
    """
    retrain_seconds = _validate_seconds(time_retrain, "time_retrain")
    fu_seconds = _validate_seconds(time_fu, "time_fu")
    return retrain_seconds / fu_seconds


def storage_bytes(states: Sequence[ModelState]) -> int:
    """Sum dense payload bytes over a sequence of stored states or updates.

    Each state is counted independently with ``model_state_size_bytes``,
    i.e. as if serialized separately; storage shared between tensors is not
    de-duplicated.  An empty sequence stores 0 bytes.
    """
    if isinstance(states, Mapping):
        raise TypeError(
            "storage_bytes expects a sequence of states; for one state use "
            "model_state_size_bytes."
        )
    if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
        raise TypeError("storage_bytes expects a sequence of model states.")
    total = 0
    for index, state in enumerate(states):
        if not isinstance(state, Mapping):
            raise TypeError(f"Stored entry {index} is not a state mapping.")
        total += model_state_size_bytes(state)
    return total
