"""Checks for Month 4 cost measurement: bytes, speedup and phase timing."""

from __future__ import annotations

import json
import math
import time
import unittest

import torch
from torch import nn

from algorithms.fedavg import model_state_size_bytes as fedavg_size_bytes
from evaluation.efficiency import (
    PhaseTimer,
    model_state_size_bytes,
    speedup,
    storage_bytes,
)
from models.simple_mlp import SimpleMLP


class FakeClock:
    """Return a scripted sequence of timestamps, one per call."""

    def __init__(self, timestamps: list[float]) -> None:
        self._timestamps = list(timestamps)

    def __call__(self) -> float:
        return self._timestamps.pop(0)


class PayloadByteTests(unittest.TestCase):
    def test_efficiency_reuses_the_fedavg_helper_rather_than_a_copy(self) -> None:
        self.assertIs(model_state_size_bytes, fedavg_size_bytes)

    def test_simple_mlp_float32_payload_matches_hand_count(self) -> None:
        state = SimpleMLP().state_dict()
        # 784*128 + 128 + 128*10 + 10 = 101,770 float32 values * 4 bytes.
        self.assertEqual(sum(tensor.numel() for tensor in state.values()), 101_770)
        self.assertEqual(model_state_size_bytes(state), 407_080)

    def test_mixed_dtypes_use_each_tensor_element_size(self) -> None:
        state = {
            "float32_matrix": torch.zeros(3, 4, dtype=torch.float32),  # 12*4 = 48
            "float16_vector": torch.zeros(5, dtype=torch.float16),  # 5*2 = 10
            "float64_matrix": torch.zeros(2, 2, dtype=torch.float64),  # 4*8 = 32
            "int64_scalar": torch.tensor(7, dtype=torch.int64),  # 1*8 = 8
            "bool_vector": torch.zeros(3, dtype=torch.bool),  # 3*1 = 3
        }
        self.assertEqual(model_state_size_bytes(state), 48 + 10 + 32 + 8 + 3)

    def test_batchnorm_buffers_are_counted(self) -> None:
        state = nn.BatchNorm1d(4).state_dict()
        # weight, bias, running_mean, running_var: 4 float32 each = 64 bytes;
        # num_batches_tracked: one int64 scalar = 8 bytes.
        self.assertEqual(model_state_size_bytes(state), 4 * 4 * 4 + 8)

    def test_non_tensor_entry_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            model_state_size_bytes({"weight": [1.0, 2.0]})


class StorageByteTests(unittest.TestCase):
    def test_storage_sums_each_stored_state(self) -> None:
        state = SimpleMLP().state_dict()
        self.assertEqual(storage_bytes([state, state, state]), 3 * 407_080)

    def test_storage_of_mixed_dtype_states(self) -> None:
        first = {"w": torch.zeros(10, dtype=torch.float32)}  # 40
        second = {"w": torch.zeros(10, dtype=torch.float16)}  # 20
        self.assertEqual(storage_bytes([first, second]), 60)
        self.assertEqual(storage_bytes((first, second)), 60)

    def test_empty_storage_is_zero_bytes(self) -> None:
        self.assertEqual(storage_bytes([]), 0)

    def test_single_state_instead_of_a_sequence_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "model_state_size_bytes"):
            storage_bytes(SimpleMLP().state_dict())

    def test_non_mapping_entry_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "entry 1"):
            storage_bytes([{"w": torch.zeros(1)}, torch.zeros(1)])

    def test_string_is_not_a_sequence_of_states(self) -> None:
        with self.assertRaises(TypeError):
            storage_bytes("not states")


class SpeedupTests(unittest.TestCase):
    def test_speedup_is_retrain_time_over_fu_time(self) -> None:
        self.assertEqual(speedup(10.0, 2.0), 5.0)
        self.assertEqual(speedup(3, 4), 0.75)

    def test_zero_or_negative_times_are_rejected(self) -> None:
        for retrain, fu in ((10.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (1.0, -2.0)):
            with self.subTest(retrain=retrain, fu=fu):
                with self.assertRaises(ValueError):
                    speedup(retrain, fu)

    def test_non_finite_times_are_rejected(self) -> None:
        for retrain, fu in ((math.nan, 1.0), (1.0, math.nan), (math.inf, 1.0), (1.0, math.inf)):
            with self.subTest(retrain=retrain, fu=fu):
                with self.assertRaises(ValueError):
                    speedup(retrain, fu)

    def test_non_numeric_times_are_rejected(self) -> None:
        for retrain, fu in ((True, 1.0), (1.0, False), ("10", 1.0), (1.0, None)):
            with self.subTest(retrain=retrain, fu=fu):
                with self.assertRaises(TypeError):
                    speedup(retrain, fu)


class PhaseTimerTests(unittest.TestCase):
    def test_default_clock_is_perf_counter(self) -> None:
        timer = PhaseTimer()
        with timer.phase("real"):
            pass
        self.assertGreaterEqual(timer.seconds("real"), 0.0)
        self.assertIs(timer._clock, time.perf_counter)

    def test_repeated_phases_accumulate_and_total_sums_all_phases(self) -> None:
        # unlearning: 0->2 (2s); recovery: 5->6 (1s), 10->13 (3s).
        timer = PhaseTimer(clock=FakeClock([0.0, 2.0, 5.0, 6.0, 10.0, 13.0]))
        with timer.phase("unlearning"):
            pass
        for _ in range(2):
            with timer.phase("recovery"):
                pass
        self.assertEqual(timer.seconds("unlearning"), 2.0)
        self.assertEqual(timer.seconds("recovery"), 4.0)
        self.assertEqual(timer.count("unlearning"), 1)
        self.assertEqual(timer.count("recovery"), 2)
        self.assertEqual(timer.total_seconds, 6.0)
        self.assertEqual(list(timer.phase_seconds), ["unlearning", "recovery"])

    def test_time_outside_phases_is_not_counted(self) -> None:
        # Gap between 1.0 and 100.0 happens outside any phase.
        timer = PhaseTimer(clock=FakeClock([0.0, 1.0, 100.0, 101.5]))
        with timer.phase("a"):
            pass
        with timer.phase("b"):
            pass
        self.assertEqual(timer.total_seconds, 2.5)

    def test_to_dict_is_json_serializable(self) -> None:
        timer = PhaseTimer(clock=FakeClock([0.0, 0.5, 1.0, 1.25]))
        with timer.phase("unlearning"):
            pass
        with timer.phase("calibration"):
            pass
        summary = json.loads(json.dumps(timer.to_dict()))
        self.assertEqual(
            summary,
            {
                "phase_seconds": {"unlearning": 0.5, "calibration": 0.25},
                "phase_counts": {"unlearning": 1, "calibration": 1},
                "total_seconds": 0.75,
            },
        )

    def test_nested_phases_are_rejected(self) -> None:
        timer = PhaseTimer(clock=FakeClock([0.0, 1.0]))
        with self.assertRaisesRegex(RuntimeError, "nested"):
            with timer.phase("outer"):
                with timer.phase("inner"):
                    pass
        # The outer phase is still recorded and the timer is usable again.
        self.assertEqual(timer.seconds("outer"), 1.0)
        with self.assertRaises(KeyError):
            timer.seconds("inner")

    def test_exception_inside_phase_is_recorded_and_propagated(self) -> None:
        timer = PhaseTimer(clock=FakeClock([0.0, 3.0, 4.0, 5.0]))
        with self.assertRaisesRegex(ZeroDivisionError, "division"):
            with timer.phase("failing"):
                1 / 0
        self.assertEqual(timer.seconds("failing"), 3.0)
        with timer.phase("after"):
            pass
        self.assertEqual(timer.total_seconds, 4.0)

    def test_unknown_phase_raises(self) -> None:
        timer = PhaseTimer()
        with self.assertRaises(KeyError):
            timer.seconds("typo")
        with self.assertRaises(KeyError):
            timer.count("typo")

    def test_empty_timer_total_is_zero(self) -> None:
        self.assertEqual(PhaseTimer().total_seconds, 0.0)

    def test_invalid_phase_name_is_rejected(self) -> None:
        timer = PhaseTimer()
        for name in ("", None, 3):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    with timer.phase(name):
                        pass

    def test_clock_going_backwards_is_rejected(self) -> None:
        timer = PhaseTimer(clock=FakeClock([5.0, 4.0]))
        with self.assertRaisesRegex(RuntimeError, "backwards"):
            with timer.phase("broken"):
                pass


if __name__ == "__main__":
    unittest.main()
