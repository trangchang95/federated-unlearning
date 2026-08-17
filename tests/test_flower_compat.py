"""Week 7 checks that Flower and hand-written FedAvg agree on aggregation."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from algorithms.fedavg import clone_model_state, weighted_average_model_states
from server.fedavg_server import select_client_ids
from server.flower_compat import (
    EXPECTED_FLOWER_VERSION,
    flower_weighted_average_model_states,
    installed_flower_version,
)


class FlowerCompatibilityTests(unittest.TestCase):
    def test_expected_flower_version_is_installed(self) -> None:
        self.assertEqual(installed_flower_version(), EXPECTED_FLOWER_VERSION)
        self.assertEqual(installed_flower_version(), "1.30.0")

    def test_flower_reproduces_the_20_80_scalar_example(self) -> None:
        result = flower_weighted_average_model_states(
            [{"weight": torch.tensor(1.0)}, {"weight": torch.tensor(3.0)}],
            [20, 80],
        )
        self.assertAlmostEqual(result["weight"].item(), 2.6, places=6)

    def test_flower_and_handwritten_aggregation_match_without_mutation(self) -> None:
        states = [
            {
                "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "bias": torch.tensor([0.5, -0.5]),
            },
            {
                "weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                "bias": torch.tensor([1.5, 0.5]),
            },
        ]
        states_before = [clone_model_state(state) for state in states]
        handwritten = weighted_average_model_states(states, [4, 2])
        framework = flower_weighted_average_model_states(states, [4, 2])

        self.assertEqual(tuple(framework), ("weight", "bias"))
        for name in handwritten:
            torch.testing.assert_close(framework[name], handwritten[name])
            self.assertEqual(framework[name].dtype, states[0][name].dtype)
            self.assertEqual(framework[name].shape, states[0][name].shape)
            self.assertNotEqual(
                framework[name].data_ptr(), states[0][name].data_ptr()
            )
        for state, before in zip(states, states_before):
            for name in state:
                self.assertTrue(torch.equal(state[name], before[name]))

    def test_builtin_strategy_aggregate_fit_matches_transparent_adapter(self) -> None:
        arrays_a = [np.array([1.0, 2.0], dtype=np.float32)]
        arrays_b = [np.array([5.0, 6.0], dtype=np.float32)]
        successful_results = [
            (
                None,
                FitRes(
                    status=Status(code=Code.OK, message="synthetic"),
                    parameters=ndarrays_to_parameters(arrays),
                    num_examples=count,
                    metrics={},
                ),
            )
            for arrays, count in ((arrays_a, 4), (arrays_b, 2))
        ]
        strategy = FedAvg(
            inplace=False,
            fit_metrics_aggregation_fn=lambda _: {},
        )
        parameters, metrics = strategy.aggregate_fit(
            server_round=1,
            results=successful_results,
            failures=[],
        )
        self.assertIsNotNone(parameters)
        strategy_array = parameters_to_ndarrays(parameters)[0]
        adapter = flower_weighted_average_model_states(
            [
                {"weight": torch.from_numpy(arrays_a[0].copy())},
                {"weight": torch.from_numpy(arrays_b[0].copy())},
            ],
            [4, 2],
        )["weight"].numpy()
        np.testing.assert_allclose(strategy_array, adapter)
        self.assertEqual(metrics, {})

    def test_default_inplace_strategy_path_has_same_weighted_result(self) -> None:
        results = [
            (
                None,
                FitRes(
                    status=Status(code=Code.OK, message="synthetic"),
                    parameters=ndarrays_to_parameters(
                        [np.array([value], dtype=np.float32)]
                    ),
                    num_examples=count,
                    metrics={},
                ),
            )
            for value, count in ((1.0, 20), (3.0, 80))
        ]
        strategy = FedAvg(fit_metrics_aggregation_fn=lambda _: {})
        parameters, _ = strategy.aggregate_fit(1, results, [])
        self.assertIsNotNone(parameters)
        self.assertAlmostEqual(
            float(parameters_to_ndarrays(parameters)[0][0]), 2.6, places=6
        )

    def test_unsupported_bfloat16_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "float16, float32, and float64"):
            flower_weighted_average_model_states(
                [{"weight": torch.tensor([1.0], dtype=torch.bfloat16)}], [1]
            )

    def test_selection_rounding_difference_is_explicit(self) -> None:
        handwritten_count = len(select_client_ids([0, 1, 2, 3, 4], 0.3, 42, 1))
        strategy = FedAvg(
            fraction_fit=0.3,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )
        flower_count, required_available = strategy.num_fit_clients(5)

        self.assertEqual(handwritten_count, 2)  # ceil(0.3 * 5)
        self.assertEqual(flower_count, 1)  # max(int(0.3 * 5), min_fit_clients)
        self.assertEqual(required_available, 1)


if __name__ == "__main__":
    unittest.main()
