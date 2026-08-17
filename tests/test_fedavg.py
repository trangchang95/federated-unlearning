"""Deterministic synthetic checks for the hand-written FedAvg foundation."""

from __future__ import annotations

import copy
import random
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from algorithms.fedavg import (
    clone_model_state,
    model_state_size_bytes,
    weighted_average_model_states,
)
from clients.federated_client import train_client
from server.fedavg_server import (
    FedAvgServer,
    client_training_seed,
    select_client_ids,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def make_dataset(offset: float = 0.0) -> TensorDataset:
    features = torch.tensor(
        [
            [1.0 + offset, 0.0],
            [0.0, 1.0 + offset],
            [1.0 + offset, 1.0],
            [-1.0, 1.0 + offset],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return TensorDataset(features, labels)


class FedAvgAggregationTests(unittest.TestCase):
    def test_weighted_scalar_example_is_2_point_6(self) -> None:
        result = weighted_average_model_states(
            [{"weight": torch.tensor(1.0)}, {"weight": torch.tensor(3.0)}],
            [20, 80],
        )
        self.assertAlmostEqual(result["weight"].item(), 2.6, places=6)

    def test_multi_tensor_average_and_inputs_are_not_mutated_or_aliased(self) -> None:
        first = {
            "weight": torch.tensor([1.0, 3.0]),
            "bias": torch.tensor([2.0]),
        }
        second = {
            "weight": torch.tensor([5.0, 7.0]),
            "bias": torch.tensor([6.0]),
        }
        first_before = clone_model_state(first)
        result = weighted_average_model_states([first, second], [1, 3])

        torch.testing.assert_close(result["weight"], torch.tensor([4.0, 6.0]))
        torch.testing.assert_close(result["bias"], torch.tensor([5.0]))
        torch.testing.assert_close(first["weight"], first_before["weight"])
        self.assertNotEqual(result["weight"].data_ptr(), first["weight"].data_ptr())
        result["weight"].add_(10)
        torch.testing.assert_close(first["weight"], first_before["weight"])

    def test_non_floating_buffers_must_agree(self) -> None:
        matching = weighted_average_model_states(
            [{"counter": torch.tensor(2)}, {"counter": torch.tensor(2)}],
            [1, 1],
        )
        self.assertEqual(matching["counter"].item(), 2)
        with self.assertRaisesRegex(ValueError, "Non-floating"):
            weighted_average_model_states(
                [{"counter": torch.tensor(1)}, {"counter": torch.tensor(2)}],
                [1, 1],
            )

    def test_invalid_updates_are_rejected(self) -> None:
        state = {"weight": torch.tensor([1.0])}
        with self.assertRaisesRegex(ValueError, "at least one"):
            weighted_average_model_states([], [])
        with self.assertRaisesRegex(ValueError, "same length"):
            weighted_average_model_states([state], [1, 2])
        with self.assertRaisesRegex(ValueError, "positive"):
            weighted_average_model_states([state], [0])
        with self.assertRaisesRegex(TypeError, "mapping"):
            weighted_average_model_states([state, []], [1, 1])
        with self.assertRaisesRegex(ValueError, "keys"):
            weighted_average_model_states([state, {"bias": torch.tensor([1.0])}], [1, 1])
        with self.assertRaisesRegex(ValueError, "shape"):
            weighted_average_model_states(
                [state, {"weight": torch.tensor([1.0, 2.0])}], [1, 1]
            )
        with self.assertRaisesRegex(ValueError, "dtype"):
            weighted_average_model_states(
                [state, {"weight": torch.tensor([1.0], dtype=torch.float64)}],
                [1, 1],
            )
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            weighted_average_model_states(
                [state, {"weight": torch.tensor([float("nan")])}], [1, 1]
            )

    def test_half_precision_inputs_are_accumulated_and_returned_as_half(self) -> None:
        result = weighted_average_model_states(
            [
                {"weight": torch.tensor([1.0], dtype=torch.float16)},
                {"weight": torch.tensor([3.0], dtype=torch.float16)},
            ],
            [1, 3],
        )
        self.assertEqual(result["weight"].dtype, torch.float16)
        torch.testing.assert_close(result["weight"].float(), torch.tensor([2.5]))


class FederatedWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        torch.use_deterministic_algorithms(True)
        self.initial_model = TinyClassifier()
        self.initial_state = clone_model_state(self.initial_model.state_dict())

    def test_selection_is_deterministic_unique_and_uses_ceiling_rule(self) -> None:
        first = select_client_ids([0, 1, 2, 3, 4], 0.3, 42, 1)
        second = select_client_ids([0, 1, 2, 3, 4], 0.3, 42, 1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(set(first)), 2)

    def test_selection_and_local_training_preserve_caller_rng_state(self) -> None:
        random.seed(808)
        python_state_before = random.getstate()
        select_client_ids([0, 1, 2], 2 / 3, 42, 1)
        self.assertEqual(random.getstate(), python_state_before)

        torch.manual_seed(909)
        torch_state_before = torch.get_rng_state().clone()
        train_client(
            client_id=0,
            dataset=make_dataset(),
            global_model=self.initial_model,
            batch_size=2,
            local_epochs=1,
            learning_rate=0.1,
            device=torch.device("cpu"),
            shuffle_seed=19,
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state_before))

    def test_local_training_starts_from_a_copy_and_does_not_change_global(self) -> None:
        result = train_client(
            client_id=0,
            dataset=make_dataset(),
            global_model=self.initial_model,
            batch_size=4,
            local_epochs=1,
            learning_rate=0.1,
            device=torch.device("cpu"),
            shuffle_seed=99,
        )

        for name, tensor in self.initial_model.state_dict().items():
            torch.testing.assert_close(tensor, self.initial_state[name])
            self.assertNotEqual(result.model_state[name].data_ptr(), tensor.data_ptr())
        self.assertTrue(
            any(
                not torch.equal(result.model_state[name], self.initial_state[name])
                for name in self.initial_state
            )
        )
        self.assertEqual(result.number_of_examples, 4)
        self.assertEqual(result.optimizer_steps, 1)

    def test_identical_clients_do_not_inherit_each_others_local_updates(self) -> None:
        settings = {
            "dataset": make_dataset(),
            "global_model": self.initial_model,
            "batch_size": 4,
            "local_epochs": 1,
            "learning_rate": 0.1,
            "device": torch.device("cpu"),
        }
        first = train_client(client_id=0, shuffle_seed=10, **settings)
        second = train_client(client_id=1, shuffle_seed=20, **settings)

        for name in self.initial_state:
            torch.testing.assert_close(first.model_state[name], second.model_state[name])
            torch.testing.assert_close(
                self.initial_model.state_dict()[name], self.initial_state[name]
            )

    def test_one_client_one_step_matches_centralized_sgd(self) -> None:
        dataset = make_dataset()
        server_model = copy.deepcopy(self.initial_model)
        server = FedAvgServer(
            global_model=server_model,
            client_datasets={0: dataset},
            client_fraction=1.0,
            local_epochs=1,
            batch_size=len(dataset),
            learning_rate=0.1,
            random_seed=77,
            device=torch.device("cpu"),
        )
        round_result = server.run_round(1)

        centralized_model = copy.deepcopy(self.initial_model)
        optimizer = torch.optim.SGD(centralized_model.parameters(), lr=0.1)
        features, labels = dataset.tensors
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(centralized_model(features), labels)
        loss.backward()
        optimizer.step()

        for name, tensor in centralized_model.state_dict().items():
            torch.testing.assert_close(server.global_state()[name], tensor)
        self.assertEqual(round_result.selected_client_ids, (0,))

    def test_two_client_round_is_reproducible_and_records_communication(self) -> None:
        datasets = {0: make_dataset(0.0), 1: make_dataset(0.25)}

        def build_server() -> FedAvgServer:
            model = TinyClassifier()
            model.load_state_dict(self.initial_state)
            return FedAvgServer(
                global_model=model,
                client_datasets=datasets,
                client_fraction=1.0,
                local_epochs=2,
                batch_size=2,
                learning_rate=0.05,
                random_seed=42,
                device=torch.device("cpu"),
            )

        first_server = build_server()
        second_server = build_server()
        first_round = first_server.run_round(1)
        second_round = second_server.run_round(1)

        for name, tensor in first_server.global_state().items():
            self.assertTrue(torch.equal(tensor, second_server.global_state()[name]))
            self.assertFalse(torch.equal(tensor, self.initial_state[name]))

        self.assertEqual(first_round.selected_client_ids, (0, 1))
        self.assertEqual(first_round.client_example_counts, {0: 4, 1: 4})
        self.assertEqual(first_round.client_optimizer_steps, {0: 4, 1: 4})
        self.assertEqual(first_round.selected_client_ids, second_round.selected_client_ids)
        self.assertEqual(first_round.client_example_counts, second_round.client_example_counts)
        self.assertAlmostEqual(
            first_round.weighted_mean_local_loss,
            second_round.weighted_mean_local_loss,
            places=12,
        )

        payload = model_state_size_bytes(self.initial_state)
        # TinyClassifier has 2x2 weights + 2 biases = 6 float32 values.
        self.assertEqual(payload, 6 * 4)
        self.assertEqual(first_round.model_payload_bytes, payload)
        self.assertEqual(first_round.download_bytes, 2 * payload)
        self.assertEqual(first_round.upload_bytes, 2 * payload)
        self.assertEqual(first_round.communication_cost_bytes, 4 * payload)

    def test_server_result_equals_explicit_client_training_then_manual_average(self) -> None:
        datasets = {
            0: make_dataset(),
            1: TensorDataset(*[tensor[:2].clone() for tensor in make_dataset(0.5).tensors]),
        }
        server_model = TinyClassifier()
        server_model.load_state_dict(self.initial_state)
        server = FedAvgServer(
            global_model=server_model,
            client_datasets=datasets,
            client_fraction=1.0,
            local_epochs=1,
            batch_size=4,
            learning_rate=0.1,
            random_seed=314,
            device=torch.device("cpu"),
        )

        manual_results = [
            train_client(
                client_id=client_id,
                dataset=datasets[client_id],
                global_model=self.initial_model,
                batch_size=4,
                local_epochs=1,
                learning_rate=0.1,
                device=torch.device("cpu"),
                shuffle_seed=client_training_seed(314, 1, client_id),
            )
            for client_id in (0, 1)
        ]
        expected_state = weighted_average_model_states(
            [result.model_state for result in manual_results], [4, 2]
        )

        round_result = server.run_round(1)
        self.assertEqual(round_result.client_example_counts, {0: 4, 1: 2})
        for name, expected_tensor in expected_state.items():
            torch.testing.assert_close(server.global_state()[name], expected_tensor)

    def test_two_round_progression_is_exactly_reproducible(self) -> None:
        datasets = {0: make_dataset(), 1: make_dataset(0.25)}

        def build_server() -> FedAvgServer:
            model = TinyClassifier()
            model.load_state_dict(self.initial_state)
            return FedAvgServer(
                global_model=model,
                client_datasets=datasets,
                client_fraction=1.0,
                local_epochs=1,
                batch_size=2,
                learning_rate=0.05,
                random_seed=2026,
                device=torch.device("cpu"),
            )

        first_server = build_server()
        second_server = build_server()
        first_round_one = first_server.run_round(1)
        first_state_after_round_one = first_server.global_state()
        first_round_two = first_server.run_round(2)
        first_state_after_round_two = first_server.global_state()

        second_round_one = second_server.run_round(1)
        second_state_after_round_one = second_server.global_state()
        second_round_two = second_server.run_round(2)
        second_state_after_round_two = second_server.global_state()

        for name in self.initial_state:
            self.assertTrue(
                torch.equal(
                    first_state_after_round_one[name],
                    second_state_after_round_one[name],
                )
            )
            self.assertTrue(
                torch.equal(
                    first_state_after_round_two[name],
                    second_state_after_round_two[name],
                )
            )
        self.assertTrue(
            any(
                not torch.equal(
                    first_state_after_round_one[name],
                    first_state_after_round_two[name],
                )
                for name in self.initial_state
            )
        )
        self.assertEqual(first_round_one, second_round_one)
        self.assertEqual(first_round_two, second_round_two)
        expected_total_communication = 2 * (2 * 24 + 2 * 24)
        actual_total_communication = (
            first_round_one.communication_cost_bytes
            + first_round_two.communication_cost_bytes
        )
        self.assertEqual(actual_total_communication, expected_total_communication)
        with self.assertRaisesRegex(ValueError, "must increase"):
            first_server.run_round(2)


if __name__ == "__main__":
    unittest.main()
