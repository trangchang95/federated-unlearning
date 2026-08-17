"""Small in-memory smoke check for the Week 8 comparison workflow."""

from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from algorithms.fedavg import clone_model_state, model_state_size_bytes
from experiments.iid.month2_week8_mnist_fedavg import (
    train_centralized_reference,
    train_fedavg,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class Week8ProtocolTests(unittest.TestCase):
    def test_centralized_and_fedavg_training_paths_run_from_same_state(self) -> None:
        torch.manual_seed(55)
        torch.use_deterministic_algorithms(True)
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [1.2, 0.9],
                [-0.8, 1.1],
            ]
        )
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        dataset = TensorDataset(features, labels)
        validation_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        clients = {
            0: Subset(dataset, [0, 1, 2, 3]),
            1: Subset(dataset, [4, 5, 6, 7]),
        }
        config = {
            "batch_size": 4,
            "centralized_loader_seed": 100,
            "learning_rate": 0.1,
            "centralized_epochs": 1,
            "client_fraction": 1.0,
            "local_epochs": 1,
            "random_seed": 200,
            "number_of_rounds": 1,
        }
        initial = TinyClassifier()
        initial_state = clone_model_state(initial.state_dict())
        central_model = copy.deepcopy(initial)
        fedavg_model = copy.deepcopy(initial)

        central_state, central_history, _, central_steps = (
            train_centralized_reference(
                model=central_model,
                training_data=dataset,
                validation_loader=validation_loader,
                config=config,
                device=torch.device("cpu"),
            )
        )
        fedavg_state, fedavg_history, _, fedavg_steps, communication = train_fedavg(
            model=fedavg_model,
            client_training_data=clients,
            validation_loader=validation_loader,
            config=config,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(central_history), 2)
        self.assertEqual(len(fedavg_history), 2)
        self.assertEqual([row["epoch"] for row in central_history], [0, 1])
        self.assertEqual([row["round"] for row in fedavg_history], [0, 1])
        self.assertNotIn("round_number", fedavg_history[1])
        self.assertEqual(
            sum(row["selected_as_best_checkpoint"] for row in central_history), 1
        )
        self.assertEqual(
            sum(row["selected_as_best_checkpoint"] for row in fedavg_history), 1
        )
        self.assertEqual(fedavg_history[1]["selected_client_ids"], (0, 1))
        self.assertEqual(fedavg_history[1]["client_example_counts"], {0: 4, 1: 4})
        self.assertEqual(central_steps, 2)
        self.assertEqual(fedavg_steps, 2)
        self.assertEqual(
            communication, 2 * len(clients) * model_state_size_bytes(initial_state)
        )
        self.assertTrue(
            any(
                not torch.equal(central_state[name], initial_state[name])
                for name in initial_state
            )
        )
        self.assertTrue(
            any(
                not torch.equal(fedavg_state[name], initial_state[name])
                for name in initial_state
            )
        )


if __name__ == "__main__":
    unittest.main()
