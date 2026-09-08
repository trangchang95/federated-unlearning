"""Deterministic synthetic checks for the hand-written FedProx addition."""

from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from clients.federated_client import train_client
from clients.fedprox_client import train_client_prox
from server.fedavg_server import FedAvgServer
from server.fedprox_server import FedProxServer


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


def state_distance(first, second) -> float:
    return sum(
        (first[name] - second[name]).pow(2).sum().item() for name in first
    ) ** 0.5


class FedProxClientTests(unittest.TestCase):
    def test_mu_zero_matches_plain_fedavg_local_training_exactly(self) -> None:
        torch.manual_seed(0)
        global_model = TinyClassifier()
        dataset = make_dataset()

        fedavg_result = train_client(
            client_id=0,
            dataset=dataset,
            global_model=copy.deepcopy(global_model),
            batch_size=2,
            local_epochs=3,
            learning_rate=0.1,
            device=torch.device("cpu"),
            shuffle_seed=7,
        )
        prox_result = train_client_prox(
            client_id=0,
            dataset=dataset,
            global_model=copy.deepcopy(global_model),
            batch_size=2,
            local_epochs=3,
            learning_rate=0.1,
            mu=0.0,
            device=torch.device("cpu"),
            shuffle_seed=7,
        )

        for name in fedavg_result.model_state:
            self.assertTrue(
                torch.allclose(fedavg_result.model_state[name], prox_result.model_state[name]),
                f"mu=0 FedProx must match plain FedAvg exactly for {name!r}.",
            )
        self.assertAlmostEqual(fedavg_result.mean_loss, prox_result.mean_loss, places=6)
        self.assertEqual(fedavg_result.optimizer_steps, prox_result.optimizer_steps)

    def test_larger_mu_keeps_the_local_model_closer_to_the_global_model(self) -> None:
        torch.manual_seed(1)
        global_model = TinyClassifier()
        global_state = {
            name: tensor.clone() for name, tensor in global_model.state_dict().items()
        }
        dataset = make_dataset(offset=2.0)  # a bigger offset gives local training more to chase

        low_mu = train_client_prox(
            client_id=0, dataset=dataset, global_model=copy.deepcopy(global_model),
            batch_size=2, local_epochs=5, learning_rate=0.2, mu=0.0,
            device=torch.device("cpu"), shuffle_seed=3,
        )
        # mu=50 here would make lr*mu=10, far outside SGD's stable range for
        # a quadratic proximal term (stable requires roughly 0 < lr*mu < 2),
        # causing divergence rather than a tighter pull to the global model.
        # mu=3 (lr*mu=0.6) demonstrates the same pull-back effect stably.
        high_mu = train_client_prox(
            client_id=0, dataset=dataset, global_model=copy.deepcopy(global_model),
            batch_size=2, local_epochs=5, learning_rate=0.2, mu=3.0,
            device=torch.device("cpu"), shuffle_seed=3,
        )

        self.assertLess(
            state_distance(high_mu.model_state, global_state),
            state_distance(low_mu.model_state, global_state),
        )

    def test_invalid_mu_is_rejected(self) -> None:
        dataset = make_dataset()
        model = TinyClassifier()
        for bad_mu in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                train_client_prox(
                    client_id=0, dataset=dataset, global_model=copy.deepcopy(model),
                    batch_size=2, local_epochs=1, learning_rate=0.1, mu=bad_mu,
                    device=torch.device("cpu"), shuffle_seed=1,
                )


class FedProxServerTests(unittest.TestCase):
    def test_mu_zero_round_matches_fedavg_server_round(self) -> None:
        client_datasets = {0: make_dataset(), 1: make_dataset(offset=1.0)}

        torch.manual_seed(11)
        fedavg_model = TinyClassifier()
        fedavg_server = FedAvgServer(
            global_model=fedavg_model,
            client_datasets=client_datasets,
            client_fraction=1.0,
            local_epochs=2,
            batch_size=2,
            learning_rate=0.15,
            random_seed=42,
            device=torch.device("cpu"),
        )
        fedavg_round = fedavg_server.run_round(1)

        torch.manual_seed(11)
        prox_model = TinyClassifier()
        prox_server = FedProxServer(
            global_model=prox_model,
            client_datasets=client_datasets,
            client_fraction=1.0,
            local_epochs=2,
            batch_size=2,
            learning_rate=0.15,
            mu=0.0,
            random_seed=42,
            device=torch.device("cpu"),
        )
        prox_round = prox_server.run_round(1)

        self.assertEqual(fedavg_round.selected_client_ids, prox_round.selected_client_ids)
        self.assertEqual(fedavg_round.client_example_counts, prox_round.client_example_counts)
        self.assertAlmostEqual(
            fedavg_round.weighted_mean_local_loss, prox_round.weighted_mean_local_loss, places=6
        )
        for name in fedavg_server.global_state():
            self.assertTrue(
                torch.allclose(
                    fedavg_server.global_state()[name], prox_server.global_state()[name]
                )
            )

    def test_invalid_mu_is_rejected_at_construction(self) -> None:
        client_datasets = {0: make_dataset()}
        with self.assertRaisesRegex(ValueError, "mu must be"):
            FedProxServer(
                global_model=TinyClassifier(),
                client_datasets=client_datasets,
                client_fraction=1.0,
                local_epochs=1,
                batch_size=2,
                learning_rate=0.1,
                mu=-0.5,
                random_seed=1,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
