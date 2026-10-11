"""Toy checks for federated retrain without the target client (M_retrain^(-A))."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from algorithms.fedavg import clone_model_state, weighted_average_model_states
from algorithms.federated_retrain import (
    remaining_client_datasets,
    train_fedavg_excluding,
)
from clients.federated_client import train_client
from evaluation.efficiency import PhaseTimer
from server.fedavg_server import client_training_seed


def toy_dataset(number_of_examples: int, offset: float) -> TensorDataset:
    """Two-feature examples; the label is 1 when the first feature is positive."""
    features = torch.arange(number_of_examples * 2, dtype=torch.float32).reshape(-1, 2)
    features = features - offset
    return TensorDataset(features, (features[:, 0] > 0).long())


def toy_clients() -> dict[int, TensorDataset]:
    # n_0 = 2, n_1 = 6, n_2 = 4 (client 2 is the target in these tests).
    return {0: toy_dataset(2, 1.0), 1: toy_dataset(6, 5.0), 2: toy_dataset(4, 3.0)}


def initial_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(2, 2)


def run(client_datasets, exclude_clients, rounds=2, seed=7):
    """Train a fresh copy of the same initial model; return (state, rounds, timer)."""
    model = initial_model()
    timer = PhaseTimer()
    results = train_fedavg_excluding(
        model=model,
        client_datasets=client_datasets,
        exclude_clients=exclude_clients,
        number_of_rounds=rounds,
        client_fraction=1.0,
        local_epochs=1,
        batch_size=2,
        learning_rate=0.1,
        random_seed=seed,
        device=torch.device("cpu"),
        timer=timer,
        phase_name="training",
    )
    return clone_model_state(model.state_dict()), results, timer


def states_equal(first, second) -> bool:
    return all(torch.equal(first[name], second[name]) for name in first)


class RemainingClientDatasetsTests(unittest.TestCase):
    def test_target_is_removed_and_ids_are_kept(self) -> None:
        clients = toy_clients()
        remaining = remaining_client_datasets(clients, (2,))
        self.assertEqual(list(remaining), [0, 1])
        self.assertIs(remaining[0], clients[0])
        self.assertIs(remaining[1], clients[1])
        self.assertEqual(list(clients), [0, 1, 2])  # input is not mutated

    def test_excluding_nobody_keeps_everyone(self) -> None:
        self.assertEqual(list(remaining_client_datasets(toy_clients(), ())), [0, 1, 2])

    def test_unknown_client_and_excluding_everyone_are_errors(self) -> None:
        with self.assertRaises(ValueError):
            remaining_client_datasets(toy_clients(), (9,))
        with self.assertRaises(ValueError):
            remaining_client_datasets(toy_clients(), (0, 1, 2))


class FilterThenAverageArithmeticTests(unittest.TestCase):
    """Arithmetic only: filtering the states and then calling the UNCHANGED
    ``weighted_average_model_states`` gives weights n_k / (n - n_A).  The
    training function itself is tested in ``TrainFedAvgExcludingTests``."""

    def test_average_of_the_filtered_states_uses_renormalised_weights(self) -> None:
        # One scalar weight per client: w_0 = 0 (n=1), w_1 = 4 (n=3), w_2 = 100 (n=6).
        states = {0: {"w": torch.tensor([0.0])}, 1: {"w": torch.tensor([4.0])}, 2: {"w": torch.tensor([100.0])}}
        counts = {0: 1, 1: 3, 2: 6}
        # All clients: (1*0 + 3*4 + 6*100) / 10 = 61.2
        everyone = weighted_average_model_states(list(states.values()), list(counts.values()))
        self.assertAlmostEqual(everyone["w"].item(), 61.2, places=5)
        # Without client 2: (1*0 + 3*4) / 4 = 3.0   (weights 1/4 and 3/4, not 1/10 and 3/10)
        kept = remaining_client_datasets(states, (2,))
        retrain = weighted_average_model_states(
            list(kept.values()), [counts[client_id] for client_id in kept]
        )
        self.assertAlmostEqual(retrain["w"].item(), 3.0, places=6)


class TrainFedAvgExcludingTests(unittest.TestCase):
    def test_target_is_never_selected_and_its_examples_are_not_counted(self) -> None:
        _state, results, timer = run(toy_clients(), (2,))
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result.selected_client_ids, (0, 1))
            self.assertEqual(result.client_example_counts, {0: 2, 1: 6})
            self.assertEqual(result.total_selected_examples, 8)  # 2 + 6, not 12
        self.assertEqual(timer.count("training"), 2)  # one timed phase per round

    def test_one_round_equals_hand_weighted_average_of_remaining_clients(self) -> None:
        state, _results, _timer = run(toy_clients(), (2,), rounds=1, seed=7)
        clients = toy_clients()
        local = {
            client_id: train_client(
                client_id=client_id, dataset=clients[client_id], global_model=initial_model(),
                batch_size=2, local_epochs=1, learning_rate=0.1, device=torch.device("cpu"),
                shuffle_seed=client_training_seed(7, 1, client_id),
            ).model_state
            for client_id in (0, 1)
        }
        for name in state:
            # n_0 / (n_0 + n_1) = 2/8 = 0.25 and n_1 / (n_0 + n_1) = 6/8 = 0.75
            expected = 0.25 * local[0][name] + 0.75 * local[1][name]
            self.assertTrue(torch.allclose(state[name], expected, atol=1e-6))

    def test_changing_the_target_data_does_not_change_the_retrain(self) -> None:
        clients = toy_clients()
        changed = dict(clients)
        changed[2] = toy_dataset(4, -50.0)  # different features and labels for the target
        self.assertTrue(states_equal(run(clients, (2,))[0], run(changed, (2,))[0]))
        # ...but the original model (all clients) does depend on the target's data.
        self.assertFalse(states_equal(run(clients, ())[0], run(changed, ())[0]))

    def test_retrain_differs_from_original_and_is_reproducible(self) -> None:
        clients = toy_clients()
        retrain = run(clients, (2,))[0]
        self.assertFalse(states_equal(retrain, run(clients, ())[0]))
        self.assertTrue(states_equal(retrain, run(clients, (2,))[0]))  # same seed, same result

    def test_a_different_training_seed_gives_a_different_retrain(self) -> None:
        # Same initial weights and data; only the minibatch order differs.
        self.assertFalse(states_equal(run(toy_clients(), (2,), seed=7)[0], run(toy_clients(), (2,), seed=8)[0]))

    def test_excluding_client_zero_keeps_ids_and_per_client_seeds(self) -> None:
        # Exclude the LOWEST id: clients 1 and 2 must keep their ids, and so
        # the shuffle seeds derived from those ids.
        clients = toy_clients()
        self.assertEqual(list(remaining_client_datasets(clients, (0,))), [1, 2])
        state, results, _timer = run(clients, (0,), rounds=1, seed=7)
        self.assertEqual(results[0].selected_client_ids, (1, 2))
        self.assertEqual(results[0].client_example_counts, {1: 6, 2: 4})
        local = {
            client_id: train_client(
                client_id=client_id, dataset=clients[client_id], global_model=initial_model(),
                batch_size=2, local_epochs=1, learning_rate=0.1, device=torch.device("cpu"),
                shuffle_seed=client_training_seed(7, 1, client_id),  # ORIGINAL id: 1 or 2
            ).model_state
            for client_id in (1, 2)
        }
        for name in state:
            # n_1 / (n_1 + n_2) = 6/10 = 0.6 and n_2 / (n_1 + n_2) = 4/10 = 0.4
            expected = 0.6 * local[1][name] + 0.4 * local[2][name]
            self.assertTrue(torch.allclose(state[name], expected, atol=1e-6))
        # If the clients had been renumbered to 0 and 1, client 1 would have
        # used the seed of id 0, and the result would be different.
        renumbered = train_client(
            client_id=0, dataset=clients[1], global_model=initial_model(),
            batch_size=2, local_epochs=1, learning_rate=0.1, device=torch.device("cpu"),
            shuffle_seed=client_training_seed(7, 1, 0),
        ).model_state
        self.assertFalse(states_equal(renumbered, local[1]))


class RunnerInitialWeightsTests(unittest.TestCase):
    """The runner's ``train_and_evaluate`` on a toy federation of 28x28 images."""

    def test_both_runs_start_from_the_same_unchanged_initial_weights(self) -> None:
        from experiments.unlearning.month4_week14_mnist_federated_retrain import (
            make_initial_state,
            train_and_evaluate,
        )

        config = {
            "device": "cpu", "hidden_units": 4, "target_client": 2, "number_of_rounds": 1,
            "client_fraction": 1.0, "local_epochs": 1, "batch_size": 2,
            "learning_rate": 0.1, "evaluation_batch_size": 4,
        }
        torch.manual_seed(1)
        clients = {
            client_id: TensorDataset(torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 2, 3]))
            for client_id in (0, 1, 2)
        }
        test_data = TensorDataset(torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 2, 3]))
        initial_state = make_initial_state(config, seed=5)
        before = clone_model_state(initial_state)
        original = train_and_evaluate(config, clients, test_data, initial_state, (), 7, PhaseTimer(), "m")
        self.assertTrue(states_equal(initial_state, before))  # run 1 did not change w_0
        retrain = train_and_evaluate(config, clients, test_data, initial_state, (2,), 7, PhaseTimer(), "r")
        self.assertTrue(states_equal(initial_state, before))  # run 2 did not change w_0
        # Both runs trained (moved away from w_0) and ended in different places.
        self.assertFalse(states_equal(original["state"], before))
        self.assertFalse(states_equal(retrain["state"], before))
        self.assertFalse(states_equal(original["state"], retrain["state"]))
        self.assertEqual(retrain["record"]["rounds"][0]["selected_client_ids"], [0, 1])


if __name__ == "__main__":
    unittest.main()
