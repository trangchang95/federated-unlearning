"""Transparent Flower 1.30 aggregation adapter for Month 2, Week 7.

The hand-written PyTorch FedAvg implementation remains the project's source
of truth. This module converts model states to Flower's NumPy representation,
calls Flower's published weighted aggregation helper, and converts the result
back. It exists to compare semantics, not to replace the Week 6 algorithm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import version
from numbers import Integral

import numpy as np
import torch
from flwr.server.strategy.aggregate import aggregate as flower_aggregate


EXPECTED_FLOWER_VERSION = "1.30.0"
SUPPORTED_FLOWER_DTYPES = {torch.float16, torch.float32, torch.float64}
ModelState = Mapping[str, torch.Tensor]


def installed_flower_version() -> str:
    """Return the installed framework version used by this adapter."""
    return version("flwr")


def _validate_compatibility_inputs(
    client_states: Sequence[ModelState], client_example_counts: Sequence[int]
) -> tuple[tuple[str, ...], list[int]]:
    if installed_flower_version() != EXPECTED_FLOWER_VERSION:
        raise RuntimeError(
            "This compatibility adapter was verified with Flower "
            f"{EXPECTED_FLOWER_VERSION}, found {installed_flower_version()}."
        )
    if not client_states:
        raise ValueError("Flower comparison requires at least one client state.")
    if len(client_states) != len(client_example_counts):
        raise ValueError(
            "client_states and client_example_counts must have the same length."
        )
    if any(not isinstance(state, Mapping) for state in client_states):
        raise TypeError("Every client state must be a mapping of tensor names.")

    counts: list[int] = []
    for index, count in enumerate(client_example_counts):
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(f"Client {index} example count must be an integer.")
        if int(count) <= 0:
            raise ValueError(f"Client {index} example count must be positive.")
        counts.append(int(count))

    keys = tuple(client_states[0].keys())
    if not keys:
        raise ValueError("A model state cannot be empty.")
    key_set = set(keys)
    reference_state = client_states[0]
    for client_index, state in enumerate(client_states):
        if set(state) != key_set:
            raise ValueError(f"Client {client_index} has different state keys.")
        for name in keys:
            tensor = state[name]
            reference = reference_state[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"State entry {name!r} is not a tensor.")
            if tensor.dtype not in SUPPORTED_FLOWER_DTYPES:
                raise ValueError(
                    "The Week 7 Flower/NumPy comparison supports float16, "
                    f"float32, and float64 only; {name!r} has dtype {tensor.dtype}."
                )
            if (
                tensor.shape != reference.shape
                or tensor.dtype != reference.dtype
                or tensor.device != reference.device
            ):
                raise ValueError(
                    f"Client {client_index} tensor {name!r} is incompatible "
                    "with the reference state."
                )
            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"Client {client_index} tensor {name!r} is non-finite."
                )
    return keys, counts


def flower_weighted_average_model_states(
    client_states: Sequence[ModelState], client_example_counts: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Aggregate the same client states through Flower's NumPy helper.

    Flower 1.30's helper multiplies each layer by its client's ``num_examples``,
    sums corresponding layers, and divides by the total examples. Inputs are
    copied before conversion so this comparison cannot mutate PyTorch states.
    """
    keys, counts = _validate_compatibility_inputs(
        client_states, client_example_counts
    )
    flower_inputs: list[tuple[list[np.ndarray], int]] = []
    for state, count in zip(client_states, counts):
        arrays = [state[name].detach().cpu().numpy().copy() for name in keys]
        flower_inputs.append((arrays, count))

    averaged_arrays = flower_aggregate(flower_inputs)
    if len(averaged_arrays) != len(keys):
        raise RuntimeError("Flower returned an unexpected number of model tensors.")

    reference_state = client_states[0]
    return {
        name: torch.from_numpy(np.array(array, copy=True)).to(
            device=reference_state[name].device,
            dtype=reference_state[name].dtype,
        )
        for name, array in zip(keys, averaged_arrays)
    }
