"""Hand-written Federated Averaging primitives for Month 2, Week 6.

FedAvg combines locally trained model states according to the number of
training examples represented by each selected client.  This module contains
only that mathematical operation and small state-dict helpers.  Client
training and round orchestration live in ``clients/`` and ``server/`` so the
three responsibilities remain visible to a beginner.

No Federated Learning framework is used here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import torch


ModelState = Mapping[str, torch.Tensor]


def clone_model_state(state: ModelState) -> dict[str, torch.Tensor]:
    """Return detached tensor copies, with no storage shared with ``state``."""
    cloned: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {name!r} is not a torch.Tensor.")
        cloned[name] = tensor.detach().clone()
    return cloned


def model_state_size_bytes(state: ModelState) -> int:
    """Count the raw tensor payload bytes in a model state.

    This is useful for a later dense-tensor communication-cost estimate.  It
    does not represent sparse/custom serialization and does not include
    protocol headers, retries, or network latency.
    """
    total = 0
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {name!r} is not a torch.Tensor.")
        total += tensor.numel() * tensor.element_size()
    return total


def _validate_inputs(
    client_states: Sequence[ModelState], client_example_counts: Sequence[int]
) -> tuple[tuple[str, ...], list[int]]:
    if not client_states:
        raise ValueError("FedAvg requires at least one selected client state.")
    if len(client_states) != len(client_example_counts):
        raise ValueError(
            "client_states and client_example_counts must have the same length."
        )

    counts: list[int] = []
    for index, count in enumerate(client_example_counts):
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(
                f"Client {index} example count must be a positive integer, got {count!r}."
            )
        integer_count = int(count)
        if integer_count <= 0:
            raise ValueError(
                f"Client {index} example count must be positive, got {integer_count}."
            )
        counts.append(integer_count)

    for client_index, state in enumerate(client_states):
        if not isinstance(state, Mapping):
            raise TypeError(
                f"Client {client_index} model state must be a mapping of tensor names."
            )

    reference_state = client_states[0]
    reference_keys = tuple(reference_state.keys())
    if not reference_keys:
        raise ValueError("FedAvg cannot aggregate an empty model state.")

    reference_key_set = set(reference_keys)
    for client_index, state in enumerate(client_states):
        if set(state.keys()) != reference_key_set:
            missing = sorted(reference_key_set - set(state.keys()))
            unexpected = sorted(set(state.keys()) - reference_key_set)
            raise ValueError(
                f"Client {client_index} state keys do not match; "
                f"missing={missing}, unexpected={unexpected}."
            )

        for name in reference_keys:
            reference_tensor = reference_state[name]
            client_tensor = state[name]
            if not isinstance(reference_tensor, torch.Tensor):
                raise TypeError(f"State entry {name!r} is not a torch.Tensor.")
            if not isinstance(client_tensor, torch.Tensor):
                raise TypeError(
                    f"Client {client_index} state entry {name!r} is not a torch.Tensor."
                )
            if client_tensor.shape != reference_tensor.shape:
                raise ValueError(
                    f"Client {client_index} tensor {name!r} has shape "
                    f"{tuple(client_tensor.shape)}, expected {tuple(reference_tensor.shape)}."
                )
            if client_tensor.dtype != reference_tensor.dtype:
                raise ValueError(
                    f"Client {client_index} tensor {name!r} has dtype "
                    f"{client_tensor.dtype}, expected {reference_tensor.dtype}."
                )
            if client_tensor.device != reference_tensor.device:
                raise ValueError(
                    f"Client {client_index} tensor {name!r} is on "
                    f"{client_tensor.device}, expected {reference_tensor.device}."
                )
            if (
                client_tensor.is_floating_point() or client_tensor.is_complex()
            ) and not torch.isfinite(client_tensor).all().item():
                raise ValueError(
                    f"Client {client_index} tensor {name!r} contains NaN or infinity."
                )

    return reference_keys, counts


def weighted_average_model_states(
    client_states: Sequence[ModelState], client_example_counts: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Compute sample-count-weighted FedAvg without mutating any input.

    For selected clients ``k`` with ``n_k`` examples and returned model state
    ``w_k``, every floating-point tensor is combined as::

        sum_k (n_k / sum_j n_j) * w_k

    Model parameters are floating point.  A state dict can also contain an
    integer/bool buffer (for example, a BatchNorm counter).  A fractional
    weighted mean has no faithful representation in such a dtype, so this
    beginner implementation preserves a non-floating buffer only when every
    client returned the same value; otherwise it raises an explicit error.
    """
    state_keys, counts = _validate_inputs(client_states, client_example_counts)
    total_examples = sum(counts)
    reference_state = client_states[0]
    averaged_state: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for name in state_keys:
            reference_tensor = reference_state[name]
            if reference_tensor.is_floating_point() or reference_tensor.is_complex():
                # Accumulate half-precision model states in float32 to avoid
                # needless rounding at every client addition. The result is
                # converted back to the model's original dtype afterward.
                accumulator_dtype = (
                    torch.float32
                    if reference_tensor.dtype in {torch.float16, torch.bfloat16}
                    else reference_tensor.dtype
                )
                averaged_tensor = torch.zeros_like(
                    reference_tensor, dtype=accumulator_dtype
                )
                for state, count in zip(client_states, counts):
                    averaged_tensor.add_(
                        state[name].to(dtype=accumulator_dtype),
                        alpha=count / total_examples,
                    )
                averaged_state[name] = averaged_tensor.to(
                    dtype=reference_tensor.dtype
                )
                continue

            if not all(
                torch.equal(state[name], reference_tensor)
                for state in client_states[1:]
            ):
                raise ValueError(
                    f"Non-floating state entry {name!r} differs between clients; "
                    "its weighted mean would be ambiguous."
                )
            averaged_state[name] = reference_tensor.detach().clone()

    return averaged_state
