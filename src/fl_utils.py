"""Federated learning utilities for Week 3 adversarial FL experiments."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import torch
from torch.utils.data import TensorDataset

StateDict = Mapping[str, torch.Tensor]


def clone_state_dict(state: StateDict) -> OrderedDict[str, torch.Tensor]:
    """Return a detached CPU clone of a PyTorch model state dictionary."""
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in state.items())


def flatten_state_dict(state: StateDict) -> torch.Tensor:
    """Flatten all floating-point tensors from a model state dict into one vector."""
    return torch.cat([value.detach().cpu().reshape(-1).float() for value in state.values()])


def unflatten_state_dict(vector: torch.Tensor, reference_state: StateDict) -> OrderedDict[str, torch.Tensor]:
    """Rebuild a state dict from a flat vector using a reference state dict's shapes."""
    rebuilt = OrderedDict()
    offset = 0
    vector = vector.detach().cpu()
    for key, ref_value in reference_state.items():
        numel = ref_value.numel()
        rebuilt[key] = vector[offset : offset + numel].view_as(ref_value).to(dtype=ref_value.dtype).clone()
        offset += numel
    if offset != vector.numel():
        raise ValueError("Flat vector length does not match the reference state dictionary.")
    return rebuilt


def state_dict_to_update(local_state: StateDict, global_state: StateDict) -> OrderedDict[str, torch.Tensor]:
    """Convert local model parameters into an update relative to the global model."""
    return OrderedDict((key, local_state[key].detach().cpu() - global_state[key].detach().cpu()) for key in global_state)


def update_to_state_dict(update: StateDict, global_state: StateDict) -> OrderedDict[str, torch.Tensor]:
    """Apply a model update to the global model parameters."""
    return OrderedDict((key, global_state[key].detach().cpu() + update[key].detach().cpu()) for key in global_state)


def _normalise_client_states(
    client_states: Sequence[StateDict] | Sequence[tuple[StateDict, int]],
) -> tuple[list[StateDict], list[int]]:
    states: list[StateDict] = []
    weights: list[int] = []
    for item in client_states:
        if isinstance(item, tuple):
            state, n_samples = item
            states.append(state)
            weights.append(int(n_samples))
        else:
            states.append(item)
            weights.append(1)
    if not states:
        raise ValueError("At least one client state is required for aggregation.")
    return states, weights


def fedavg(client_states: Sequence[StateDict] | Sequence[tuple[StateDict, int]]) -> OrderedDict[str, torch.Tensor]:
    """Weighted Federated Averaging aggregation."""
    states, weights = _normalise_client_states(client_states)
    total_weight = float(sum(weights))
    averaged = OrderedDict()
    for key in states[0].keys():
        averaged[key] = sum(state[key].detach().cpu() * (weight / total_weight) for state, weight in zip(states, weights))
    return averaged


def coordinate_median(client_states: Sequence[StateDict] | Sequence[tuple[StateDict, int]]) -> OrderedDict[str, torch.Tensor]:
    """Coordinate-wise median aggregation over flattened model parameters."""
    states, _ = _normalise_client_states(client_states)
    stacked = torch.stack([flatten_state_dict(state) for state in states], dim=0)
    median_vector = torch.median(stacked, dim=0).values
    return unflatten_state_dict(median_vector, states[0])


def trimmed_mean(
    client_states: Sequence[StateDict] | Sequence[tuple[StateDict, int]],
    trim_ratio: float = 0.2,
) -> OrderedDict[str, torch.Tensor]:
    """Coordinate-wise trimmed mean aggregation.

    trim_ratio is the fraction removed from each tail. With 10 clients and
    trim_ratio=0.2, the 2 smallest and 2 largest values are discarded per
    parameter coordinate before averaging.
    """
    if not 0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5).")
    states, _ = _normalise_client_states(client_states)
    stacked = torch.stack([flatten_state_dict(state) for state in states], dim=0)
    n_clients = stacked.shape[0]
    trim_k = int(trim_ratio * n_clients)
    sorted_values = torch.sort(stacked, dim=0).values
    if trim_k > 0:
        sorted_values = sorted_values[trim_k : n_clients - trim_k]
    mean_vector = sorted_values.mean(dim=0)
    return unflatten_state_dict(mean_vector, states[0])


def krum(
    client_states: Sequence[StateDict] | Sequence[tuple[StateDict, int]],
    num_malicious: int,
) -> OrderedDict[str, torch.Tensor]:
    """Krum aggregation: select the update closest to its nearest neighbours.

    Krum requires n >= 2f + 3, where n is the number of clients and f is the
    assumed number of Byzantine clients.
    """
    states, _ = _normalise_client_states(client_states)
    n_clients = len(states)
    if n_clients < 2 * num_malicious + 3:
        raise ValueError(f"Krum requires n >= 2f + 3, got n={n_clients}, f={num_malicious}.")

    stacked = torch.stack([flatten_state_dict(state) for state in states], dim=0)
    distances = torch.cdist(stacked, stacked, p=2).pow(2)
    neighbour_count = n_clients - num_malicious - 2
    scores = []
    for i in range(n_clients):
        nearest = torch.topk(distances[i], k=neighbour_count + 1, largest=False).values[1:]
        scores.append(nearest.sum())
    selected_idx = int(torch.argmin(torch.stack(scores)).item())
    return clone_state_dict(states[selected_idx])


def aggregate_with_timing(
    client_states: Sequence[StateDict] | Sequence[tuple[StateDict, int]],
    method: str,
    *,
    num_malicious: int = 0,
    trim_ratio: float = 0.2,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    """Aggregate client states and return (aggregated_state, elapsed_seconds)."""
    start = time.perf_counter()
    method_key = method.lower().replace("-", "_")
    if method_key == "fedavg":
        aggregated = fedavg(client_states)
    elif method_key in {"krum", "multi_krum"}:
        aggregated = krum(client_states, num_malicious=num_malicious)
    elif method_key in {"median", "coordinate_median", "coordinatewise_median"}:
        aggregated = coordinate_median(client_states)
    elif method_key in {"trimmed_mean", "trimmedmean"}:
        aggregated = trimmed_mean(client_states, trim_ratio=trim_ratio)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    elapsed = time.perf_counter() - start
    return aggregated, elapsed


def make_label_flipped_dataset(dataset: TensorDataset, binary: bool = True) -> TensorDataset:
    """Return a copy of a TensorDataset with poisoned labels.

    For binary CheXpert labels, 0 becomes 1 and 1 becomes 0.
    """
    tensors = [tensor.detach().clone() for tensor in dataset.tensors]
    labels = tensors[-1]
    if binary:
        tensors[-1] = 1.0 - labels.float()
    else:
        tensors[-1] = labels[torch.randperm(len(labels))]
    return TensorDataset(*tensors)


def add_gaussian_noise_to_state(
    local_state: StateDict,
    std: float = 0.05,
    scale: float = 1.0,
    seed: int | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Model-poisoning helper that adds Gaussian noise to a client update/state."""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    noisy = OrderedDict()
    for key, value in local_state.items():
        if torch.is_floating_point(value):
            noise = torch.randn(value.shape, generator=generator, dtype=value.dtype) * std * scale
            noisy[key] = value.detach().cpu() + noise
        else:
            noisy[key] = value.detach().cpu().clone()
    return noisy


def amplify_model_update(
    local_state: StateDict,
    global_state: StateDict,
    attack_strength: float,
) -> OrderedDict[str, torch.Tensor]:
    """Amplify a malicious client's model delta before server aggregation.

    This simulates a stronger poisoning attacker: the client trains on poisoned
    labels, then scales its update so FedAvg is pulled toward the poisoned
    objective. Robust aggregators should treat these states as outliers.
    """
    if attack_strength < 0:
        raise ValueError("attack_strength must be non-negative.")
    amplified = OrderedDict()
    for key, local_value in local_state.items():
        global_value = global_state[key].detach().cpu()
        local_value = local_value.detach().cpu()
        if torch.is_floating_point(local_value):
            amplified[key] = global_value + attack_strength * (local_value - global_value)
        else:
            amplified[key] = local_value.clone()
    return amplified


def simulate_malicious_clients(
    clients: Iterable[tuple[int, TensorDataset]],
    malicious_client_ids: set[int],
    attack_type: str = "label_flip",
) -> list[tuple[int, TensorDataset, bool]]:
    """Mark clients as malicious and apply the requested data-poisoning attack."""
    poisoned_clients = []
    for client_id, dataset in clients:
        is_malicious = client_id in malicious_client_ids
        if is_malicious and attack_type == "label_flip":
            dataset = make_label_flipped_dataset(dataset)
        elif is_malicious and attack_type not in {"none", "label_flip"}:
            raise ValueError(f"Unsupported data attack type: {attack_type}")
        poisoned_clients.append((client_id, dataset, is_malicious))
    return poisoned_clients


def copy_model_state_to_device(state: StateDict, device: torch.device | str) -> OrderedDict[str, torch.Tensor]:
    """Move a state dict to the target device before loading into a model."""
    return OrderedDict((key, value.to(device)) for key, value in state.items())
