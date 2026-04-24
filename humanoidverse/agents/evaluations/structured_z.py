from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LinearProbeResult:
    train_mse: float
    test_mse: float
    num_train: int
    num_test: int


def extract_structured_latents(model, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not model.has_structured_z:
        raise ValueError("Structured-z evaluation requires a model with z_hand_dim > 0.")

    z = model.backward_map(obs)
    z_hand = model.extract_z_hand(z, decode=True)
    z_body = model.extract_z_body(z)
    hand_target = model.extract_hand_pos(obs)
    return z_hand, z_body, hand_target


def build_hand_intervention_contexts(
    model,
    z_body: torch.Tensor,
    hand_targets: torch.Tensor,
) -> torch.Tensor:
    if not model.has_structured_z:
        raise ValueError("Hand interventions require a model with structured z.")
    if hand_targets.ndim != 2 or hand_targets.shape[-1] != model.z_hand_dim:
        raise ValueError(
            f"hand_targets must have shape [N, {model.z_hand_dim}], got {tuple(hand_targets.shape)}."
        )

    if z_body.ndim == 1:
        z_body = z_body.unsqueeze(0)
    z_body = z_body.expand(hand_targets.shape[0], -1)
    return model.merge_z(hand_targets, z_body)


def ridge_linear_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    train_fraction: float = 0.8,
    ridge: float = 1e-4,
) -> LinearProbeResult:
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must both be rank-2 tensors.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of rows.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")

    num_samples = features.shape[0]
    num_train = max(1, int(num_samples * train_fraction))
    num_test = max(1, num_samples - num_train)
    if num_train + num_test > num_samples:
        num_train = num_samples - num_test

    perm = torch.randperm(num_samples, device=features.device)
    features = features[perm]
    targets = targets[perm]

    x_train = torch.cat([features[:num_train], torch.ones(num_train, 1, device=features.device)], dim=-1)
    y_train = targets[:num_train]
    x_test = torch.cat([features[num_train : num_train + num_test], torch.ones(num_test, 1, device=features.device)], dim=-1)
    y_test = targets[num_train : num_train + num_test]

    eye = torch.eye(x_train.shape[1], device=features.device, dtype=features.dtype)
    weights = torch.linalg.solve(x_train.T @ x_train + ridge * eye, x_train.T @ y_train)

    train_pred = x_train @ weights
    test_pred = x_test @ weights
    train_mse = torch.mean((train_pred - y_train) ** 2).item()
    test_mse = torch.mean((test_pred - y_test) ** 2).item()
    return LinearProbeResult(train_mse=train_mse, test_mse=test_mse, num_train=num_train, num_test=num_test)


def evaluate_body_leakage(
    model,
    obs: dict[str, torch.Tensor],
    train_fraction: float = 0.8,
    ridge: float = 1e-4,
) -> LinearProbeResult:
    _, z_body, hand_target = extract_structured_latents(model, obs)
    return ridge_linear_probe(z_body, hand_target, train_fraction=train_fraction, ridge=ridge)


def compare_metric_tables(
    baseline_metrics: dict[str, float],
    structured_metrics: dict[str, float],
) -> dict[str, dict[str, float]]:
    shared_keys = sorted(set(baseline_metrics) & set(structured_metrics))
    comparison = {}
    for key in shared_keys:
        baseline = float(baseline_metrics[key])
        structured = float(structured_metrics[key])
        comparison[key] = {
            "baseline": baseline,
            "structured": structured,
            "delta": structured - baseline,
        }
    return comparison
