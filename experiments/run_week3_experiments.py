"""Run Week 3 adversarial federated learning experiments.

Scenarios:
    A. FedAvg without attack
    B. FedAvg with label-flipping malicious clients
    C. Krum with label-flipping malicious clients
    D. Coordinate-wise median with label-flipping malicious clients
    E. Trimmed mean with label-flipping malicious clients
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from fl_utils import (
    aggregate_with_timing,
    add_gaussian_noise_to_state,
    amplify_model_update,
    simulate_malicious_clients,
)


TARGET = "Cardiomegaly"


class MLPImageClassifier(nn.Module):
    """Small Week 2-compatible classifier for flattened CheXpert image features."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pixel_features(raw_dir: Path, paths: pd.Series, image_size: int) -> np.ndarray:
    features = []
    for rel_path in paths:
        image_path = raw_dir / rel_path
        image = Image.open(image_path).convert("L").resize((image_size, image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        features.append(arr.ravel())
    return np.vstack(features)


def prepare_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_dir = ROOT / "data" / "raw"
    csv_path = raw_dir / "train.csv"
    df = pd.read_csv(csv_path)
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    df[TARGET] = df[TARGET].replace(-1, 0).fillna(0).astype(int)

    sample_size = min(args.sample_size, len(df))
    sample = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)

    x = load_pixel_features(raw_dir, sample["Path"], args.image_size)
    y = sample[TARGET].to_numpy().astype(np.float32)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
    x_test = scaler.transform(x[test_idx]).astype(np.float32)
    y_train = y[train_idx].astype(np.float32)
    y_test = y[test_idx].astype(np.float32)
    return x_train, x_test, y_train, y_test


def make_clients(x_train: np.ndarray, y_train: np.ndarray, num_clients: int, seed: int) -> list[tuple[int, TensorDataset]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_train))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    clients = []
    for client_id, idx in enumerate(splits):
        x_client = torch.from_numpy(x_train[idx])
        y_client = torch.from_numpy(y_train[idx])
        clients.append((client_id, TensorDataset(x_client, y_client)))
    return clients


def local_train(
    global_model: nn.Module,
    client_dataset: TensorDataset,
    input_dim: int,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    local_model = MLPImageClassifier(input_dim).to(device)
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()

    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
    running_loss = 0.0
    seen = 0

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = local_model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
            seen += len(yb)

    state = {key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()}
    return state, len(client_dataset), running_loss / max(seen, 1)


def evaluate(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        xb = torch.from_numpy(x_test).to(device)
        yb = torch.from_numpy(y_test).to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb).item()
        proba = torch.sigmoid(logits).cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    return {
        "test_loss": loss,
        "test_accuracy": accuracy_score(y_test, pred),
        "predicted_positive_rate": float(pred.mean()),
    }


def run_scenario(
    scenario_name: str,
    aggregation: str,
    attack_enabled: bool,
    base_clients: list[tuple[int, TensorDataset]],
    malicious_client_ids: set[int],
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    set_seed(args.seed)
    input_dim = base_clients[0][1].tensors[0].shape[1]
    global_model = MLPImageClassifier(input_dim).to(device)

    if attack_enabled:
        clients = simulate_malicious_clients(base_clients, malicious_client_ids, attack_type="label_flip")
    else:
        clients = [(client_id, dataset, False) for client_id, dataset in base_clients]

    rows = []
    for round_id in range(1, args.rounds + 1):
        client_states = []
        local_losses = []

        for client_id, client_dataset, is_malicious in clients:
            global_state_before_client = {
                key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()
            }
            state, n_samples, local_loss = local_train(
                global_model,
                client_dataset,
                input_dim,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
            )
            if is_malicious and args.attack_strength != 1.0:
                state = amplify_model_update(
                    state,
                    global_state_before_client,
                    attack_strength=args.attack_strength,
                )
            if is_malicious and args.model_noise_std > 0:
                state = add_gaussian_noise_to_state(
                    state,
                    std=args.model_noise_std,
                    scale=args.model_noise_scale,
                    seed=args.seed + round_id + client_id,
                )
            client_states.append((state, n_samples))
            local_losses.append(local_loss)

        aggregated_state, aggregation_time = aggregate_with_timing(
            client_states,
            aggregation,
            num_malicious=len(malicious_client_ids),
            trim_ratio=args.trim_ratio,
        )
        global_model.load_state_dict(aggregated_state)
        metrics = evaluate(global_model, x_test, y_test, device)

        rows.append(
            {
                "scenario": scenario_name,
                "aggregation": aggregation,
                "attack_enabled": attack_enabled,
                "round": round_id,
                "num_clients": args.num_clients,
                "num_malicious": len(malicious_client_ids) if attack_enabled else 0,
                "attack_strength": args.attack_strength if attack_enabled else 0.0,
                "local_train_loss": float(np.mean(local_losses)),
                "aggregation_time_sec": aggregation_time,
                **metrics,
            }
        )

        print(
            f"{scenario_name} | round {round_id:02d}/{args.rounds} | "
            f"acc={metrics['test_accuracy']:.4f} | agg_time={aggregation_time:.4f}s"
        )

    return rows


def add_attack_success_rate(results: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        results[results["scenario"] == "A_FedAvg_No_Attack"][["round", "test_accuracy"]]
        .rename(columns={"test_accuracy": "clean_fedavg_accuracy"})
    )
    merged = results.merge(baseline, on="round", how="left")
    merged["accuracy_drop"] = merged["clean_fedavg_accuracy"] - merged["test_accuracy"]
    merged["attack_success_rate"] = merged["accuracy_drop"].clip(lower=0)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 3 adversarial FL experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-malicious", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument(
        "--attack-strength",
        type=float,
        default=8.0,
        help=(
            "Scale factor applied to malicious model updates after label flipping. "
            "Use 1.0 for plain label flipping without update amplification."
        ),
    )
    parser.add_argument("--model-noise-std", type=float, default=0.0)
    parser.add_argument("--model-noise-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    output_dir = ROOT / "experiments"
    output_dir.mkdir(exist_ok=True)

    x_train, x_test, y_train, y_test = prepare_data(args)
    base_clients = make_clients(x_train, y_train, args.num_clients, args.seed)
    malicious_client_ids = set(range(args.num_malicious))

    scenarios = [
        ("A_FedAvg_No_Attack", "fedavg", False),
        ("B_FedAvg_LabelFlip", "fedavg", True),
        ("C_Krum_LabelFlip", "krum", True),
        ("D_Median_LabelFlip", "coordinate_median", True),
        ("E_TrimmedMean_LabelFlip", "trimmed_mean", True),
    ]

    all_rows = []
    for scenario_name, aggregation, attack_enabled in scenarios:
        all_rows.extend(
            run_scenario(
                scenario_name,
                aggregation,
                attack_enabled,
                base_clients,
                malicious_client_ids,
                x_test,
                y_test,
                args,
                device,
            )
        )

    results = add_attack_success_rate(pd.DataFrame(all_rows))
    csv_path = output_dir / "week3_results.csv"
    json_path = output_dir / "week3_results.json"
    results.to_csv(csv_path, index=False)
    results.to_json(json_path, orient="records", indent=2)

    config_path = output_dir / "week3_experiment_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"\nSaved results to {csv_path}")
    print(f"Saved JSON results to {json_path}")
    print(f"Saved run config to {config_path}")


if __name__ == "__main__":
    main()
