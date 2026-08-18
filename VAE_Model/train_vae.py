"""Train the two-dimensional VAE used for PfHT1 pathway fingerprints."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PathVAE(nn.Module):
    """900-256-64-2-64-256-900 VAE for 30 x 30 path fingerprints."""

    def __init__(self, input_dimension: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, 2)
        self.logvar = nn.Linear(64, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dimension),
        )

    def encode(self, values):
        hidden = self.encoder(values)
        return self.mu(hidden), self.logvar(hidden)

    def forward(self, values):
        mu, logvar = self.encode(values)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        log_probability = torch.log_softmax(self.decoder(latent), dim=1)
        return log_probability, mu, logvar


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def loss_terms(values, log_probability, mu, logvar):
    reconstruction = -torch.sum(values * log_probability, dim=1)
    kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1)
    return reconstruction, kl


def evaluate(model, loader, beta):
    model.eval()
    reconstruction_total = kl_total = 0.0
    count = 0
    with torch.no_grad():
        for (batch,) in loader:
            log_probability, mu, logvar = model(batch)
            reconstruction, kl = loss_terms(batch, log_probability, mu, logvar)
            reconstruction_total += float(reconstruction.sum())
            kl_total += float(kl.sum())
            count += len(batch)
    reconstruction = reconstruction_total / count
    kl = kl_total / count
    return reconstruction, kl, reconstruction + beta * kl


def encode(model, values):
    output = []
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=1024)
    with torch.no_grad():
        for (batch,) in loader:
            output.append(model.encode(batch)[0].numpy())
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--system", required=True, choices=("apo", "holo"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--kl-warmup-epochs", type=int, default=30)
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    with np.load(args.fingerprints) as archive:
        values = np.asarray(archive["fingerprints"], dtype=np.float32)
        groups = np.asarray(archive["sequence_hash"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=20260804)
    train_indices, validation_indices = next(splitter.split(values, groups=groups))
    set_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(values[train_indices])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(values[validation_indices])),
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    model = PathVAE(values.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation = np.inf
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        warm_beta = args.beta * min(1.0, epoch / args.kl_warmup_epochs)
        reconstruction_total = kl_total = 0.0
        count = 0
        for (batch,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            log_probability, mu, logvar = model(batch)
            reconstruction, kl = loss_terms(batch, log_probability, mu, logvar)
            torch.mean(reconstruction + warm_beta * kl).backward()
            optimizer.step()
            reconstruction_total += float(reconstruction.detach().sum())
            kl_total += float(kl.detach().sum())
            count += len(batch)

        train_reconstruction = reconstruction_total / count
        train_kl = kl_total / count
        validation_reconstruction, validation_kl, validation_total = evaluate(
            model, validation_loader, args.beta
        )
        history.append(
            {
                "epoch": epoch,
                "kl_weight": warm_beta,
                "train_total": train_reconstruction + args.beta * train_kl,
                "train_reconstruction": train_reconstruction,
                "train_kl": train_kl,
                "validation_total": validation_total,
                "validation_reconstruction": validation_reconstruction,
                "validation_kl": validation_kl,
            }
        )
        if validation_total < best_validation:
            best_validation = validation_total
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    latent = encode(model, values)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "system": args.system,
            "seed": args.seed,
            "input_dimension": values.shape[1],
            "state_dict": best_state,
        },
        args.output / "model.pt",
    )
    np.savez_compressed(args.output / "embedding.npz", mu=latent)
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    with (args.output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "system": args.system,
                "seed": args.seed,
                "best_validation_loss": float(best_validation),
                "training_paths": int(len(train_indices)),
                "validation_paths": int(len(validation_indices)),
                "latent_variance": np.var(latent, axis=0).tolist(),
            },
            handle,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
