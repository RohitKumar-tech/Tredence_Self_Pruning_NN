"""
Self-Pruning Neural Network for CIFAR-10
Tredence Analytics — AI Engineering Intern Case Study

Implements a feed-forward neural network that learns to prune itself during
training using learnable sigmoid gate parameters and L1 sparsity regularization.

    Total Loss = CrossEntropyLoss + lambda * sum(sigmoid(gate_scores))

Run:
    python self_pruning_network.py

Requirements:
    pip install torch torchvision matplotlib numpy
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — PrunableLinear layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Custom linear layer with learnable gate parameters.

    Each weight w_ij has a corresponding gate score g_ij (a learnable scalar).
    The gate value is sigmoid(g_ij) ∈ (0, 1).
    The effective weight used in the forward pass is:

        pruned_weight = weight * sigmoid(gate_scores)

    When gate_scores → -∞, sigmoid → 0, so the weight is effectively pruned.
    Gradients flow through both `weight` and `gate_scores` automatically
    because all operations are differentiable PyTorch ops.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight, registered as parameter
        # Initialized to 0 → sigmoid(0) = 0.5, gates start half-open
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)          # (out, in) ∈ (0,1)
        pruned_weights = self.weight * gates             # element-wise gating
        return F.linear(x, pruned_weights, self.bias)   # x @ W^T + b

    def gates(self) -> torch.Tensor:
        """Current gate values (detached, for analysis)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def sparsity_contribution(self) -> torch.Tensor:
        """L1 norm of gates for this layer (used in total sparsity loss)."""
        return torch.sigmoid(self.gate_scores).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 using only PrunableLinear layers.
    Architecture: 3072 → 512 → 256 → 128 → 10
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.prunable_layers = nn.ModuleList([
            PrunableLinear(3072, 512),
            PrunableLinear(512, 256),
            PrunableLinear(256, 128),
            PrunableLinear(128, 10),
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)                           # flatten CIFAR image
        for i, layer in enumerate(self.prunable_layers[:-1]):
            x = F.relu(layer(x))
            if i < 2:
                x = self.drop(x)
        return self.prunable_layers[-1](x)                  # raw logits

    # ── Sparsity helpers ──────────────────────────────────────────────────────

    def sparsity_loss(self) -> torch.Tensor:
        """Part 2: SparsityLoss = sum of all gate values (L1 of sigmoid gates)."""
        return sum(layer.sparsity_contribution() for layer in self.prunable_layers)

    def all_gates(self) -> torch.Tensor:
        """Concatenated gate values from every layer (for plotting/analysis)."""
        return torch.cat([layer.gates().flatten() for layer in self.prunable_layers])

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Fraction of weights whose gate < threshold.
        A high value means successful self-pruning.
        """
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def cifar10_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Training and Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lam: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    """One training epoch. Returns (total_loss, ce_loss, sparsity_loss) averages."""
    model.train()
    tot = ce_sum = sp_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        ce   = F.cross_entropy(logits, y)           # classification loss
        sp   = model.sparsity_loss()                # L1 of gate values
        loss = ce + lam * sp                        # Part 2: Total Loss formula

        loss.backward()
        optimizer.step()

        tot   += loss.item()
        ce_sum += ce.item()
        sp_sum += sp.item()

    n = len(loader)
    return tot / n, ce_sum / n, sp_sum / n


@torch.no_grad()
def evaluate(model: SelfPruningNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def run_experiment(
    lam: float,
    train_dl: DataLoader,
    test_dl: DataLoader,
    device: torch.device,
    epochs: int = 30,
) -> Tuple[float, float, SelfPruningNet]:
    log.info("\n" + "=" * 60)
    log.info(f"  Training with λ = {lam:.0e}  ({epochs} epochs)")
    log.info("=" * 60)

    model = SelfPruningNet().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        total, ce, sp = train_one_epoch(model, train_dl, opt, lam, device)
        acc = evaluate(model, test_dl, device)
        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            sparsity = model.sparsity_level()
            log.info(
                f"  Ep {epoch:3d}/{epochs} | loss={total:.4f}"
                f" (ce={ce:.4f} sp={sp:.1f}) | acc={acc*100:.2f}%"
                f" | sparsity={sparsity*100:.1f}%"
            )

    final_acc      = evaluate(model, test_dl, device)
    final_sparsity = model.sparsity_level()
    log.info(f"\n  FINAL → acc={final_acc*100:.2f}%  sparsity={final_sparsity*100:.1f}%")
    return final_acc, final_sparsity, model


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(model: SelfPruningNet, lam: float, path: str) -> None:
    """
    Histogram of all gate values for the given model.
    A successful result shows a large spike near 0 (pruned weights) and
    a smaller cluster away from 0 (surviving important weights).
    """
    gates = model.all_gates().numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gates, bins=120, color="steelblue", alpha=0.85, edgecolor="none")
    ax.axvline(0.01, color="red", linestyle="--", linewidth=1.5, label="Prune threshold (0.01)")
    ax.set_xlabel("Gate Value", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"Gate Value Distribution  —  λ = {lam:.0e}", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved gate distribution plot → {path}")


def plot_tradeoff(results: Dict[float, dict], path: str) -> None:
    """Dual-axis plot: accuracy and sparsity vs lambda."""
    lams     = list(results.keys())
    accs     = [results[l]["accuracy"] * 100 for l in lams]
    spars    = [results[l]["sparsity"] * 100 for l in lams]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    c1, c2   = "steelblue", "darkorange"
    ax1.plot(range(len(lams)), accs, "o-", color=c1, lw=2, ms=8, label="Test Accuracy (%)")
    ax1.set_xticks(range(len(lams)))
    ax1.set_xticklabels([f"{l:.0e}" for l in lams], fontsize=11)
    ax1.set_xlabel("Lambda (λ)", fontsize=13)
    ax1.set_ylabel("Test Accuracy (%)", color=c1, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2 = ax1.twinx()
    ax2.plot(range(len(lams)), spars, "s--", color=c2, lw=2, ms=8, label="Sparsity (%)")
    ax2.set_ylabel("Sparsity Level (%)", color=c2, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=c2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=11)
    fig.suptitle("Sparsity vs Accuracy Trade-off Across λ Values", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved trade-off plot → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_dl, test_dl = cifar10_loaders(batch_size=128)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Three lambda values: low / medium / high
    lambda_values = [1e-5, 1e-4, 1e-3]
    epochs        = 30

    results: Dict[float, dict] = {}
    best_lam   = None
    best_acc   = -1.0

    for lam in lambda_values:
        acc, sparsity, model = run_experiment(lam, train_dl, test_dl, device, epochs=epochs)
        results[lam] = {"accuracy": acc, "sparsity": sparsity, "model": model}
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Lambda':<12}{'Test Accuracy':>16}{'Sparsity Level':>16}")
    print("-" * 55)
    for lam, res in results.items():
        print(f"{lam:<12.0e}{res['accuracy']*100:>15.2f}%{res['sparsity']*100:>15.1f}%")
    print("=" * 55)

    # ── Plots ─────────────────────────────────────────────────────────────────
    best_model = results[best_lam]["model"]
    plot_gate_distribution(best_model, best_lam, os.path.join(out_dir, "gate_distribution.png"))
    plot_tradeoff(results, os.path.join(out_dir, "lambda_comparison.png"))

    # ── Update report with actual numbers ────────────────────────────────────
    report_path = os.path.join(out_dir, "report.md")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = f.read()
        rows = ""
        for lam, res in results.items():
            marker = " ← best" if lam == best_lam else ""
            rows += f"| {lam:.0e}       | {res['accuracy']*100:.2f}%             | {res['sparsity']*100:.1f}%              |{marker}\n"
        report = report.replace("<!-- RESULTS_PLACEHOLDER -->", rows)
        with open(report_path, "w") as f:
            f.write(report)
        log.info(f"Report updated with real results → {report_path}")

    log.info("\nAll done! Files written to: " + out_dir)


if __name__ == "__main__":
    main()
