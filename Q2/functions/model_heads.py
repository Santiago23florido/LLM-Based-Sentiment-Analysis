from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    """Return a PyTorch activation module by name."""
    name = name.lower().strip()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}. Use one of: relu, gelu, tanh")


def build_mlp(
    input_dim: int,
    num_classes: int,
    hidden_layers: Sequence[int] = (256, 64),
    dropout: float = 0.2,
    activation: str = "relu",
) -> nn.Sequential:
    """Build a simple feed-forward MLP classifier head.

    Parameters
    ----------
    input_dim:
        Dimensionality of input features (e.g., 768 for BERT base embeddings).
    num_classes:
        Number of target classes.
    hidden_layers:
        Tuple/list of hidden layer sizes, e.g., (256, 64).
        If empty, the network becomes a single linear layer.
    dropout:
        Dropout probability applied after each hidden layer.
    activation:
        Activation function name.

    Returns
    -------
    nn.Sequential
        A sequential module that maps (batch, input_dim) -> (batch, num_classes).
    """

    act = _get_activation(activation)

    layers = []
    prev_dim = input_dim

    for h in hidden_layers:
        layers.append(nn.Linear(prev_dim, int(h)))
        layers.append(act)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev_dim = int(h)

    # Output layer (logits)
    layers.append(nn.Linear(prev_dim, num_classes))

    return nn.Sequential(*layers)


class SentimentMLPClassifier(nn.Module):
    """A thin wrapper around an MLP head.

    Forward returns logits (no softmax). Use CrossEntropyLoss during training.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: Sequence[int] = (256, 64),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearHead(nn.Module):
    """Baseline linear head: equivalent to multinomial logistic regression."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
