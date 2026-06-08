"""
models/gnn_model.py
===================
AttentiveFP-based GNN for PK parameter prediction.

Architecture:
  - AttentiveFP encoder (PyTorch Geometric built-in)
      • node features : 66 (atom-level one-hot + properties)
      • edge features : 11 (bond-level)
      • hidden_channels: tunable (64–256)
      • num_layers      : tunable (2–6)  — graph-level attention rounds
      • num_timesteps   : tunable (2–6)  — readout GRU timesteps
      • dropout         : tunable (0–0.5)
  - Two separate linear heads: one for log10(CL), one for log10(Vd)
  - Outputs a (2,) tensor per molecule: [log10_CL, log10_Vd]

Training contract:
  - Targets are log10-transformed throughout (same as RF/XGB)
  - Loss: MSE on log10 scale (one loss per param, summed)
  - Optimizer: AdamW with cosine annealing LR schedule
  - Early stopping on validation loss (patience configurable)

Usage:
    from models.gnn_model import PKAttentiveFP
    model = PKAttentiveFP(hidden_channels=128, num_layers=3, num_timesteps=3)
    # training handled by training/train_gnn.py
"""

import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP

# ── Feature dimensions (must match features/graph_builder.py) ─────────────────
NODE_DIM = 66   # atom feature vector length
EDGE_DIM = 11   # bond feature vector length


class PKAttentiveFP(nn.Module):
    """
    AttentiveFP encoder + two linear prediction heads.

    Predicts log10(CL) and log10(Vd) simultaneously.
    Both targets are on the log10 scale throughout.
    """

    def __init__(
        self,
        hidden_channels: int   = 128,
        num_layers:      int   = 3,
        num_timesteps:   int   = 3,
        dropout:         float = 0.2,
        node_dim:        int   = NODE_DIM,
        edge_dim:        int   = EDGE_DIM,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers      = num_layers
        self.num_timesteps   = num_timesteps
        self.dropout_rate    = dropout

        # Graph encoder — outputs graph-level embedding of shape (B, hidden_channels)
        self.encoder = AttentiveFP(
            in_channels      = node_dim,
            hidden_channels  = hidden_channels,
            out_channels     = hidden_channels,
            edge_dim         = edge_dim,
            num_layers       = num_layers,
            num_timesteps    = num_timesteps,
            dropout          = dropout,
        )

        # Prediction heads (separate for CL and Vd)
        self.head_CL = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.head_Vd = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x          : node features  (N_atoms_total, node_dim)
            edge_index : edge connectivity (2, N_edges)
            edge_attr  : edge features   (N_edges, edge_dim)
            batch      : graph membership (N_atoms_total,)
        Returns:
            out : (B, 2) — [log10_CL, log10_Vd] per molecule
        """
        emb = self.encoder(x, edge_index, edge_attr, batch)   # (B, hidden)
        cl  = self.head_CL(emb)                                # (B, 1)
        vd  = self.head_Vd(emb)                                # (B, 1)
        return torch.cat([cl, vd], dim=1)                      # (B, 2)

    # ── Convenience save/load ─────────────────────────────────────────────────

    def save(self, path: str, hyperparams: Optional[dict] = None):
        """Save model weights + architecture hyperparams."""
        payload = {
            'state_dict': self.state_dict(),
            'hyperparams': hyperparams or {
                'hidden_channels': self.hidden_channels,
                'num_layers':      self.num_layers,
                'num_timesteps':   self.num_timesteps,
                'dropout':         self.dropout_rate,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PKAttentiveFP':
        """Load model from saved checkpoint."""
        payload = torch.load(path, map_location=device, weights_only=False)
        hp      = payload['hyperparams']
        model   = cls(**hp)
        model.load_state_dict(payload['state_dict'])
        model = model.to(device)
        model.eval()
        return model
