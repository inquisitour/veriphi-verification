# src/core/models/trm_adapter.py
import torch
import torch.nn as nn
from typing import Optional

class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            act(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x):
        return self.net(x)

class TinyRecursiveMLP(nn.Module):
    """
    VeriPhi-compatible TRM-style model (fixed-depth unroll, MLP updates).

    Inputs:
      x: [B, Dx]   (embedded puzzle/question)
    Internals:
      y: [B, Dy]   (answer state)
      z: [B, Dz]   (latent state)

    Update schedule per "improvement step" k:
      - Repeat H times: z <- f_z([x, y, z])
      - Then y <- f_y([y, z])

    We export logits = head(y). All ops are plain Linear/ReLU → α,β-CROWN friendly.
    """
    def __init__(
        self,
        x_dim: int = 512,
        y_dim: int = 512,
        z_dim: int = 512,
        hidden: int = 1024,
        num_classes: int = 10,
        H_cycles: int = 3,   # "recursive reasoning" inner loop
        L_cycles: int = 4,   # "improvement steps"
        init_scale: float = 0.02,
        act=nn.ReLU,
    ):
        super().__init__()
        self.x_dim, self.y_dim, self.z_dim = x_dim, y_dim, z_dim
        self.H_cycles, self.L_cycles = H_cycles, L_cycles

        # encoders (optional: identity if data already embedded)
        self.x_enc = nn.Identity()
        self.y_init = nn.Parameter(torch.zeros(1, y_dim))
        self.z_init = nn.Parameter(torch.zeros(1, z_dim))

        # update blocks
        self.fz_in = nn.Linear(x_dim + y_dim + z_dim, z_dim)
        self.fz_mlp = MLPBlock(z_dim, hidden, act=act)

        self.fy_in = nn.Linear(y_dim + z_dim, y_dim)
        self.fy_mlp = MLPBlock(y_dim, hidden, act=act)

        self.head = nn.Linear(y_dim, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_scale)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: [B, x_dim]  (if you have images, flatten/encode before calling)
        """
        B = x.size(0)
        x = self.x_enc(x)

        y = self.y_init.expand(B, -1)
        z = self.z_init.expand(B, -1)

        for _ in range(self.L_cycles):
            for _ in range(self.H_cycles):
                z = self.fz_in(torch.cat([x, y, z], dim=-1))
                z = self.fz_mlp(z)
            y = self.fy_in(torch.cat([y, z], dim=-1))
            y = self.fy_mlp(y)

        logits = self.head(y)
        return logits


def create_trm_mlp(
    x_dim=512, y_dim=512, z_dim=512, hidden=1024,
    num_classes=10, H_cycles=3, L_cycles=4
) -> nn.Module:
    return TinyRecursiveMLP(
        x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, hidden=hidden,
        num_classes=num_classes, H_cycles=H_cycles, L_cycles=L_cycles
    )
