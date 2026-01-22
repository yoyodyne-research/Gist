from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

from ncps.torch import CfC


class FiLM(nn.Module):
    """Feature-wise linear modulation for conditioning.
    gamma, beta from macros/metadata; initialized near identity (gamma≈1, beta≈0).
    """
    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, out_dim)
        self.beta = nn.Linear(cond_dim, out_dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # y: [T,B,D] or [B,D]; c: [T,B,C] or [B,C] (broadcast along time as needed)
        if y.dim() == 3 and c.dim() == 2:
            c = c.unsqueeze(0).expand(y.size(0), -1, -1)
        if y.dim() == 2 and c.dim() == 3:
            c = c[-1]
        g = 1.0 + 0.5 * torch.tanh(self.gamma(c))
        b = 0.5 * torch.tanh(self.beta(c))
        return g * y + b


class TinyController(nn.Module):
    def __init__(self, in_dim: int, macro_dim: int, cond_dim: int,
                 hid: int = 12, enc_dim: int = 16, out_dim: int = 4):
        super().__init__()
        self.enc = nn.Linear(in_dim, enc_dim)
        # CfC recurrent core (NCP/CfC only; no GRU fallback)
        self.rnn = CfC(enc_dim, hid)
        self.head = nn.Linear(hid, out_dim)
        self.film_out = FiLM(macro_dim + cond_dim, out_dim)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], macros: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [T,B,in_dim], h: [1,B,hid] for GRU or [B,hid] for CfC, macros: [B,M] or [T,B,M], cond: [B,C] or [T,B,C]
        z = torch.tanh(self.enc(x))
        y, h_next = self.rnn(z, h)  # CfC API: (T,B,H), (B,H)
        out = self.head(y)
        out = self.film_out(out, torch.cat([macros, cond], dim=-1))
        return out, h_next
