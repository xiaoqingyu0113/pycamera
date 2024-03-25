import torch
from torch import nn


class Factor(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        reveal_type(out)
        return out
