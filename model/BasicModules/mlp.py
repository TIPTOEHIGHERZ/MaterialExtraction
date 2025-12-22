import torch
import torch.nn as nn


class BasicMlp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mlp_ratio=4.
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(in_channels * mlp_ratio), in_channels)
        )

        return
    
    def forward(self, x):
        return self.mlp(x)
