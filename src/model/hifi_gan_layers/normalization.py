import torch.nn as nn


class WeightNorm(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()

        self.module = nn.utils.weight_norm(module, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class SpectralNorm(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()

        self.module = nn.utils.spectral_norm(module, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
