import torch.nn as nn

from src.model.layers.normalization import SpectralNorm, WeightNorm


class MSDSubBlock(nn.Module):
    def __init__(self, negatival_slope: float = 0.1, norm_type: str | None = None):
        super().__init__()

        self.activation = nn.LeakyReLU(negative_slope=negatival_slope)

        Norm = nn.Identity
        if norm_type == "weight":
            Norm = WeightNorm
        elif norm_type == "spectral":
            Norm = SpectralNorm

        self.blocks = nn.ModuleList(
            [
                Norm(nn.Conv1d(1, 16, 15, 1, 7)),
                Norm(nn.Conv1d(16, 64, 41, 4, 20, groups=4)),
                Norm(nn.Conv1d(64, 256, 41, 4, 20, groups=16)),
                Norm(nn.Conv1d(256, 1024, 41, 4, 20, groups=64)),
                Norm(nn.Conv1d(1024, 1024, 41, 4, 20, groups=256)),
                Norm(nn.Conv1d(1024, 1024, 5, 1, 2)),
            ]
        )

        self.final = nn.Conv1d(1024, 1, 3, 1, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, T)

        Returns:
            output (Tensor): (B, 1, T')
            activations (list[Tensor])
        """

        activations = []
        for block in self.blocks:
            x = block(x)
            x = self.activation(x)
            activations.append(x)

        x = self.final(x)
        activations.append(x)

        return x.view(x.size(0), -1), activations


class MSD(nn.Module):
    def __init__(
        self, n_blocks: int = 3, negatival_slope: float = 0.1, use_norm: bool = True
    ):
        super().__init__()

        norm_type = lambda i: None
        if use_norm:
            norm_type = lambda i: "spectral" if i == 0 else "weight"

        self.blocks = nn.ModuleList(
            [MSDSubBlock(negatival_slope, norm_type(i)) for i in range(n_blocks)]
        )

        # using hierarchical avg pooling for more robustness and fewer parameters
        self.avg_pools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, 2) for _ in range(n_blocks - 1)]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, T)

        Returns:
            outputs (list[Tensor])
            activations (list[list[Tensor]])
        """

        outputs = []
        activations = []
        for block in self.blocks:
            if len(activations) > 0:
                x = self.avg_pools[len(activations) - 1](x)

            output, subactivations = block(x)

            outputs.append(output)
            activations.append(subactivations)

        return outputs, activations

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
