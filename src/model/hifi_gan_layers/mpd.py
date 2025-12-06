import torch
from torch import nn

from src.model.hifi_gan_layers.normalization import SpectralNorm, WeightNorm


class MPDSubBlock(nn.Module):
    def __init__(
        self,
        period: int,
        n_blocks: int = 4,
        negative_slope: float = 0.1,
        norm_type: str | None = None,
    ):
        super().__init__()

        self.period = period

        Norm = nn.Identity
        if norm_type == "weight":
            Norm = WeightNorm
        elif norm_type == "spectral":
            Norm = SpectralNorm

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    Norm(
                        nn.Conv2d(
                            in_channels=2 ** (5 + i) if i != 0 else 1,
                            out_channels=2 ** (5 + (i + 1)),
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            padding=(2, 0),
                        )
                    ),
                    nn.LeakyReLU(negative_slope=negative_slope),
                )
            )

        self.head = nn.Sequential(
            Norm(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=(5, 1),
                    stride=(1, 1),
                    padding=(1, 0),
                )
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            Norm(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1,
                    kernel_size=(3, 1),
                    stride=(1, 1),
                    padding=(1, 0),
                )
            ),
        )

    def _reshape(self, x):
        """
        Args:
            x (Tensor): (B, C, T)

        Returns:
            Tensor: (B, C, ⌈T / period⌉, period)
        """

        x = torch.nn.functional.pad(x, (0, self.period - (x.shape[2] % self.period)))
        return x.reshape(x.shape[0], x.shape[1], -1, self.period)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, 1, T)

        Returns:
            output (Tensor): (B, 1, T')
            activations (list[Tensor])
        """
        x = self._reshape(x)

        activations = []
        for block in self.blocks:
            x = block(x)
            activations.append(x)

        x = self.head(x)
        activations.append(x)

        return x.view(x.size(0), -1), activations


class MPD(nn.Module):
    def __init__(self, periods: list[int], norm_type: str | None = "weigth"):
        super().__init__()

        self.blocks = nn.ModuleList(
            [MPDSubBlock(period, norm_type=norm_type) for period in periods]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, 1, T)

        Returns:
            outputs (list[Tensor])
            activations (list[list[Tensor]])
        """

        outputs = []
        activations = []
        for block in self.blocks:
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
