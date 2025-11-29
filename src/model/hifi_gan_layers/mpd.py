import torch
from torch import nn


class MPDSubBlock(nn.Module):
    def __init__(self, period: int, n_blocks: int = 4, negative_slope: float = 0.1):
        super().__init__()

        self.period = period

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=2 ** (5 + i) if i != 0 else 1,
                        out_channels=2 ** (5 + (i + 1)),
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    ),
                    nn.LeakyReLU(negative_slope=negative_slope),
                )
            )

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=(1, 0),
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
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


class MSD(nn.Module):
    def __init__(self, periods: list[int]):
        super().__init__()

        self.blocks = nn.ModuleList([MPDSubBlock(period) for period in periods])

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
