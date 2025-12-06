import torch
import torch.nn as nn

from src.model.layers.normalization import SpectralNorm, WeightNorm


class ResSubBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        k_r: int,
        D_r: tuple[int, int],
        negative_slope: float = 0.1,
        norm_type: str | None = None,
    ):
        super().__init__()

        Norm = nn.Identity
        if norm_type == "weight":
            Norm = WeightNorm
        elif norm_type == "spectral":
            Norm = SpectralNorm

        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

        self.conv1 = Norm(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=k_r,
                dilation=D_r[0],
                padding=(k_r * D_r[0] - D_r[0]) // 2,
            )
        )

        self.conv2 = Norm(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=k_r,
                dilation=D_r[1],
                padding=(k_r * D_r[1] - D_r[1]) // 2,
            )
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, T)

        Returns:
            Tensor: (B, C, T)
        """

        output = self.activation(x)
        output = self.conv1(output)
        output = self.activation(output)
        output = self.conv2(output)

        return x + output


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        D_r: list[tuple[int, int]],
        norm_type: str | None = None,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for m in range(len(D_r)):
            self.blocks.append(ResSubBlock(channels, k, D_r[m], norm_type=norm_type))

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, T)

        Returns:
            Tensor: (B, C, T)
        """

        output = x

        for block in self.blocks:
            output = block(output)

        return output


class MRF(nn.Module):
    def __init__(
        self,
        channels: int,
        k_r: list[int],
        D_r: list[tuple[int, int]],
        norm_type: str | None = None,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for k_i in k_r:
            self.blocks.append(ResBlock(channels, k_i, D_r, norm_type=norm_type))

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, T)

        Returns:
            Tensor: (B, C, T)
        """

        output = torch.zeros_like(x)

        for block in self.blocks:
            output += block(x)

        return output


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        h_u: int,
        k_u: list[int],
        k_r: list[int],
        D_r: list[tuple[int, int]],
        negative_slope: float = 0.1,
        expand_kernel_size: int = 7,
        project_kernel_size: int = 7,
        norm_type: str | None = "weight",
    ):
        super().__init__()

        Norm = nn.Identity
        if norm_type == "weight":
            Norm = WeightNorm
        elif norm_type == "spectral":
            Norm = SpectralNorm

        self.expand = Norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=h_u,
                kernel_size=expand_kernel_size,
                padding=(expand_kernel_size - 1) // 2,
            )
        )

        self.blocks = nn.ModuleList()
        for l in range(len(k_u)):
            self.blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(negative_slope=negative_slope),
                    Norm(
                        nn.ConvTranspose1d(
                            in_channels=h_u // (2**l),
                            out_channels=h_u // (2 ** (l + 1)),
                            kernel_size=k_u[l],
                            stride=k_u[l] // 2,
                            padding=(k_u[l] - k_u[l] // 2) // 2,
                        )
                    ),
                    MRF(h_u // (2 ** (l + 1)), k_r, D_r, norm_type=norm_type),
                )
            )

        self.project = nn.Sequential(
            nn.LeakyReLU(negative_slope=negative_slope),
            Norm(
                nn.Conv1d(
                    in_channels=h_u // (2 ** len(k_u)),
                    out_channels=1,
                    kernel_size=project_kernel_size,
                    padding=(project_kernel_size - 1) // 2,
                )
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, F, T)

        Returns:
            Tensor: (B, 1, hop_length x T)
        """

        x = self.expand(x)

        for block in self.blocks:
            x = block(x)

        x = self.project(x)

        return x

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
