from torch import nn

from src.model.layers.generator import Generator
from src.model.layers.mpd import MPD
from src.model.layers.msd import MSD


class HiFiGAN(nn.Module):
    def __init__(
        self,
        generator_in_channels: int,
        h_u: int,
        k_u: list[int],
        k_r: list[int],
        D_r: list[tuple[int, int]],
        periods: list[int],
        msd_n_blocks: int = 3,
        generator_norm_type: str | None = "weight",
        mpd_norm_type: str | None = "weight",
        msd_use_norm: bool = True,
        negative_slope: float = 0.1,
        expand_kernel_size: int = 7,
        project_kernel_size: int = 7,
    ):
        super().__init__()

        self.generator = Generator(
            in_channels=generator_in_channels,
            h_u=h_u,
            k_u=k_u,
            k_r=k_r,
            D_r=D_r,
            negative_slope=negative_slope,
            expand_kernel_size=expand_kernel_size,
            project_kernel_size=project_kernel_size,
            norm_type=generator_norm_type,
        )
        self.mpd = MPD(periods, negative_slope, mpd_norm_type)
        self.msd = MSD(msd_n_blocks, negative_slope, msd_use_norm)

    def module_params(self, module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    def module_trainable_params(self, module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def __str__(self):
        lines = []
        lines.append(super().__str__())
        lines.append("=" * 60)

        for name, module in [
            ("Generator", self.generator),
            ("MPD", self.mpd),
            ("MSD", self.msd),
        ]:
            total = self.module_params(module)
            trainable = self.module_trainable_params(module)
            lines.append(f"{name}:")
            lines.append(f"  Total parameters:      {total:,}")
            lines.append(f"  Trainable parameters:  {trainable:,}")
            lines.append("-" * 60)

        all_params = self.module_params(self)
        trainable_params = self.module_trainable_params(self)

        lines.append(f"MODEL TOTAL:")
        lines.append(f"  Total parameters:      {all_params:,}")
        lines.append(f"  Trainable parameters:  {trainable_params:,}")
        lines.append("=" * 60)

        return "\n".join(lines)
