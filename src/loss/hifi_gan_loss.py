from torch import nn

from src.loss.discriminator_loss import DiscriminatorLoss
from src.loss.generator_loss import GeneratorLoss


class HiFiGANLoss(nn.Module):
    def __init__(
        self,
        feature_weight: float = 2.0,
        mel_weight: float = 45.0,
        normalize_adv: bool = False,
        normalize_feature: bool = False,
    ):
        super().__init__()

        self.generator_loss = GeneratorLoss(
            feature_weight, mel_weight, normalize_adv, normalize_feature
        )
        self.discriminator_loss = DiscriminatorLoss()
