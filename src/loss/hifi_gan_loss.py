from torch import nn

from src.loss.generator_loss import GeneratorLoss
from src.loss.discriminator_loss import DiscriminatorLoss

class HiFiGANLoss(nn.Module):
    def __init__(self, feature_weight: float = 2.0, mel_weight: float = 45.0):
        super().__init__()

        self.generator_loss = GeneratorLoss(feature_weight, mel_weight)
        self.discriminator_loss = DiscriminatorLoss()

