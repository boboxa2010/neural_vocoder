import torch
from torch import Tensor, nn


class GeneratorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, discriminators_fake: list[Tensor]):
        """
        Args:
            discriminators_fake (list[Tensor])

        Returns:
            loss (Tensor)
        """

        loss = 0.0
        for fake in discriminators_fake:
            loss += self.loss(fake, torch.ones_like(fake))
        return loss


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(
        self, features_fake: list[list[Tensor]], features_true: list[list[Tensor]]
    ):
        """
        Args:
            features_fake (list[list[Tensor]])
            features_true (list[list[Tensor]])

        Returns:
            loss (Tensor)
        """

        loss = 0.0
        for discriminator in range(len(features_fake)):
            for prediction, target in zip(
                features_fake[discriminator], features_true[discriminator]
            ):
                loss += self.loss(prediction, target)
        return loss


class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, mel_fake: Tensor, mel_true: Tensor):
        """
        Args:
            mel_fake (Tensor): (B, F, T)
            mel_true (Tensor): (B, F, T)

        Returns:
            loss (Tensor)
        """
        return self.loss(mel_fake, mel_true)


class GeneratorLoss(nn.Module):
    def __init__(self, feature_weight: float = 2.0, mel_weight: float = 45.0):
        super().__init__()

        self.feature_weight = feature_weight
        self.mel_weight = mel_weight

        self.adv_loss = GeneratorAdvLoss()
        self.feature_loss = FeatureLoss()
        self.mel_loss = MelLoss()

    def forward(
        self,
        discriminators_fake: list[Tensor],
        features_fake: list[list[Tensor]],
        features_true: list[list[Tensor]],
        mel_fake: Tensor,
        mel_true: Tensor,
    ):
        """
        Args:
            discriminators_fake (list[Tensor])
            features_fake (list[list[Tensor]])
            features_true (list[list[Tensor]])
            mel_fake (Tensor): (B, F, T)
            mel_true (Tensor): (B, F, T)

        Returns:
            loss (Tensor)
        """

        adv_loss = self.adv_loss(discriminators_fake)
        feature_loss = self.feature_loss(features_fake, features_true)
        mel_loss = self.mel_loss(mel_fake, mel_true)

        return {
            "loss": adv_loss
            + self.feature_weight * feature_loss
            + self.mel_weight * mel_loss,
            "adv_loss": adv_loss,
            "feature_loss": feature_loss,
            "mel_loss": mel_loss,
        }
