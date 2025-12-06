import torch
from torch import Tensor, nn


class DiscriminatorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, discriminator_fake: Tensor, discriminator_true: Tensor):
        """
        Args:
            discriminator_fake (Tensor): (B, T')
            discriminator_true (Tensor): (B, T')

        Returns:
            loss (Tensor)
        """
        return self.loss(discriminator_true, torch.ones_like(discriminator_true)) + self.loss(discriminator_fake, torch.zeros_like(discriminator_fake))


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.adv_loss = DiscriminatorAdvLoss()

    def forward(
        self, discriminators_fake: list[Tensor], discriminators_true: list[Tensor]
    ):
        """
        Args:
            discriminators_fake (list[Tensor])
            discriminators_true (list[Tensor])

        Returns:
            dict with:
                loss (Tensor)
        """

        loss = 0.0
        for fake, true in zip(discriminators_fake, discriminators_true):
            loss += self.adv_loss(fake, true)
        return {"loss": loss}
