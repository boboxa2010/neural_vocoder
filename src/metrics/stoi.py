from torch import Tensor
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(self, fs, device: str = "cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metric = ShortTimeObjectiveIntelligibility(fs=fs).to(device)

    def __call__(self, predicted: Tensor, target: Tensor, **batch):
        """
        Args:
            predicted (Tensor): (B, 1, T)
            target (Tensor): (B, 1, T)

        Returns:
            metric (Tensor)
        """
        return self.metric(predicted, target)
