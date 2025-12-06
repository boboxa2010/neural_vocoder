from torch import Tensor
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from src.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    def __init__(self, fs, mode, device: str = "cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metric = PerceptualEvaluationSpeechQuality(fs=fs, mode=mode).to(device)

    def __call__(self, predicted: Tensor, target: Tensor, **batch):
        """
        Args:
            predicted (Tensor): (B, 1, T)
            target (Tensor): (B, 1, T)

        Returns:
            metric (Tensor)
        """
        return self.metric(predicted, target)
