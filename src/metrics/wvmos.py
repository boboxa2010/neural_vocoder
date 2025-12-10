import tempfile

import soundfile as sf
import torch
from torch import Tensor
from wvmos import get_wvmos

from src.metrics.base_metric import BaseMetric


class WVMOS(BaseMetric):
    def __init__(self, device: str = "cpu", sr: int = 22050, *args, **kwargs):
        super().__init__(*args, **kwargs)

        use_cuda = device.startswith("cuda")
        self.model = get_wvmos(cuda=use_cuda)
        self.sr = sr

    @torch.no_grad()
    def __call__(self, predicted: Tensor, **batch):
        """
        Args:
            predicted (Tensor): (B, 1, T)

        Returns:
            mos (Tensor): (B,)
        """
        predicted = predicted.squeeze(1)

        mos_scores = []

        for audio in predicted:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio.cpu().numpy(), self.sr)
                mos = self.model.calculate_one(f.name)
                mos_scores.append(mos)

        return torch.tensor(mos_scores, dtype=torch.float32).mean()
