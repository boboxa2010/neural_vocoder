import torch
from speechbrain.inference.TTS import FastSpeech2


class FastSpeech2Wrapper:
    def __init__(
        self,
        tts_model="speechbrain/tts-fastspeech2-ljspeech",
        savedir="pretrained_models/tts-fastspeech2-ljspeech",
        device="cpu",
    ):
        self.device = device
        self.tts = FastSpeech2.from_hparams(source=tts_model, savedir=savedir).to(
            device
        )

    @torch.no_grad()
    def generate(self, text):
        mel, _, _, _ = self.tts.encode_text([text])
        return mel.to(self.device)
