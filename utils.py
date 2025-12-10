from IPython.display import Audio, display
from matplotlib import pyplot as plt

import torch
import torchaudio

from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig

def visualize_audio(wav: torch.Tensor, sr: int = 22050, title: str = "Audio"):
    if wav.dim() == 2:
        wav = wav.mean(dim=0)

    if sr != 22050:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=22050)
        sr = 22050
        
    mel_transform = MelSpectrogram(MelSpectrogramConfig())
    mel = mel_transform(wav.unsqueeze(0))
    mel = mel.squeeze(0).cpu()

    fig, axs = plt.subplots(1, 2, figsize=(25, 5))

    axs[0].plot(wav, alpha=.7, c='green')
    axs[0].set_title(title, size=22)
    axs[0].set_xlabel('Time', size=20)
    axs[0].set_ylabel('Amplitude', size=20)
    axs[0].grid(True)

    im = axs[1].imshow(mel, aspect='auto', origin='lower')
    axs[1].set_title("Mel Spectrogram", size=22)
    axs[1].set_xlabel("Frames", size=20)
    axs[1].set_ylabel("Mel bins", size=20)
    fig.colorbar(im, ax=axs[1])

    plt.tight_layout()
    plt.show()

    display(Audio(wav, rate=sr, normalize=False))

def visualize_diff(wav1: torch.Tensor, wav2: torch.Tensor, sr: int = 22050, sr_orig: int = 22050):

    if wav1.dim() == 2:
        wav1 = wav1.mean(dim=0)
    if wav2.dim() == 2:
        wav2 = wav2.mean(dim=0)

    if sr != 22050:
        wav1 = torchaudio.functional.resample(wav1, orig_freq=sr_orig, new_freq=22050)
        wav2 = torchaudio.functional.resample(wav2, orig_freq=sr, new_freq=22050)
        sr = 22050


    mel_transform = MelSpectrogram(MelSpectrogramConfig())
    mel1 = mel_transform(wav1.unsqueeze(0)).squeeze(0).cpu()
    mel2 = mel_transform(wav2.unsqueeze(0)).squeeze(0).cpu()
    mel_diff = mel1 - mel2

    plt.figure(figsize=(12, 5))
    plt.imshow(mel_diff, aspect='auto', origin='lower', cmap='bwr')
    plt.title("Mel Spectrogram Diff", size=18)
    plt.xlabel("Frames", size=14)
    plt.ylabel("Mel bins", size=14)
    plt.colorbar()
    plt.show()