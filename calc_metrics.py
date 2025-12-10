import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def load_audio(path, target_sr):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor


def get_audio_files(folder: Path, exts=(".wav", ".mp3", ".flac")):
    files = {}
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            files.setdefault(p.stem, p)
    return files


def load_wavs_directory(directory_path, target_sr):
    directory = Path(directory_path)

    if not directory.is_dir():
        raise ValueError(f"Expected audios in {directory}")

    mix_files = get_audio_files(directory)

    wavs = {}
    for id in mix_files.keys():
        wavs[id] = load_audio(mix_files[id], target_sr)

    return wavs


def calculate_metrics(config: DictConfig):
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    print(f"using device: {device}")

    target_sr = config.get("sample_rate", 22050)
    print(f"sample rate: {target_sr}")

    ground_truth_wavs = load_wavs_directory(config.paths.ground_truth, target_sr)
    predicted_wavs = load_wavs_directory(config.paths.predictions, target_sr)

    common_ids = set(ground_truth_wavs.keys()) & set(predicted_wavs.keys())

    if not common_ids:
        raise ValueError("no common ids")

    print(f"found {len(common_ids)} common ids")

    metrics = []
    for metric_config in config.metrics:
        metric = instantiate(metric_config, device=device)
        metrics.append(metric)

    print(f"metrics: {[m.name for m in metrics]}")

    metric_totals = {metric.name: 0.0 for metric in metrics}
    count = 0

    for id in tqdm(common_ids):
        ground_truth = ground_truth_wavs[id].to(device)
        predicted = predicted_wavs[id].to(device)

        batch = {
            "predicted": predicted,
            "target": ground_truth,
        }

        for metric in metrics:
            metric_value = metric(**batch)
            metric_totals[metric.name] += metric_value

        count += 1

    results = {"num_wavs": count}

    for metric_name, total_value in metric_totals.items():
        avg_value = total_value / count if count > 0 else 0.0
        results[metric_name] = avg_value

    return results


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config: DictConfig):
    results = calculate_metrics(config)
    for key, value in results.items():
        if key != "num_wavs":
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
