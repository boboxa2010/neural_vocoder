from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomAudioDirDataset(BaseDataset):
    def __init__(self, path: Path | str, *args, **kwargs):
        """
        Custom directory dataset for Vocoder inference and evaluation (Audio only).

        Args:
            path (Path | str): Path to the dataset directory containing:
                - gt_audio/ subdirectory with audio files
        """
        path = Path(path)

        if not (path / "gt_audio").exists():
            raise ValueError("gt_audio directory not found")

        audio_dir = path / "gt_audio"

        data = []
        for file_path in audio_dir.iterdir():
            entry = {"text_id": file_path.stem, "text": None}
            if file_path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(file_path)
                t_info = torchaudio.info(entry["path"])
                entry["audio_len"] = t_info.num_frames / t_info.sample_rate
            data.append(entry)

        super().__init__(data, *args, **kwargs)
