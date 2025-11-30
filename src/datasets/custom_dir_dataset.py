from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, path: Path | str, *args, **kwargs):
        """
        Custom directory dataset for Vocoder inference and evaluation.

        Args:
            path (Path | str): Path to the dataset directory containing:
                - transcriptions/ subdirectory with text files
        """

        path = Path(path)

        if not (path / "transcriptions").exists():
            raise ValueError("Transcriptions directory not found")

        transcription_dir = path / "transcriptions"

        data = []
        for path in Path(transcription_dir).iterdir():
            entry = {"uttid": path.stem}
            with path.open() as f:
                entry["text"] = f.read().strip()

            if len(entry) > 0:
                data.append(entry)

            print(entry.values())
        super().__init__(data, *args, **kwargs)
