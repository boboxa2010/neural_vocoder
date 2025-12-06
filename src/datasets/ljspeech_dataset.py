import json
import os
import shutil
from pathlib import Path

import torchaudio
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

LJS_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


class LJSpeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, val_ratio=0.1, *args, **kwargs):
        assert part in ["train", "val"]

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)

        self._data_dir = data_dir
        self.val_ratio = val_ratio

        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print("Downloading LJSpeech ...")

        wget.download(LJS_URL, str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        os.remove(str(arch_path))

    def _get_or_load_index(self, part):
        train_index_path = self._data_dir / "ljs_train_index.json"
        val_index_path = self._data_dir / "ljs_val_index.json"

        if train_index_path.exists() and val_index_path.exists():
            with train_index_path.open() as f:
                train_index = json.load(f)
            with val_index_path.open() as f:
                val_index = json.load(f)
            return train_index if part == "train" else val_index

        full_index = self._create_index()

        n_total = len(full_index)
        n_val = int(n_total * self.val_ratio)
        n_train = n_total - n_val

        train_index = full_index[:n_train]
        val_index = full_index[n_train:]

        with train_index_path.open("w") as f:
            json.dump(train_index, f, indent=2)
        with val_index_path.open() as f:
            json.dump(val_index, f, indent=2)

        return train_index if part == "train" else val_index

    def _create_index(self):
        dataset_root = self._data_dir / "LJSpeech-1.1"

        if not dataset_root.exists():
            self._load_dataset()

        metadata_path = dataset_root / "metadata.csv"
        wav_dir = dataset_root / "wavs"

        index = []

        with metadata_path.open() as f:
            for line in tqdm(f, total=13100):
                parts = line.strip().split("|")
                file_id = parts[0]
                text = parts[2].strip()

                wav_path = wav_dir / f"{file_id}.wav"

                info = torchaudio.info(str(wav_path))
                audio_len = info.num_frames / info.sample_rate

                index.append(
                    {
                        "path": str(wav_path.resolve()),
                        "text": text,
                        "audio_len": audio_len,
                    }
                )
        return index
