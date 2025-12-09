from pathlib import Path

import torch
import torchaudio
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Synthesizer(BaseTrainer):
    """
    Synthesizer class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        acoustic_model,
        config,
        device,
        save_path,
        dataloaders=None,
        text: str | None = None,
        text_name: str | None = None,
        metrics=None,
        batch_transforms=None,
        instance_transforms=None,
        skip_model_load=False,
        resynthesize=False,
    ):
        """
        Initialize the Synthesizer.

        Args:
            model (nn.Module): PyTorch model.
            acoustic_model (nn.Module): Acoustic model.
            config (DictConfig): run config containing synthesizer config.
            device (str): device for tensors and model.
            save_path (str): path to save model predictions and other
                information.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text (str | None): text to synthesize.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Synthesizer Class.
        """
        assert (
            skip_model_load or config.synthesizer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.synthesizer

        self.device = device

        self.model = model
        self.acoustic_model = acoustic_model

        self.text = text
        self.text_name = text_name

        self.batch_transforms = batch_transforms

        self.evaluation_dataloaders = (
            {k: v for k, v in dataloaders.items()} if dataloaders is not None else {}
        )

        self.save_path = save_path

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        self.resynthesize = resynthesize

        self.mel_spec = instance_transforms["inference"]["get_spectrogram"].to(device)

        if not skip_model_load:
            self._from_pretrained(config.synthesizer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        if len(self.evaluation_dataloaders) > 0:
            part_logs = {}
            for part, dataloader in self.evaluation_dataloaders.items():
                logs = self._inference_part(part, dataloader)
                part_logs[part] = logs
            return part_logs

        if self.text is not None:
            return self._inference_single_text()

        raise ValueError("You must provide either a dataset or a text string")

    def _inference_single_text(self):
        self.is_train = False
        self.model.eval()

        with torch.no_grad():
            if self.acoustic_model is None:
                raise ValueError("You must provide an acoustic model")

            mel = self.acoustic_model.generate(self.text)
            audio = self.model.generator(mel).squeeze(0)

        if self.save_path is not None:
            out_dir = self.save_path
            out_dir.mkdir(parents=True, exist_ok=True)

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            out_path = out_dir / f"{self.text_name}.wav"

            sample_rate = getattr(self.config, "sr", None)
            if sample_rate is None:
                sample_rate = self.cfg_trainer.get("sample_rate", 22050)

            torchaudio.save(str(out_path), audio.cpu(), sample_rate)

        return {"audio": audio}

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        text_ids = batch.get("text_id", None)

        mels = []

        if self.resynthesize:
            if "audio" not in batch:
                raise ValueError("Does not contain audio")

            for wav in batch["audio"]:
                mel = self.mel_spec(wav.to(self.device))
                mels.append(mel)
        else:
            for text in batch["text"]:
                mel = self.acoustic_model.generate(text=text)
                mels.append(mel.to(self.device))

        audios = []
        for mel in mels:
            audio = self.model.generator(mel).squeeze(0)
            audios.append(audio)

        if metrics is not None and self.metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            out_dir = self.save_path / part
            out_dir.mkdir(parents=True, exist_ok=True)

            batch_size = len(audios)
            for i in range(batch_size):
                audio = audios[i]

                if text_ids is not None:
                    base_name = str(text_ids[i])
                else:
                    base_name = f"utt_{batch_idx}_{i}"

                out_path = out_dir / f"{base_name}.wav"

                sample_rate = getattr(self.config, "sr", None)
                if sample_rate is None:
                    sample_rate = self.cfg_trainer.get("sample_rate", 22050)

                torchaudio.save(str(out_path), audio.cpu(), sample_rate)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return (
            {} if self.evaluation_metrics is None else self.evaluation_metrics.result()
        )
