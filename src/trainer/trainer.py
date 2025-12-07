from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        true_audio = batch["audio"]
        spectrogram = batch["spectrogram"]

        fake_audio = self.model.generator(spectrogram)[:, :, : true_audio.shape[-1]]
        batch["fake_audio"] = fake_audio.squeeze(1)

        # discriminator
        true_mpd_out, true_mpd_activations = self.model.mpd(true_audio)
        fake_mpd_out, fake_mpd_activations = self.model.mpd(fake_audio.detach())

        true_msd_out, true_msd_activations = self.model.msd(true_audio)
        fake_msd_out, fake_msd_activations = self.model.msd(fake_audio.detach())

        loss_mpd = self.criterion.discriminator_loss(fake_mpd_out, true_mpd_out)
        loss_msd = self.criterion.discriminator_loss(fake_msd_out, true_msd_out)
        d_loss = loss_mpd["loss"] + loss_msd["loss"]

        batch.update(
            {
                "d_loss_mpd": loss_mpd["loss"],
                "d_loss_msd": loss_msd["loss"],
                "d_loss": d_loss,
            }
        )

        if self.is_train:
            self.optimizer_d.zero_grad()
            d_loss.backward()
            self._clip_grad_norm()
            self.optimizer_d.step()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

        # generator
        true_mpd_out, true_mpd_activations = self.model.mpd(true_audio)
        true_msd_out, true_msd_activations = self.model.msd(true_audio)

        fake_mpd_out, fake_mpd_activations = self.model.mpd(fake_audio)
        fake_msd_out, fake_msd_activations = self.model.msd(fake_audio)

        discriminators_fake = fake_mpd_out + fake_msd_out

        features_fake = fake_mpd_activations + fake_msd_activations
        features_true = true_mpd_activations + true_msd_activations

        mel_fake = self.instance_transforms["train"]["get_spectrogram"](
            batch["fake_audio"]
        )
        batch["fake_spectrogram"] = mel_fake.squeeze(1)

        mel_true = batch["spectrogram"]

        g_losses = self.criterion.generator_loss(
            discriminators_fake, features_fake, features_true, mel_fake, mel_true
        )
        batch.update(g_losses)

        if self.is_train:
            self.optimizer_g.zero_grad()
            g_losses["g_loss"].backward()
            self._clip_grad_norm()
            self.optimizer_g.step()
            if self.lr_scheduler_g is not None:
                self.lr_scheduler_g.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(batch["spectrogram"], "true_spectrogram")
            self.log_spectrogram(batch["fake_spectrogram"], "fake_spectrogram")
            self.log_audio(audio_name="true_audio", audio=batch["audio"][0])
            self.log_audio(audio_name="fake_audio", audio=batch["fake_audio"][0])
        else:
            self.log_spectrogram(batch["spectrogram"], "true_spectrogram")
            self.log_spectrogram(batch["fake_spectrogram"], "fake_spectrogram")
            self.log_audio(audio_name="true_audio", audio=batch["audio"][0])
            self.log_audio(audio_name="fake_audio", audio=batch["fake_audio"][0])

    def log_spectrogram(self, spectrogram, spectrogram_name="spectrogram", **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image(spectrogram_name, image)

    def log_audio(self, audio_name, audio, **batch):
        self.writer.add_audio(audio_name, audio, 22050)
