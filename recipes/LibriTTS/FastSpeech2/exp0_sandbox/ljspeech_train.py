"""
 Recipe for training the FastSpeech2 Text-To-Speech model, an end-to-end
 neural text-to-speech (TTS) system introduced in 'FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
synthesis' paper
 (https://arxiv.org/abs/2006.04558)
 To run this recipe, do the following:
 # python train.py hparams/train.yaml
 Authors
 * Sathvik Udupa 2022
 * Yingzhi Wang 2022
"""

import os
import sys
import torch
import logging
import torchaudio
import numpy as np
import speechbrain as sb
# from speechbrain.pretrained import HIFIGAN
from fs2_pretrained_interfaces import HIFIGAN
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import scalarize


logger = logging.getLogger(__name__)


class FastSpeech2Brain(sb.Brain):
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics"""
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def compute_forward(self, batch, stage):
        """Computes the forward pass
        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage
        Returns
        -------
        the model output
        """
        inputs, _ = self.batch_to_device(batch)
        return self.hparams.model(*inputs)

    def fit_batch(self, batch):
        """Fits a single batch
        Arguments
        ---------
        batch: tuple
            a training batch
        Returns
        -------
        loss: torch.Tensor
            detached loss
        """
        result = super().fit_batch(batch)
        return result

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        x, y, metadata = self.batch_to_device(batch, return_metadata=True)
        self.last_batch = [x[0], y[-1], y[-2], predictions[0], *metadata]
        self._remember_sample([x[0], *y, *metadata], predictions)
        loss = self.hparams.criterion(predictions, y)
        self.last_loss_stats[stage] = scalarize(loss)
        return loss["total_loss"]

    def _remember_sample(self, batch, predictions):
        """Remembers samples of spectrograms and the batch for logging purposes
        Arguments
        ---------
        batch: tuple
            a training batch
        predictions: tuple
            predictions (raw output of the FastSpeech2
             model)
        """
        (
            tokens,
            spectogram,
            durations,
            pitch,
            energy,
            mel_lengths,
            input_lengths,
            labels,
            wavs,
        ) = batch
        (
            mel_post,
            postnet_mel_out,
            predict_durations,
            predict_pitch,
            predict_energy,
            predict_mel_lens,
        ) = predictions
        self.hparams.progress_sample_logger.remember(
            target=self.process_mel(spectogram, mel_lengths),
            output=self.process_mel(postnet_mel_out, mel_lengths),
            raw_batch=self.hparams.progress_sample_logger.get_batch_sample(
                {
                    "tokens": tokens,
                    "input_lengths": input_lengths,
                    "mel_target": spectogram,
                    "mel_out": postnet_mel_out,
                    "mel_lengths": predict_mel_lens,
                    "durations": durations,
                    "predict_durations": predict_durations,
                    "labels": labels,
                    "wavs": wavs,
                }
            ),
        )

    def process_mel(self, mel, len, index=0):
        """Converts a mel spectrogram to one that can be saved as an image
        sample  = sqrt(exp(mel))
        Arguments
        ---------
        mel: torch.Tensor
            the mel spectrogram (as used in the model)
        len: int
            length of the mel spectrogram
        index: int
            batch index
        Returns
        -------
        mel: torch.Tensor
            the spectrogram, for image saving purposes
        """
        assert mel.dim() == 3
        return torch.sqrt(torch.exp(mel[index][: len[index]]))

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            self.last_epoch = epoch
            lr = self.optimizer.param_groups[-1]["lr"]

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr": lr},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
                and epoch >= self.hparams.progress_samples_min_run
            )

            if output_progress_sample:
                logger.info("Saving predicted samples")
                inference_mel, mel_lens = self.run_inference()
                self.hparams.progress_sample_logger.save(epoch)
                self.run_vocoder(inference_mel, mel_lens)
            # Save the current checkpoint and delete previous checkpoints.
            # UNCOMMENT THIS
            self.checkpointer.save_and_keep_only(
                meta=self.last_loss_stats[stage],
                min_keys=["total_loss"],
            )
        # We also write statistics about test data spectogramto stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )

    def run_inference(self):
        """Produces a sample in inference mode with predicted durations.
        """
        if self.last_batch is None:
            return
        tokens, *_ = self.last_batch

        _, postnet_mel_out, _, _, _, predict_mel_lens =  self.hparams.model(tokens)
        self.hparams.progress_sample_logger.remember(
            infer_output=self.process_mel(postnet_mel_out, [len(postnet_mel_out[0])])
        )
        return postnet_mel_out, predict_mel_lens

    def run_vocoder(self, inference_mel, mel_lens):
        """Uses a pretrained vocoder to generate audio from predicted mel
        spectogram. By default, uses speechbrain hifigan.
        Arguments
        ---------
        inference_mel: torch.Tensor
            predicted mel from fastspeech2 inference
        mel_lens: torch.Tensor
            predicted mel lengths from fastspeech2 inference
            used to mask the noise from padding
        """
        if self.last_batch is None:
            return
        *_, wavs = self.last_batch

        inference_mel = inference_mel[: self.hparams.progress_batch_sample_size]
        mel_lens = mel_lens[0 : self.hparams.progress_batch_sample_size]
        assert (
            self.hparams.vocoder == "hifi-gan"
            and self.hparams.pretrained_vocoder is True
        ), "Specified vocoder not supported yet"
        logger.info(
            f"Generating audio with pretrained {self.hparams.vocoder_source} vocoder"
        )
        hifi_gan = HIFIGAN.from_hparams(
            source=self.hparams.vocoder_source,
            savedir=self.hparams.vocoder_download_path,
        )
        waveforms = hifi_gan.decode_batch(
            inference_mel.transpose(2, 1), mel_lens, self.hparams.hop_length
        )
        for idx, wav in enumerate(waveforms):

            path = os.path.join(
                self.hparams.progress_sample_path,
                str(self.last_epoch),
                f"pred_{Path(wavs[idx]).stem}.wav",
            )
            torchaudio.save(path, wav, self.hparams.sample_rate)

    def batch_to_device(self, batch, return_metadata=False):
        """Transfers the batch to the target device
            Arguments
            ---------
            batch: tuple
                the batch to use
            Returns
            -------
            batch: tuple
                the batch on the correct device
            """

        (
            text_padded,
            durations,
            input_lengths,
            mel_padded,
            pitch_padded,
            energy_padded,
            output_lengths,
            len_x,
            labels,
            wavs,
            spk_embs,
        ) = batch

        durations = durations.to(self.device, non_blocking=True).long()
        phonemes = text_padded.to(self.device, non_blocking=True).long()
        input_lengths = input_lengths.to(self.device, non_blocking=True).long()
        spectogram = mel_padded.to(self.device, non_blocking=True).float()
        pitch = pitch_padded.to(self.device, non_blocking=True).float()
        energy = energy_padded.to(self.device, non_blocking=True).float()
        mel_lengths = output_lengths.to(self.device, non_blocking=True).long()
        spk_embs = spk_embs.to(self.device, non_blocking=True).float()

        x = (phonemes, durations, pitch, energy)
        y = (spectogram, durations, pitch, energy, mel_lengths, input_lengths)

        metadata = (labels, wavs)
        if return_metadata:
            return x, y, metadata
        return x, y


def dataio_prepare(hparams):
    # read saved lexicon
    with open(os.path.join(hparams["save_folder"], "lexicon"), "r") as f:
        lexicon = f.read().split("\t")
    input_encoder = hparams.get("input_encoder")

    # add a dummy symbol for idx 0 - used for padding.
    lexicon = ["@@"] + lexicon
    input_encoder.update_from_iterable(lexicon, sequence_input=False)
    # load audio, text and durations on the fly; encode audio and text.

    @sb.utils.data_pipeline.takes("wav", "label", "durations", "pitch")
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def audio_pipeline(wav, label, dur, pitch):
        durs = np.load(dur)
        durs_seq = torch.from_numpy(durs).int()
        label = label.strip()
        text_seq = input_encoder.encode_sequence_torch(label.lower()).int()
        assert len(text_seq) == len(durs), f'{len(text_seq)}, {len(durs), len(label)}, ({label})'  # ensure every token has a duration
        audio = sb.dataio.dataio.read_audio(wav)
        mel, energy = hparams["mel_spectogram"](audio=audio)
        pitch = np.load(pitch)
        pitch = torch.from_numpy(pitch)
        pitch = pitch[: mel.shape[-1]]
        return text_seq, durs_seq, mel, pitch, energy, len(text_seq)

    # define splits and load it as sb dataset
    datasets = {}

    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_json"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["mel_text_pair", "wav", "label", "durations", "pitch", "uttid"],
        )
    return datasets


def main():

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sys.path.append("../")
    from ljspeech_prepare import prepare_ljspeech

    sb.utils.distributed.run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
            "duration_link": hparams["duration_link"],
            "duration_folder": hparams["duration_folder"],
            "compute_pitch": True,
            "pitch_folder": hparams["pitch_folder"],
            "pitch_n_fft": hparams["n_fft"],
            "pitch_hop_length": hparams["hop_length"],
            "pitch_min_f0": hparams["min_f0"],
            "pitch_max_f0": hparams["max_f0"],
            "skip_prep": hparams["skip_prep"],
            "create_symbol_list": True,
            "use_custom_cleaner":True,
        },
    )


    from compute_ecapa_embeddings import compute_speaker_embeddings

    sb.utils.distributed.run_on_main(
        compute_speaker_embeddings,
        kwargs={
            "input_filepaths": [hparams["train_json"], hparams["valid_json"]],
            "output_file_paths": [
                hparams["train_speaker_embeddings_pickle"],
                hparams["valid_speaker_embeddings_pickle"],
            ],
            "data_folder": hparams["data_folder"],
            "audio_sr": hparams["sample_rate"],
            "spk_emb_sr": hparams["spk_emb_sample_rate"],
            "mel_spec_params": {
              "sample_rate": hparams["sample_rate"],
              "hop_length": hparams["hop_length"],
              "win_length": hparams["win_length"],
              "n_mel_channels": hparams["n_mel_channels"],
              "n_fft": hparams["n_fft"],
              "mel_fmin": hparams["mel_fmin"],
              "mel_fmax": hparams["mel_fmax"],
              "mel_normalized": hparams["mel_normalized"],
              "power": hparams["power"],
              "norm": hparams["norm"],
              "mel_scale": hparams["mel_scale"],
              "dynamic_range_compression": hparams["dynamic_range_compression"]
            }
        },
    )
    
    datasets = dataio_prepare(hparams)

    # Brain class initialization
    fastspeech2_brain = FastSpeech2Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # Training
    fastspeech2_brain.fit(
        fastspeech2_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )


if __name__ == "__main__":
    main()