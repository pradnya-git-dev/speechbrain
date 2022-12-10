# -*- coding: utf-8 -*-
"""
 Recipe for training the Tacotron Text-To-Speech model, an end-to-end
 neural text-to-speech (TTS) system

 To run this recipe, do the following:
 # python train.py --device=cuda:0 --max_grad_norm=1.0 --data_folder=/your_folder/LJSpeech-1.1 hparams/train.yaml

 to infer simply load saved model and do
 savemodel.infer(text_Sequence,len(textsequence))

 were text_Sequence is the ouput of the text_to_sequence function from
 textToSequence.py (from textToSequence import text_to_sequence)

 Authors
 * Georges Abous-Rjeili 2021
 * Artem Ploujnikov 2021
 * Yingzhi Wang 2022
"""
import torch
import speechbrain as sb
import sys
import logging
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.text_to_sequence import text_to_sequence
from speechbrain.utils.data_utils import scalarize
import os
from speechbrain.pretrained import HIFIGAN
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import pickle
import random
from spk_emb_pretrained_interfaces import MelSpectrogramEncoder
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


class Tacotron2Brain(sb.Brain):
    """The Brain implementation for Tacotron2"""

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics"""
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0
        self.last_batch = None
        self.last_preds = None
        self.vocoder = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="tmpdir_vocoder",
            run_opts={"device": self.device},
        )

        self.spk_emb_mel_spec_encoder = MelSpectrogramEncoder.from_hparams(
          source="/workspace/mstts_saved_models/ecapa_tdnn_mel_spec_80",
          run_opts={"device": self.device},
          freeze_params=True
        )

        
        # self.spk_emb_mel_spec_encoder = MelSpectrogramEncoder.from_hparams(
        #   source="/content/drive/MyDrive/ecapa_tdnn/mel_spec_input",
        #   run_opts={"device": self.device},
        #   freeze_params=True
        # )
        

        # self.spk_emb_mel_spec_encoder.training = True
        # [param.requires_grad for param in self.spk_emb_mel_spec_encoder.parameters()] = list of all False
        # [param for param in self.spk_emb_mel_spec_encoder.parameters()] 0.0765, -0.0379,  0.0422,  0.0343
        self.last_loss_stats = {}

        self.grad_records = {
          "ave_grads": list(),
          "max_grads": list(),
          "layers": list()
        }
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
        effective_batch = self.batch_to_device(batch)
        inputs, y, num_items, _, _, spk_embs, _ = effective_batch

        _, input_lengths, _, _, _ = inputs

        max_input_length = input_lengths.max().item()

        return self.modules.model(
            inputs, spk_embs, alignments_dim=max_input_length
        )

    def fit_batch(self, batch):
        """Fits a single batch and applies annealing

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
        self.hparams.lr_annealing(self.optimizer)
        self.record_grad_flow(self.modules.model.named_parameters())
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
        effective_batch = self.batch_to_device(batch)
        # Hold on to the batch for the inference sample. This is needed because
        # the infernece sample is run from on_stage_end only, where
        # batch information is not available
        self.last_batch = effective_batch
        self.last_preds = predictions
        # Hold on to a sample (for logging)
        self._remember_sample(effective_batch, predictions)
        # Compute the loss
        loss = self._compute_loss(predictions, effective_batch, stage)
        return loss

    def _compute_loss(self, predictions, batch, stage):
        """Computes the value of the loss function and updates stats

        Arguments
        ---------
        predictions: tuple
            model predictions
        targets: tuple
            ground truth data

        Returns
        -------
        loss: torch.Tensor
            the loss value
        """
        inputs, targets, num_items, labels, wavs, spk_embs, spk_ids = batch
        text_padded, input_lengths, _, max_len, output_lengths = inputs
        
        
        self.spk_emb_mel_spec_encoder.eval()
        # Epoch 1
        # self.spk_emb_mel_spec_encoder.training = False
        # [param.requires_grad for param in self.spk_emb_mel_spec_encoder.parameters()] = list of all False
        # [param for param in self.spk_emb_mel_spec_encoder.parameters()] 0.0765, -0.0379,  0.0422,  0.0343
        
        
        target_mels = targets[0]
        pred_mels_postnet = predictions[1]

        
        param_counter = 0
        for param in self.spk_emb_mel_spec_encoder.parameters():
          if param.requires_grad:
              # print(param.data)
              param_counter = param_counter + 1
        if param_counter != 0:
          print("TEST FAILED")

        target_spk_embs = self.spk_emb_mel_spec_encoder.encode_batch(target_mels)
        target_spk_embs = target_spk_embs.squeeze().detach()
        target_spk_embs = target_spk_embs.to(self.device, non_blocking=True).float()

        pred_spk_embs = self.spk_emb_mel_spec_encoder.encode_batch(pred_mels_postnet)
        pred_spk_embs = pred_spk_embs.squeeze().detach()
        pred_spk_embs = pred_spk_embs.to(self.device, non_blocking=True).float()

        anchor_se_idx, pos_se_idx, neg_se_idx = self.get_triplets(spk_ids)
        

        spk_emb_triplets = (None, None, None)

        
        if anchor_se_idx.shape[0] != 0:

          anchor_se_idx = anchor_se_idx.to(self.device, non_blocking=True).long()
          pos_se_idx = pos_se_idx.to(self.device, non_blocking=True).long()
          neg_se_idx = neg_se_idx.to(self.device, non_blocking=True).long()
          
          anchor_spk_embs = target_spk_embs[anchor_se_idx]
          pos_spk_embs = pred_spk_embs[pos_se_idx]
          neg_spk_embs = pred_spk_embs[neg_se_idx]

          spk_emb_triplets = (anchor_spk_embs, pos_spk_embs, neg_spk_embs)
        

        loss_stats = self.hparams.criterion(
            predictions, targets, input_lengths, output_lengths, spk_emb_triplets, self.last_epoch
        )
        self.last_loss_stats[stage] = scalarize(loss_stats)
        return loss_stats.loss

    def _remember_sample(self, batch, predictions):
        """Remembers samples of spectrograms and the batch for logging purposes

        Arguments
        ---------
        batch: tuple
            a training batch
        predictions: tuple
            predictions (raw output of the Tacotron model)
        """
        inputs, targets, num_items, labels, wavs, spk_embs, spk_ids = batch
        text_padded, input_lengths, _, max_len, output_lengths = inputs
        mel_target, _ = targets
        mel_out, mel_out_postnet, gate_out, alignments = predictions
        alignments_max = (
            alignments[0]
            .max(dim=-1)
            .values.max(dim=-1)
            .values.unsqueeze(-1)
            .unsqueeze(-1)
        )
        alignments_output = alignments[0].T.flip(dims=(1,)) / alignments_max
        self.hparams.progress_sample_logger.remember(
            target=self._get_spectrogram_sample(mel_target),
            output=self._get_spectrogram_sample(mel_out),
            output_postnet=self._get_spectrogram_sample(mel_out_postnet),
            alignments=alignments_output,
            raw_batch=self.hparams.progress_sample_logger.get_batch_sample(
                {
                    "text_padded": text_padded,
                    "input_lengths": input_lengths,
                    "mel_target": mel_target,
                    "mel_out": mel_out,
                    "mel_out_postnet": mel_out_postnet,
                    "max_len": max_len,
                    "output_lengths": output_lengths,
                    "gate_out": gate_out,
                    "alignments": alignments,
                    "labels": labels,
                    "wavs": wavs,
                    "spk_embs": spk_embs,
                    "spk_ids": spk_ids,
                }
            ),
        )

    def batch_to_device(self, batch):
        """Transfers the batch to the target device

        Arguments
        ---------
        batch: tuple
            the batch to use

        Returns
        -------
        batch: tiuple
            the batch on the correct device
        """
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x,
            labels,
            wavs,
            spk_embs,
            spk_ids,
        ) = batch
        text_padded = text_padded.to(self.device, non_blocking=True).long()
        input_lengths = input_lengths.to(self.device, non_blocking=True).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.to(self.device, non_blocking=True).float()
        gate_padded = gate_padded.to(self.device, non_blocking=True).float()

        output_lengths = output_lengths.to(
            self.device, non_blocking=True
        ).long()
        x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
        y = (mel_padded, gate_padded)
        len_x = torch.sum(output_lengths)
        spk_embs = spk_embs.to(self.device, non_blocking=True).float()
        return (x, y, len_x, labels, wavs, spk_embs, spk_ids)

    def _get_spectrogram_sample(self, raw):
        """Converts a raw spectrogram to one that can be saved as an image
        sample  = sqrt(exp(raw))

        Arguments
        ---------
        raw: torch.Tensor
            the raw spectrogram (as used in the model)

        Returns
        -------
        sample: torch.Tensor
            the spectrogram, for image saving purposes
        """
        sample = raw[0]
        return torch.sqrt(torch.exp(sample))

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

        # import pdb; pdb.set_trace()

        if stage == sb.Stage.TRAIN and (
            self.hparams.epoch_counter.current % 10 == 0
        ):
            # self.last_batch = batch_to_device (x, y, len_x, original_texts, wavs, spk_embs)
            # self.last_preds = (mel_out, mel_out_postnet, gate_out, alignments)
            if self.last_batch is None:
                return

            train_sample_path = os.path.join(
                self.hparams.progress_sample_path,
                str(self.hparams.epoch_counter.current),
            )
            if not os.path.exists(train_sample_path):
                os.makedirs(train_sample_path)

            _, targets, _, labels, wavs, spk_embs, _ = self.last_batch

            # Extra lines
            # _, mel_out_postnet, _, _ = self.last_preds
            # waveform_ss = self.vocoder.decode_batch(mel_out_postnet[0])

            train_sample_text = os.path.join(
                self.hparams.progress_sample_path,
                str(self.hparams.epoch_counter.current),
                "train_input_text.txt",
            )
            with open(train_sample_text, "w") as f:
                f.write(labels[0])

            train_input_audio = os.path.join(
                self.hparams.progress_sample_path,
                str(self.hparams.epoch_counter.current),
                "train_input_audio.wav",
            )
            torchaudio.save(
                train_input_audio,
                sb.dataio.dataio.read_audio(wavs[0]).unsqueeze(0),
                self.hparams.sample_rate,
            )

            _, mel_out_postnet, _, _ = self.last_preds
            waveform_ss = self.vocoder.decode_batch(mel_out_postnet[0])
            train_sample_audio = os.path.join(
                self.hparams.progress_sample_path,
                str(self.hparams.epoch_counter.current),
                "train_output_audio.wav",
            )
            torchaudio.save(
                train_sample_audio,
                waveform_ss.squeeze(1).cpu(),
                self.hparams.sample_rate,
            )

            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_audio(
                    f"{stage}/train_audio_target",
                    sb.dataio.dataio.read_audio(wavs[0]).unsqueeze(0),
                    self.hparams.sample_rate,
                )
                self.tensorboard_logger.log_audio(
                    f"{stage}/train_audio_pred",
                    waveform_ss.squeeze(1),
                    self.hparams.sample_rate,
                )
                self.tensorboard_logger.log_figure(
                    f"{stage}/train_mel_target", targets[0][0]
                )
                self.tensorboard_logger.log_figure(
                    f"{stage}/train_mel_pred", mel_out_postnet[0]
                )

        # Store the train loss until the validation stage.

        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            lr = self.optimizer.param_groups[-1]["lr"]
            self.last_epoch = epoch

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr": lr},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )

            # The tensorboard_logger writes a summary to stdout and to the logfile.
            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_stats(
                    stats_meta={"Epoch": epoch, "lr": lr},
                    train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                    valid_stats=self.last_loss_stats[sb.Stage.VALID],
                )

            # Save the current checkpoint and delete previous checkpoints.
            epoch_metadata = {
                **{"epoch": epoch},
                **self.last_loss_stats[sb.Stage.VALID],
            }
            self.checkpointer.save_and_keep_only(
                meta=epoch_metadata,
                min_keys=["loss"],
                ckpt_predicate=(
                    lambda ckpt: (
                        ckpt.meta["epoch"]
                        % self.hparams.keep_checkpoint_interval
                        != 0
                    )
                )
                if self.hparams.keep_checkpoint_interval is not None
                else None,
            )
            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
            )
            if output_progress_sample:
                self.run_inference_sample(sb.Stage.VALID)
                self.hparams.progress_sample_logger.save(epoch)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )
            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_stats(
                    {"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=self.last_loss_stats[sb.Stage.TEST],
                )
            if self.hparams.progress_samples:
                self.run_inference_sample(sb.Stage.TEST)
                self.hparams.progress_sample_logger.save("test")

    def run_inference_sample(self, stage):
        """Produces a sample in inference mode. This is called when producing
        samples and can be useful because"""
        if self.last_batch is None:
            return
        inputs, targets, _, labels, wavs, spk_embs, _ = self.last_batch
        text_padded, input_lengths, _, _, _ = inputs

        mel_out, _, _ = self.hparams.model.infer(
            text_padded[:1], spk_embs[:1], input_lengths[:1]
        )
        self.hparams.progress_sample_logger.remember(
            inference_mel_out=self._get_spectrogram_sample(mel_out)
        )

        print(
            "INFERENCE - inference_mel_out.shape: ",
            self._get_spectrogram_sample(mel_out).shape,
        )

        if stage == sb.Stage.VALID:
            # waveform_ss = self.vocoder.decode_batch(mel_out) # Extra Line
            inf_sample_path = os.path.join(
                self.hparams.progress_sample_path,
                str(self.hparams.epoch_counter.current),
            )

            if not os.path.exists(inf_sample_path):
                os.makedirs(inf_sample_path)

            inf_sample_text = os.path.join(
                self.hparams.progress_sample_path,
                str(self.hparams.epoch_counter.current),
                "inf_input_text.txt",
            )
            with open(inf_sample_text, "w") as f:
                f.write(labels[0])

            inf_input_audio = os.path.join(
                self.hparams.progress_sample_path,
                str(self.hparams.epoch_counter.current),
                "inf_input_audio.wav",
            )
            torchaudio.save(
                inf_input_audio,
                sb.dataio.dataio.read_audio(wavs[0]).unsqueeze(0),
                self.hparams.sample_rate,
            )

            waveform_ss = self.vocoder.decode_batch(mel_out)
            inf_sample_audio = os.path.join(
                self.hparams.progress_sample_path,
                str(self.hparams.epoch_counter.current),
                "inf_output_audio.wav",
            )
            torchaudio.save(
                inf_sample_audio,
                waveform_ss.squeeze(1).cpu(),
                self.hparams.sample_rate,
            )

            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_audio(
                    f"{stage}/inf_audio_target",
                    sb.dataio.dataio.read_audio(wavs[0]).unsqueeze(0),
                    self.hparams.sample_rate,
                )
                self.tensorboard_logger.log_audio(
                    f"{stage}/inf_audio_pred",
                    waveform_ss.squeeze(1),
                    self.hparams.sample_rate,
                )
                self.tensorboard_logger.log_figure(
                    f"{stage}/inf_mel_target", targets[0][0]
                )
                self.tensorboard_logger.log_figure(
                    f"{stage}/inf_mel_pred", mel_out
                )

    def get_triplets(self, spk_ids):  
      anchor_se_idx, pos_se_idx, neg_se_idx = None, None, None
      spk_idx_pairs = list()
      for i in range(len(spk_ids)):
        for j in range(i, len(spk_ids)):
          if spk_ids[i] != spk_ids[j]:
            spk_idx_pairs.append((i, j))
      
      anchor_se_idx = torch.LongTensor([i for (i, j) in spk_idx_pairs])
      pos_se_idx = torch.LongTensor([i for (i, j) in spk_idx_pairs])
      neg_se_idx = torch.LongTensor([j for (i, j) in spk_idx_pairs])

      return (anchor_se_idx, pos_se_idx, neg_se_idx)

    def record_grad_flow(self, named_parameters):
      '''Plots the gradients flowing through different layers in the net during training.
      Can be used for checking for possible gradient vanishing / exploding problems.
      
      Usage: Plug this function in Trainer class after loss.backwards() as 
      "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

      for n, p in named_parameters:
          if(p.requires_grad) and ("bias" not in n):
              self.grad_records["layers"].append(n)
              self.grad_records["ave_grads"].append(p.grad.abs().mean().cpu())
              self.grad_records["max_grads"].append(p.grad.abs().max().cpu())
      


def plot_grad_flow(grad_records):

  max_grads = grad_records["max_grads"]
  ave_grads = grad_records["ave_grads"]
  layers = grad_records["layers"]

  plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
  plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
  plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(left=0, right=len(ave_grads))
  plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
  plt.xlabel("Layers")
  plt.ylabel("average gradient")
  plt.title("Gradient flow")
  plt.grid(True)
  plt.legend([Line2D([0], [0], color="c", lw=4),
              Line2D([0], [0], color="b", lw=4),
              Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

  plt.savefig("grad_flow_record.png", dpi=300)
  plt.show()


def dataio_prepare(hparams):
    # Define audio pipeline:

    # import pdb; pdb.set_trace()
    @sb.utils.data_pipeline.takes("wav", "label_phoneme")
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def audio_pipeline(wav, label_phoneme):

        label_phoneme = "{" + label_phoneme + "}"

        try:
            text_seq = torch.IntTensor(
                text_to_sequence(label_phoneme, hparams["text_cleaners"])
            )

            audio = sb.dataio.dataio.read_audio(wav)
            mel = hparams["mel_spectogram"](audio=audio)

            len_text = len(text_seq)

            return text_seq, mel, len_text
        except Exception as ex:
            print("FIRST EXCEPTION: ", ex)

    datasets = {}
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    try:

        for dataset in hparams["splits"]:
            datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=data_info[dataset],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline],
                output_keys=["mel_text_pair", "wav", "label", "uttid"],
            )
    except Exception as ex:
        print("SECOND EXCEPTION: ", ex)

    return datasets


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sys.path.append("../")
    from libritts_prepare import prepare_libritts

    sb.utils.distributed.run_on_main(
        prepare_libritts,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_json"],
            "save_json_valid": hparams["valid_json"],
            "save_json_test": hparams["test_json"],
            "sample_rate": hparams["sample_rate"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
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

    
    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        hparams["pretrained_separator"].collect_files()
        hparams["pretrained_separator"].load_collected()
    

    # Brain class initialization
    tacotron2_brain = Tacotron2Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    
    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in tacotron2_brain.modules.values():
            tacotron2_brain.reset_layer_recursively(module)
    
    
    if hparams["use_tensorboard"]:
        tacotron2_brain.tensorboard_logger = sb.utils.train_logger.TensorboardLogger(
            save_dir=hparams["output_folder"] + "/tensorboard"
        )

    # Training
    tacotron2_brain.fit(
        tacotron2_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    if "test" in datasets:
        tacotron2_brain.evaluate(
            datasets["test"],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )

    print("tacotron2_brain.grad_records: ", tacotron2_brain.grad_records)

    plot_grad_flow(tacotron2_brain.grad_records)