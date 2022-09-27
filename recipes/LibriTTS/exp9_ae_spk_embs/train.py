#!/usr/bin/env python3
"""Recipe for training an autoencoder system.

Authors
 *
"""
import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from libritts_prepare import prepare_libritts

import torchaudio


# Brain class for autoencoder training
class AutoEncoderBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Runs all the computations that apply transformations to the provided
        input and perform reconstruction for audio tensors

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST

        Returns
        -------
        result : tuple
            A tuple (preditions, feats) that contains the reconstruction
            predictions and the original input features
        """

        # Moves the batch to the appropriate device
        batch = batch.to(self.device)

        ae_input, lens = batch.sig
        ae_input = ae_input.unsqueeze(-1)
        predictions = self.modules.autoencoder_model(ae_input, lens)

        if stage == sb.Stage.VALID:
            signal = predictions[1]
            signal = torch.transpose(signal, 0, 1).cpu()
            torchaudio.save("out.wav", signal, sample_rate=22050)

        # Returns the predictions and the original input for loss computation
        return predictions, ae_input

    def compute_objectives(self, cf_results, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        cf_results : tuple
            A tuple (preditions, feats) that contains the reconstruction
            predictions and the original input features
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        # Computes the cost function
        _, lens = batch.sig
        predictions, feats = cf_results

        # Performs padding as required
        if predictions.shape[1] < feats.shape[1]:
            predictions, _ = sb.utils.data_utils.pad_right_to(
                predictions, feats.shape)
        elif predictions.shape[1] > feats.shape[1]:
            feats, _ = sb.utils.data_utils.pad_right_to(
                feats, predictions.shape)

        # Calculates the loss
        loss = sb.nnet.losses.mse_loss(predictions, feats, lens)

        # Appends this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions, feats, lens, reduction="batch"
        )

        # Computes classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, feats, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Sets up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.mse_loss
        )

        # Sets up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
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

        # Stores the train loss until the validation stage
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarizes the statistics from the stage for record-keeping
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Saves the current checkpoint and deletes previous checkpoints
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # Writes statistics about test data to stdout and to the logfile
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect "prepare_audioMNIST" to have been called before this,
    so that the "train.json", "valid.json",  and "valid.json" manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the "hparams.yaml" file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Defines the audio input pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the "collate_fn"."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # No label pipeline is needed since the input value will be used as the label
    # for audio reconstruction

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_json"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "sig"],
        ).filtered_sorted(sort_key="length")

    return datasets


# Recipe begins!
if __name__ == "__main__":

    # Reads command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initializes ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Loads hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Creates experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process
    sb.utils.distributed.run_on_main(
        prepare_libritts,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_json"],
            "save_json_valid": hparams["valid_json"],
            "save_json_test": hparams["test_json"],
            "split_ratio": [80, 10, 10],
        },
    )

    # Creates dataset objects "train", "valid", and "test"
    datasets = dataio_prep(hparams)

    # Initializes the Brain object to prepare for mask training
    ae_brain = AutoEncoderBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The "fit()" method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    ae_brain.fit(
        epoch_counter=ae_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Loads the best checkpoint for evaluation
    test_stats = ae_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
