import json
from speechbrain.pretrained import EncoderClassifier
from speechbrain.processing.speech_augmentation import Resample
import torchaudio
import pickle
import torch
import logging
import os

logger = logging.getLogger(__name__)


def compute_speaker_embeddings(input_filepaths, output_file_paths, data_folder, audio_sr, spk_emb_sr):
    """This function processes a JSON file to compute the speaker embeddings.
    Arguments
    ---------
    input_filepaths : list
    A list of paths to the JSON files to be processed
    output_file_paths : list
    A list of paths to the output pickle files corresponding to the input JSON files
    data_folder : str
    Path to the folder where LibriTTS data is stored
    audio_sr : int
    Sample rate of the audio files from the dataset
    spk_emb_sr : int
    Sample rate used by the speaker embedding encoder
    """

    # Checks if this phase is already done (if so, skips it)
    if skip(output_file_paths):
        logger.info("Preparation completed in previous run, skipping.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spk_emb_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                     run_opts={"device": device})
    resampler = None
    resample_audio = False
    if audio_sr != spk_emb_sr:
        resampler = Resample(orig_freq=audio_sr, new_freq=spk_emb_sr)
        resample_audio = True
        logger.info(
            f"Audio file sample rate is {audio_sr} and speaker embedding sample rate is {spk_emb_sr}.\nResampling audio files to match the sample rate required for speaker embeddings.")

    for i in range(len(input_filepaths)):
        logger.info(f"Creating {output_file_paths[i]}.")
        speaker_embeddings = dict()
        json_file = open(input_filepaths[i])
        json_data = json.load(json_file)

        for utt_id, utt_data in json_data.items():
            utt_wav_path = utt_data["wav"]
            utt_wav_path = utt_wav_path.replace("{data_root}", data_folder)
            signal, sig_sr = torchaudio.load(utt_wav_path)
            if resample_audio:
                signal = resampler(signal)
            signal = signal.to(device)

            spk_emb = spk_emb_encoder.encode_batch(signal)
            spk_emb = spk_emb.squeeze()
            spk_emb = spk_emb.detach()

            speaker_embeddings[utt_id] = spk_emb.cpu()

        with open(output_file_paths[i], "wb") as output_file:
            pickle.dump(speaker_embeddings, output_file, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Created {output_file_paths[i]}.")


def skip(filepaths):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filepath in filepaths:
        if not os.path.isfile(filepath):
            return False
    return True


if __name__ == "__main__":
    compute_speaker_embeddings(["train.json"],
                               "/content/libritts_data_sr_16000",
                               ["train_speaker_embeddings.pickle"],
                               16000,
                               16000)