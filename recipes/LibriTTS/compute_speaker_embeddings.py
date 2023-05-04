import json
from speechbrain.pretrained import EncoderClassifier
import torchaudio
import pickle
import torch
import logging
import os
from interfaces.pretrained import MelSpectrogramEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectogram( 
    sample_rate,
    hop_length,
    win_length,
    n_fft,
    n_mels,
    f_min,
    f_max,
    power,
    normalized,
    norm,
    mel_scale,
    compression,
    audio,
):
    """calculates MelSpectrogram for a raw audio signal

    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_fft : int
        Size of FFT.
    n_mels : int
        Number of mel filterbanks.
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    power : float
        Exponent for the magnitude spectrogram.
    normalized : bool
        Whether to normalize by magnitude after stft.
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    compression : bool
        whether to do dynamic range compression
    audio : torch.tensor
        input audio signal
    """
    from torchaudio import transforms

    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
        normalized=normalized,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)

    mel = audio_to_mel(audio)

    if compression:
        mel = dynamic_range_compression(mel)

    return mel


def compute_speaker_embeddings(input_filepaths, output_file_paths, data_folder, spk_emb_encoder_path, audio_sr, spk_emb_sr, mel_spec_params):
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

    ENCODER_PATH =  spk_emb_encoder_path

    spk_emb_encoder = MelSpectrogramEncoder.from_hparams(
          source=ENCODER_PATH,
          run_opts={"device": device}
        )

    for i in range(len(input_filepaths)):
        logger.info(f"Creating {output_file_paths[i]}.")
        speaker_embeddings = dict()
        json_file = open(input_filepaths[i])
        json_data = json.load(json_file)

        for utt_id, utt_data in tqdm(json_data.items()):
            utt_wav_path = utt_data["wav"]
            utt_wav_path = utt_wav_path.replace("{data_root}", data_folder)
            signal, sig_sr = torchaudio.load(utt_wav_path)
            if sig_sr != spk_emb_sr:
                signal = torchaudio.functional.resample(signal, sig_sr, spk_emb_sr)
            signal = signal.to(device)

            mel_spec = mel_spectogram(
              sample_rate=mel_spec_params["sample_rate"],
              hop_length=mel_spec_params["hop_length"],
              win_length=mel_spec_params["win_length"],
              n_fft=mel_spec_params["n_fft"],
              n_mels=mel_spec_params["n_mel_channels"],
              f_min=mel_spec_params["mel_fmin"],
              f_max=mel_spec_params["mel_fmax"],
              power=mel_spec_params["power"],
              normalized=mel_spec_params["mel_normalized"],
              norm=mel_spec_params["norm"],
              mel_scale=mel_spec_params["mel_scale"],
              compression=mel_spec_params["dynamic_range_compression"],
              audio=signal
            )

            spk_emb = spk_emb_encoder.encode_batch(mel_spec)
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