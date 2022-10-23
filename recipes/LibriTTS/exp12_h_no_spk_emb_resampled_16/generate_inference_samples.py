import torch
import torchaudio
from speechbrain.pretrained import HIFIGAN
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files
import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier

INF_SAMPLE_DIR = "/content/libritts_data_inference_22050"
ORIGINAL_SR = 22050
NEW_SR = 16000
RESAMPLE_FILES = False
if ORIGINAL_SR != NEW_SR:
  RESAMPLE_FILES = True
  print("RESAMPLE_FILES set to True.")

resampler = Resample(orig_freq=ORIGINAL_SR, new_freq=NEW_SR)

hifi_gan = HIFIGAN.from_hparams(source="/content/drive/MyDrive/hifigan/libritts_train_clean_resampled_16_e100",
                                hparams_file="/content/speechbrain/recipes/LibriTTS/exp12_h_no_spk_emb_resampled_16/hifigan_inf_hparams.yaml")


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


extension = [".wav"]
wav_list = get_all_files(INF_SAMPLE_DIR, match_and=extension)

for wav_file in wav_list:

  print(wav_file)

  path_parts = wav_file.split(os.path.sep)
  uttid, _ = os.path.splitext(path_parts[-1])

  signal, fs =torchaudio.load(wav_file)

  if RESAMPLE_FILES:
    signal = resampler(signal)

  mel_spec = mel_spectogram(
    NEW_SR,
    256,
    1024,
    1024,
    80,
    0.0,
    8000,
    1,
    None,
    "slaney",
    "slaney",
    True,
    audio=signal,
  )
  
  # Running Vocoder (spectrogram-to-waveform)
  waveform = hifi_gan.decode_batch(mel_spec)

  # Save the waverform
  synthesized_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized.wav")
  torchaudio.save(synthesized_audio_path, waveform.squeeze(1), NEW_SR)