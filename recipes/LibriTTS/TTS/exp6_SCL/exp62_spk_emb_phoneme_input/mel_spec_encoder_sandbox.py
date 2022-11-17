from pretrained_interfaces import MelSpectrogramEncoder
import torch
import torchaudio
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files
import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter


DEVICE = "cuda:0"

INF_SAMPLE_DIR = "/content/libritts_test_clean_subset_sr24000"
ORIGINAL_AUDIO_SR = 24000
EXP_AUDIO_SR = 16000
SPK_EMB_SR = 16000
PHONEME_INPUT = False
TENSORBOARD_LOG_DIR = "/content/speechbrain/recipes/LibriTTS/TTS/exp6_SCL/exp62_spk_emb_phoneme_input/tensorboard_sandbox"


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectogram_exp(
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


spk_emb_waveform_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                 run_opts={"device": DEVICE})

spk_emb_mel_spec_encoder = MelSpectrogramEncoder.from_hparams(source="/content/drive/MyDrive/xvector/mel_spec_input",
                                                              hparams_file="/content/speechbrain/recipes/LibriTTS/TTS/exp6_SCL/exp62_spk_emb_phoneme_input/hyperparams.yaml",
                                                              run_opts={"device": DEVICE})
                                                 
spk_emb_resampler = Resample(orig_freq=ORIGINAL_AUDIO_SR, new_freq=SPK_EMB_SR)
mel_spec_resampler = Resample(orig_freq=ORIGINAL_AUDIO_SR, new_freq=EXP_AUDIO_SR)

writer = SummaryWriter(TENSORBOARD_LOG_DIR)

extension = [".wav"]
wav_list = get_all_files(INF_SAMPLE_DIR, match_and=extension)

spk_embs_list = list()
spk_embs_labels = list()

for wav_file in wav_list:

  if wav_file.__contains__("oracle_spec") or wav_file.__contains__("synthesized"):
    continue

  path_parts = wav_file.split(os.path.sep)
  uttid, _ = os.path.splitext(path_parts[-1])
  spk_id = uttid.split("_")[0]

  signal, fs = torchaudio.load(wav_file)

  if ORIGINAL_AUDIO_SR != SPK_EMB_SR:
    spk_emb_signal = spk_emb_resampler(signal)

  spk_emb_waveform = spk_emb_waveform_encoder.encode_batch(spk_emb_signal)
  spk_emb_waveform = spk_emb_waveform.squeeze()
  spk_embs_list.append(spk_emb_waveform)
  spk_embs_labels.append(spk_id + "_waveform")

  print("spk_emb_waveform.shape: ", spk_emb_waveform.shape)

  if ORIGINAL_AUDIO_SR != EXP_AUDIO_SR:
    mel_spec_signal = mel_spec_resampler(signal)

  oracle_mel_spec = mel_spectogram_exp(EXP_AUDIO_SR,
    256,
    1024,
    1024,
    80,
    0.0,
    8000.0,
    1,
    False,
    "slaney",
    "slaney",
    True,
    mel_spec_signal.squeeze(),)

  # mel_spec_list.append(oracle_mel_spec)
  spk_emb_mel_spec = spk_emb_mel_spec_encoder.encode_mel_spectrogram(oracle_mel_spec)
  spk_emb_mel_spec = spk_emb_mel_spec.squeeze()
  spk_embs_list.append(spk_emb_mel_spec)
  spk_embs_labels.append(spk_id + "_mel_spec")

  print("spk_emb_mel_spec.shape: ", spk_emb_mel_spec.shape)

# spk_emb_waveform_tensor = torch.stack(spk_emb_waveform_list)
# spk_emb_mel_spec_tensor = torch.stack(spk_emb_mel_spec_list)

spk_embs_tensor = torch.stack(spk_embs_list)
writer.add_embedding(spk_embs_tensor, metadata=spk_embs_labels)