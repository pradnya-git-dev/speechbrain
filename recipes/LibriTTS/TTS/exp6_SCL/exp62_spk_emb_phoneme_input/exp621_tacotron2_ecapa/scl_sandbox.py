import torch
import torchaudio
from speechbrain.pretrained import HIFIGAN
from spk_emb_pretrained_interfaces import MSTacotron2
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files
import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from spk_emb_pretrained_interfaces import MelSpectrogramEncoder

from speechbrain.pretrained import GraphemeToPhoneme
import torch

DEVICE = "cuda:0"

INF_SAMPLE_DIR = "/content/ljspeech_test_subset_sr22050"
ORIGINAL_AUDIO_SR = 22050
EXP_AUDIO_SR = 16000
SPK_EMB_SR = 16000
PHONEME_INPUT = True


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


g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", run_opts={"device":DEVICE})

spk_emb_wav_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                 run_opts={"device": DEVICE})

spk_emb_mel_spec_encoder = MelSpectrogramEncoder.from_hparams(
          source="/content/drive/MyDrive/ecapa_tdnn/mel_spec_input",
          run_opts={"device": DEVICE}
        )
                                                 
spk_emb_resampler = Resample(orig_freq=ORIGINAL_AUDIO_SR, new_freq=SPK_EMB_SR)
mel_spec_resampler = Resample(orig_freq=ORIGINAL_AUDIO_SR, new_freq=EXP_AUDIO_SR)

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2_ms = MSTacotron2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/TTS/exp3_g2p/exp31_spk_emb/exp311_tacotron2_ecapa/ljspeech_sr16000_e500",
                                        hparams_file="/content/speechbrain/recipes/LibriTTS/TTS/exp3_g2p/exp31_spk_emb/exp311_tacotron2_ecapa/tacotron2_inf_hparams.yaml",
                                        run_opts={"device": DEVICE})

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                run_opts={"device": DEVICE})

extension = [".wav"]
wav_list = get_all_files(INF_SAMPLE_DIR, match_and=extension)

for wav_file in wav_list:

  if wav_file.__contains__("oracle_spec") or wav_file.__contains__("synthesized"):
    continue

  path_parts = wav_file.split(os.path.sep)
  uttid, _ = os.path.splitext(path_parts[-1])

  signal, fs = torchaudio.load(wav_file)

  if ORIGINAL_AUDIO_SR != SPK_EMB_SR:
    spk_emb_signal = spk_emb_resampler(signal)

  spk_emb_wav = spk_emb_wav_encoder.encode_batch(spk_emb_signal)
  spk_emb_wav = spk_emb_wav.squeeze(0)

  print("Waveform-based speaker embedding shape: ", spk_emb_wav.shape)

  original_text_path = os.path.join("/", *path_parts[:-1], uttid + ".original.txt")
  with open(original_text_path) as f:
    original_text = f.read()
    if original_text.__contains__("{"):
      original_text.replace("{", "")
    if original_text.__contains__("}"):
      original_text.replace("}", "")

  if PHONEME_INPUT:
    print(original_text)
    original_text_phoneme_list = g2p(original_text)
    original_text = " ".join(original_text_phoneme_list)
    original_text = "{" + original_text + "}"
    print(original_text)

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
  
  oracle_spec_wav = hifi_gan.decode_batch(oracle_mel_spec)
  oracle_spec_wav_path = os.path.join("/", *path_parts[:-1], uttid + "_oracle_spec.wav")
  torchaudio.save(oracle_spec_wav_path, oracle_spec_wav.squeeze(1).cpu(), EXP_AUDIO_SR)

  mel_output_wav, mel_length_wav, alignment_wav = tacotron2_ms.encode_text(original_text, spk_emb_wav)
  waveform_wav = hifi_gan.decode_batch(mel_output_wav)
  print("mel_output_wav.shape: ", mel_output_wav.shape)
  synthesized_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized_wav_se.wav")
  torchaudio.save(synthesized_audio_path, waveform_wav.squeeze(1).cpu(), EXP_AUDIO_SR)

  spk_emb_mel = spk_emb_mel_spec_encoder.encode_mel_spectrogram(oracle_mel_spec)
  spk_emb_mel = spk_emb_mel.squeeze(0)

  print("Mel spectrogram-based speaker embedding shape: ", spk_emb_mel.shape)


  mel_output_mel, mel_length_mel, alignment_mel = tacotron2_ms.encode_text(original_text, spk_emb_mel)
  waveform_mel = hifi_gan.decode_batch(mel_output_mel)
  print("mel_output_mel.shape: ", mel_output_mel.shape)
  synthesized_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized_mel_se.wav")
  torchaudio.save(synthesized_audio_path, waveform_mel.squeeze(1).cpu(), EXP_AUDIO_SR)

  common_phrase = "Mary had a little lamb."
  if PHONEME_INPUT:
    print(common_phrase)
    common_phrase_phoneme_list = g2p(common_phrase)
    common_phrase = " ".join(common_phrase_phoneme_list)
    common_phrase = "{" + common_phrase + "}"
    print(common_phrase)

  mel_output_cp_wav, mel_length_ms_wav, alignment_ms_wav = tacotron2_ms.encode_text(common_phrase, spk_emb_wav)
  waveform_cp_wav = hifi_gan.decode_batch(mel_output_cp_wav)
  print("mel_output_cp_wav.shape: ", mel_output_cp_wav.shape)
  cp_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized_common_phrase_wav_se.wav")
  torchaudio.save(cp_audio_path, waveform_cp_wav.squeeze(1).cpu(), EXP_AUDIO_SR)

  mel_output_cp_mel, mel_length_ms_mel, alignment_ms_mel = tacotron2_ms.encode_text(common_phrase, spk_emb_mel)
  waveform_cp_mel = hifi_gan.decode_batch(mel_output_cp_mel)
  print("mel_output_cp_mel.shape: ", mel_output_cp_mel.shape)
  cp_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized_common_phrase_mel_se.wav")
  torchaudio.save(cp_audio_path, waveform_cp_mel.squeeze(1).cpu(), EXP_AUDIO_SR)