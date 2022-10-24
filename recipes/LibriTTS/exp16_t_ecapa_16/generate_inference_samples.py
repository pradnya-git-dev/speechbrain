import torch
import torchaudio
# from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
from pretrained_model import Tacotron2MS
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
else:
  print("RESAMPLE_FILES set to False.")


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


spk_emb_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
resampler = Resample(orig_freq=ORIGINAL_SR, new_freq=NEW_SR)

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
# tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
tacotron2_ms = Tacotron2MS.from_hparams(source="/content/drive/MyDrive/tacotron2/sb_exp16_t_ecapa_16/libritts_dev_clean_sr_16000_e23",
                                        hparams_file="/content/speechbrain/recipes/LibriTTS/exp16_t_ecapa_16/inf_hparams.yaml")

hifi_gan = HIFIGAN.from_hparams(source="/content/drive/MyDrive/hifigan/ljspeech_resampled_16",
                                hparams_file="/content/speechbrain/recipes/LibriTTS/exp16_t_ecapa_16/hifigan_inf_hparams.yaml")
extension = [".wav"]
wav_list = get_all_files(INF_SAMPLE_DIR, match_and=extension)

text_list = list()

for wav_file in wav_list:

  if wav_file.__contains__("oracle_spec") or wav_file.__contains__("synthesized"):
    continue

  path_parts = wav_file.split(os.path.sep)
  uttid, _ = os.path.splitext(path_parts[-1])

  signal, fs = torchaudio.load(wav_file)

  if RESAMPLE_FILES:
    signal = resampler(signal)

  spk_embs_list = list()
  xv_emb = spk_emb_encoder.encode_batch(signal)
  spk_embs_list.append(xv_emb.squeeze())
  spk_embs = torch.stack((spk_embs_list))

  original_text_path = os.path.join("/", *path_parts[:-1], uttid + ".original.txt")
  with open(original_text_path) as f:
    original_text = f.read()
    if original_text.__contains__("{"):
      original_text.replace("{", "")
    if original_text.__contains__("}"):
      original_text.replace("}", "")
    text_list.append(original_text)

  print("len(text_list): ", len(text_list))
  print("spk_embs.shape: ", spk_embs.shape)

  oracle_mel_spec = mel_spectogram_exp(NEW_SR,
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
    signal.squeeze(),)

  # import pdb; pdb.set_trace()
  # Running the TTS
  # mel_output_ss, mel_length_ss, alignment_ss = tacotron2.encode_text(original_text)
  mel_output_ms, mel_length_ms, alignment_ms = tacotron2_ms.encode_text(original_text, spk_embs)

  # print("mel_output_ss.shape: ", mel_output_ss.shape)
  print("mel_output_ms.shape: ", mel_output_ms.shape)

  # Running Vocoder (spectrogram-to-waveform)
  oracle_spec_wav = hifi_gan.decode_batch(oracle_mel_spec)
  # waveform_ss = hifi_gan.decode_batch(mel_output_ss)
  waveform_ms = hifi_gan.decode_batch(mel_output_ms)

  oracle_spec_wav_path = os.path.join("/", *path_parts[:-1], uttid + "_oracle_spec_16000_epoch54.wav")
  torchaudio.save(oracle_spec_wav_path, oracle_spec_wav.squeeze(1), NEW_SR)

  synthesized_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized_16000_epoch54.wav")
  torchaudio.save(synthesized_audio_path, waveform_ms.squeeze(1), NEW_SR)