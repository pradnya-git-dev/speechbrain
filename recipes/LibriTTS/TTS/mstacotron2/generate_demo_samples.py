import torch
import torchaudio
from speechbrain.pretrained import HIFIGAN
from interfaces.pretrained import MSTacotron2, MelSpectrogramEncoder
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files
import os
import torchaudio
from torch import nn
import glob
import json
from tqdm import tqdm
from speechbrain.pretrained import GraphemeToPhoneme

# Load the reference sample
REFERENCE_SAMPLE = "/content/drive/MyDrive/2023/concordia/mstts_experiments/paper/demo_recordings/pradnya_sample_recording.wav"
INPUT_TEXT = "Welcome to the presentation"

# Load the required models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JUDGE_SPK_EMB_ENCODER_PATH = "/content/drive/MyDrive/ecapa_tdnn/vc12_mel_spec_80"
EXP_SPK_EMB_ENCODER_PATH = "/content/drive/MyDrive/ecapa_tdnn/vc12_mel_spec_80"
MSTTS_MODEL_PATH = "/content/drive/MyDrive/2023/concordia/mstts_experiments/paper/saved_models/exp7_compare_with_yt/1_sr16000/2_vctk/1_mstacotron2/exp7_mstacotron2_vctk_yourtts_exp1_sr16000/e1971"

# Load the G2P model
g2p = GraphemeToPhoneme.from_hparams(
    "speechbrain/soundchoice-g2p", run_opts={"device": DEVICE}
)

# Loads speaker embedding model
SPK_EMB_SAMPLE_RATE = 16000
spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source=EXP_SPK_EMB_ENCODER_PATH,
                                                 run_opts={"device": DEVICE})

judge_spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source=JUDGE_SPK_EMB_ENCODER_PATH,
                                                 run_opts={"device": DEVICE})

# Loads TTS model
TTS_SAMPLE_RATE = 16000
ms_tacotron2 = MSTacotron2.from_hparams(source=MSTTS_MODEL_PATH,
                                        run_opts={"device": DEVICE})

# Loads Vocoder
VOCODER_SAMPLE_RATE = 16000
VOCODER_PATH = "/content/drive/MyDrive/2023/concordia/mstts_experiments/paper/saved_models/exp8_vocoder/exp80_hifigan_vctk_sr16000"
hifi_gan = HIFIGAN.from_hparams(source=VOCODER_PATH,
                                run_opts={"device": DEVICE})


## Helper functions below:

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectrogram( 
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

def compute_mel_spectrogram(
  sample_rate,
  audio,
  hop_length=256,
  win_length=1024,
  n_fft=1024,
  n_mels=80,
  f_min=0.0,
  f_max=8000.0,
  power=1,
  normalized=False,
  norm="slaney",
  mel_scale="slaney",
  compression=True
):
  mel_spec = mel_spectrogram(
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
      compression=compression,
      audio=audio
    )
  
  return mel_spec


# Helper functions end here

ref_wav_path = REFERENCE_SAMPLE

# Computes the reference speaker embedding to use when generating speech
# 0. Load the audio 
ref_signal, signal_sr = torchaudio.load(ref_wav_path)
# 1. Resample the signal if required:
if signal_sr != SPK_EMB_SAMPLE_RATE:
  print("Resampling reference speech to match the sample rate for speaker encoder.")
  ref_signal = torchaudio.functional.resample(ref_signal, signal_sr, SPK_EMB_SAMPLE_RATE)
ref_signal = ref_signal.to(DEVICE)

# 2. Convert audio to mel-spectrogram
ref_mel_spec = compute_mel_spectrogram(
  sample_rate=SPK_EMB_SAMPLE_RATE,
  audio=ref_signal
)

# 3. Compute the speaker embedding
# Using encode batch because - ref_mel_spec.shape:  torch.Size([1, 80, x])
ref_spk_emb = spk_emb_encoder.encode_batch(ref_mel_spec)
# ref_spk_emb.shape [1, 1, 192] => ref_spk_emb.shape [1, 192]
ref_spk_emb = ref_spk_emb.squeeze(0)
# Computing reference speaker embedding with the judge speaker embedding model
# To be used when calculating SECS
ref_spk_emb_for_secs = judge_spk_emb_encoder.encode_batch(ref_mel_spec).squeeze(0)

phonemes = g2p(INPUT_TEXT)
phoneme_input = " ".join(phonemes)
phoneme_input = "{" + phoneme_input + "}"

# 2.1 Generate speech
mel_output_ms, mel_length_ms, alignment_ms = ms_tacotron2.encode_text(phoneme_input, ref_spk_emb)
waveform_ms = hifi_gan.decode_batch(mel_output_ms)

# 2.2 Save generated speech
synthesized_audio_path = os.path.join("synthesized_sample.wav")
torchaudio.save(synthesized_audio_path, waveform_ms.squeeze(1).cpu(), TTS_SAMPLE_RATE)


# Compute speaker similarity score:
# 2.3.0. Load the audio 
synthesized_signal, synthesized_signal_sr = torchaudio.load(synthesized_audio_path)
# 2.3.1. Resample the signal if required:
if synthesized_signal_sr != SPK_EMB_SAMPLE_RATE:
  print("Resampling synthesized signal for the judge model.")
  synthesized_signal = torchaudio.functional.resample(synthesized_signal, synthesized_signal_sr, SPK_EMB_SAMPLE_RATE)
synthesized_signal = synthesized_signal.to(DEVICE)

# 2.3.2. Convert audio to mel-spectrogram
synthesized_mel_spec = compute_mel_spectrogram(
  sample_rate=SPK_EMB_SAMPLE_RATE,
  audio=synthesized_signal
)

# 2.3.3. Compute the speaker embedding
# Using encode batch because - ref_mel_spec.shape:  torch.Size([1, 80, x])
# After squeeze(0) ref_spk_emb.shape [1, 1, 192] => ref_spk_emb.shape [1, 192]
# Computing embedding with the judge speaker embedding model
# To be used when calculating SECS
synthesized_emb = judge_spk_emb_encoder.encode_batch(synthesized_mel_spec).squeeze(0)


# 2.4 Compute cosine similarity score w.r.t reference embedding
cos_sim_score = nn.CosineSimilarity()
cs_ref_spk_emb = cos_sim_score(ref_spk_emb_for_secs, synthesized_emb).item()

print(f"Cosine similarity score: {cs_ref_spk_emb}")