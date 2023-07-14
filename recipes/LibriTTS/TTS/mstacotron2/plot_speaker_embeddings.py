import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
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
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Writer will output to ./speaker_embedding_logs directory
writer = SummaryWriter("speaker_embedding_logs")

# Load the evaluation dataset
DATA_DIR = "mstts_evaluation_dataset_50m50f"
AUDIO_EXTENSION = ".wav"

# Load the required models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ECAPA_SPK_EMB_ENCODER_PATH = "/content/drive/MyDrive/ecapa_tdnn/vc12_mel_spec_80"
XVECTOR_SPK_EMB_ENCODER_PATH = "/content/drive/MyDrive/xvector/vc12_mel_spec_80"

# Loads speaker embedding models
SPK_EMB_SAMPLE_RATE = 16000
ecapa_spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source=ECAPA_SPK_EMB_ENCODER_PATH,
                                                 run_opts={"device": DEVICE})

xvector_spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source=XVECTOR_SPK_EMB_ENCODER_PATH,
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

XVECTOR_EMBS = list()
ECAPA_EMBS = list()
xvector_labels = list()
ecapa_labels = list()


for spk_dir in tqdm(glob.glob(f"{DATA_DIR}/*/*/*", recursive=True)):

    if spk_dir.split("/")[-3] != "unseen":
      continue

    # Gets the reference waveforms - Here, we use only one
    spk_emb_ref_dir = os.path.join(spk_dir, "speaker_embedding_references")
    ref_wav_path = get_all_files(spk_emb_ref_dir, match_and=[AUDIO_EXTENSION])[0]
    
    # Computes the reference speaker embedding to use when generating speech
    # 0. Load the audio 
    ref_signal, signal_sr = torchaudio.load(ref_wav_path)
    # 1. Resample the signal if required:
    if signal_sr != SPK_EMB_SAMPLE_RATE:
      ref_signal = torchaudio.functional.resample(ref_signal, signal_sr, SPK_EMB_SAMPLE_RATE)
    ref_signal = ref_signal.to(DEVICE)

    # 2. Convert audio to mel-spectrogram
    ref_mel_spec = compute_mel_spectrogram(
      sample_rate=SPK_EMB_SAMPLE_RATE,
      audio=ref_signal
    )

    # 3. Compute the speaker embedding
    # Using encode batch because - ref_mel_spec.shape:  torch.Size([1, 80, x])
    ref_spk_emb_ecapa = ecapa_spk_emb_encoder.encode_batch(ref_mel_spec)
    # ref_spk_emb.shape [1, 1, 192] => ref_spk_emb.shape [1, 192]
    ref_spk_emb_ecapa = ref_spk_emb_ecapa.squeeze()
    ECAPA_EMBS.append(ref_spk_emb_ecapa)
    ecapa_labels.append("ecapa_" + spk_dir.split("/")[-2]+ "_" + spk_dir.split("/")[-1])

    ref_spk_emb_xvector = xvector_spk_emb_encoder.encode_batch(ref_mel_spec)
    # ref_spk_emb.shape [1, 1, 192] => ref_spk_emb.shape [1, 192]
    ref_spk_emb_xvector = ref_spk_emb_xvector.squeeze()
    XVECTOR_EMBS.append(ref_spk_emb_xvector)
    xvector_labels.append("xvector_" + spk_dir.split("/")[-2]+ "_" + spk_dir.split("/")[-1])


    tts_ground_truth_dir = os.path.join(spk_dir, "tts_ground_truth")
    tts_gt_wavs = get_all_files(tts_ground_truth_dir, match_and=[AUDIO_EXTENSION])
    
    for tts_gt_wav in tqdm(tts_gt_wavs):

      # 1 Computes the speaker embedding for SECS calculation
      # 1.0. Load the audio 
      tts_gt_signal, tts_gt_sig_sr = torchaudio.load(tts_gt_wav)

      # 1.1. Resample the signal if required:
      if tts_gt_sig_sr != SPK_EMB_SAMPLE_RATE:
        tts_gt_signal = torchaudio.functional.resample(tts_gt_signal, tts_gt_sig_sr, SPK_EMB_SAMPLE_RATE)
      tts_gt_signal = tts_gt_signal.to(DEVICE)

      # 1.2. Convert audio to mel-spectrogram
      tts_gt_mel_spec = compute_mel_spectrogram(
        sample_rate=SPK_EMB_SAMPLE_RATE,
        audio=tts_gt_signal
      )

      tts_gt_spk_emb_ecapa = ecapa_spk_emb_encoder.encode_batch(tts_gt_mel_spec)
      tts_gt_spk_emb_ecapa = tts_gt_spk_emb_ecapa.squeeze()
      ECAPA_EMBS.append(tts_gt_spk_emb_ecapa)
      ecapa_labels.append("ecapa_" + spk_dir.split("/")[-2]+ "_" + spk_dir.split("/")[-1])

      tts_gt_spk_emb_xvector = xvector_spk_emb_encoder.encode_batch(tts_gt_mel_spec)
      tts_gt_spk_emb_xvector = tts_gt_spk_emb_xvector.squeeze()
      XVECTOR_EMBS.append(tts_gt_spk_emb_xvector)
      xvector_labels.append("xvector_" + spk_dir.split("/")[-2]+ "_" + spk_dir.split("/")[-1])

# import pdb; pdb.set_trace()

xvector_embs_combined = torch.stack(XVECTOR_EMBS)
ecapa_embs_combined = torch.stack(ECAPA_EMBS)

writer.add_embedding(
  xvector_embs_combined, 
  metadata=xvector_labels, 
  tag="XVECTOR_EMBS"
)

writer.add_embedding(
  ecapa_embs_combined, 
  metadata=ecapa_labels, 
  tag="ECAPA_EMBS"
)

print("DONE!")