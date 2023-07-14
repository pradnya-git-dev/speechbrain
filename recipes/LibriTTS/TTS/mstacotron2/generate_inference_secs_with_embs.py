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
import torchvision
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Writer will output to ./speaker_embedding_logs directory
writer = SummaryWriter("xvector_speaker_embedding_logs")

# Load the evaluation dataset
DATA_DIR = "mstts_evaluation_dataset_50m50f"
AUDIO_EXTENSION = ".wav"

# Load the required models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JUDGE_SPK_EMB_ENCODER_PATH = "/content/drive/MyDrive/ecapa_tdnn/vc12_mel_spec_80"
EXP_SPK_EMB_ENCODER_PATH = "/content/drive/MyDrive/xvector/vc12_mel_spec_80"
MSTTS_MODEL_PATH = "/content/drive/MyDrive/2023/concordia/mstts_experiments/paper/saved_models/exp3_spk_emb_models/ltc_sub/exp3_2_xvector"

# Loads speaker embedding model
SPK_EMB_SAMPLE_RATE = 16000
spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source=EXP_SPK_EMB_ENCODER_PATH,
                                                 run_opts={"device": DEVICE})

judge_spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source=JUDGE_SPK_EMB_ENCODER_PATH,
                                                 run_opts={"device": DEVICE})

# Loads TTS model
TTS_SAMPLE_RATE = 22050
ms_tacotron2 = MSTacotron2.from_hparams(source=MSTTS_MODEL_PATH,
                                        run_opts={"device": DEVICE})

# Loads Vocoder
VOCODER_SAMPLE_RATE = 22050
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz",
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

# Defines Cosine similarity
cos_sim_score = nn.CosineSimilarity()

# Stores SECS information
SECS = dict()

male_spk_embs, male_spk_embs_labels = list(), list()
female_spk_embs, female_spk_embs_labels = list(), list()

# Processed the dataset one speaker at a time
# The following line works because the evaluation dataset is structured that way
for spk_dir in tqdm(glob.glob(f"{DATA_DIR}/*/*/*", recursive=True)):

    if spk_dir.split("/")[-3] != "unseen_speakers":
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
    ref_spk_emb = spk_emb_encoder.encode_batch(ref_mel_spec)
    # ref_spk_emb.shape [1, 1, 192] => ref_spk_emb.shape [1, 192]
    ref_spk_emb = ref_spk_emb.squeeze(0)
    # Computing reference speaker embedding with the judge speaker embedding model
    # To be used when calculating SECS
    ref_spk_emb_for_secs = judge_spk_emb_encoder.encode_batch(ref_mel_spec).squeeze(0)


    # Manipulates path to get relative path and uttid
    path_parts = ref_wav_path.split(os.path.sep)
    ref_uttid, _ = os.path.splitext(path_parts[-1])

    # Gets the speaker-id from the utterance-id
    spk_id = ref_uttid.split("_")[0]

    if spk_id not in SECS.keys():
      SECS[spk_id] = {
        "spk_emb_ref_uttid": ref_uttid,
        "secs": dict()
      }

    # Gets the ground truth input phoneme files:
    # Here, we already have the phoneme files computed using G2P
    # So we load phoneme files to avoid computing them every time
    tts_ground_truth_dir = os.path.join(spk_dir, "tts_ground_truth")
    tts_gt_wavs = get_all_files(tts_ground_truth_dir, match_and=[AUDIO_EXTENSION])
    
    for tts_gt_wav in tqdm(tts_gt_wavs):

      if tts_gt_wav.__contains__("_synthesized"):
        continue

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

      # 1.3. Compute the speaker embedding
      # Using encode batch because - tts_gt_mel_spec.shape:  torch.Size([1, 80, x])
      tts_gt_spk_emb = judge_spk_emb_encoder.encode_batch(tts_gt_mel_spec)
      tts_gt_spk_emb = tts_gt_spk_emb.squeeze(0)

      # 2 Map text-to-speech - text for GT audio, spk emb for reference audio
      # 2.0 Get the phoneme input file

      # Manipulates path to get the uttid
      gt_path_parts = tts_gt_wav.split(os.path.sep)
      gt_uttid, _ = os.path.splitext(gt_path_parts[-1])

      phoneme_text_path = os.path.join(
          *gt_path_parts[:-1], gt_uttid + ".normalized_phoneme.txt"
      )
      with open(phoneme_text_path) as ph_f:
        phoneme_input = ph_f.read()
      phoneme_input = "{" + phoneme_input + "}"

      # 2.1 Generate speech
      mel_output_ms, mel_length_ms, alignment_ms = ms_tacotron2.encode_text(phoneme_input, ref_spk_emb)
      waveform_ms = hifi_gan.decode_batch(mel_output_ms)

      # 2.2 Save generated speech
      synthesized_audio_path = os.path.join(*gt_path_parts[:-1], gt_uttid + "_synthesized.wav")
      torchaudio.save(synthesized_audio_path, waveform_ms.squeeze(1).cpu(), TTS_SAMPLE_RATE)

      # 2.3 Compute speaker embedding for synthesized audio
      # synthesized_emb = judge_spk_emb_encoder.encode_mel_spectrogram(mel_output_ms).squeeze(0)

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
      cs_ref_spk_emb = cos_sim_score(ref_spk_emb_for_secs, synthesized_emb).item()
      cs_ground_truth = cos_sim_score(tts_gt_spk_emb, synthesized_emb).item()

      SECS[spk_id]["secs"][gt_uttid] = {
        "secs_ref_spk_emb": cs_ref_spk_emb,
        "secs_gt": cs_ground_truth
      }


      # For logging

      if spk_dir.split("/")[-2] != "male_speakers":
        male_spk_embs.append(tts_gt_spk_emb.squeeze())
        male_spk_embs_labels.append("gt_male_" + spk_id)

        male_spk_embs.append(synthesized_emb.squeeze())
        male_spk_embs_labels.append("synth_male_" + spk_id)


      if spk_dir.split("/")[-2] != "female_speakers":
        female_spk_embs.append(tts_gt_spk_emb.squeeze())
        female_spk_embs_labels.append("gt_female_" + spk_id)

        female_spk_embs.append(synthesized_emb.squeeze())
        female_spk_embs_labels.append("synth_female_" + spk_id)




      """
      gt_male_spk_embs, gt_male_spk_embs_labels = list(), list()
      synth_male_spk_embs, synth_male_spk_embs_labels = list(), list()
      gt_female_spk_embs, gt_female_spk_embs_labels = list(), list()
      synth_female_spk_embs, synth_female_spk_embs_labels = list(), list()
      """

    print(SECS[spk_id])

male_spk_embs_combined = torch.stack(male_spk_embs)
female_spk_embs_combined = torch.stack(female_spk_embs)

writer.add_embedding(
  male_spk_embs_combined, 
  metadata=male_spk_embs_labels, 
  tag="male_spk_embs"
)

writer.add_embedding(
  female_spk_embs_combined, 
  metadata=female_spk_embs_labels, 
  tag="female_spk_embs"
)


with open("SECS.json", "w") as secs_outfile:
    json.dump(SECS, secs_outfile, indent=4)

# import pdb; pdb.set_trace()

organized_SECS = {
  "seen_speakers": {
    "male_speakers": {
      "460": None, 
      "2952": None, 
      "8770": None,
      "60": None,
      "374": None
    },
    "female_speakers": {
      "8465": None,
      "2836": None,
      "6818": None,
      "5163": None,
      "8312": None
    }
  },
  "unseen_speakers": {
    "male_speakers": {
      "260": None,
      "908": None, 
      "1089": None, 
      "1188": None, 
      "2300": None
    },
    "female_speakers": {
      "121": None, 
      "237": None, 
      "2961": None, 
      "1580": None, 
      "1995": None
    }
  }
}

for spk_type in organized_SECS:
  for spk_gender in organized_SECS[spk_type]:
    for spk_id in organized_SECS[spk_type][spk_gender]:
      organized_SECS[spk_type][spk_gender][spk_id] = SECS[spk_id]

with open("organized_SECS.json", "w") as secs_outfile:
    json.dump(organized_SECS, secs_outfile, indent=4)