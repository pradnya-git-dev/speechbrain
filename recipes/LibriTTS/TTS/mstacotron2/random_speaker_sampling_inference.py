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
import pickle
from joblib import dump, load
import numpy as np
import torch
import sklearn
import sklearn.mixture

# Load the evaluation dataset
DATA_DIR = "vctk_mstts_evaluation_dataset"
AUDIO_EXTENSION = ".wav"

# Load the required models
# Load GMM
SAVED_GMM_PATH = "/content/speechbrain/recipes/LibriTTS/TTS/mstacotron2/inference/random_speaker_generation/saved_gmm.joblib"
loaded_gmm = load(SAVED_GMM_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSTTS_MODEL_PATH = "/content/drive/MyDrive/2023/concordia/mstts_experiments/paper/saved_models/exp7_compare_with_yt/1_sr16000/2_vctk/1_mstacotron2/exp7_mstacotron2_vctk_yourtts_exp1_sr16000/e1349"

# Loads TTS model
TTS_SAMPLE_RATE = 16000
ms_tacotron2 = MSTacotron2.from_hparams(source=MSTTS_MODEL_PATH,
                                        run_opts={"device": DEVICE})

# Loads Vocoder
VOCODER_SAMPLE_RATE = 16000
VOCODER_PATH = "/content/drive/MyDrive/2023/concordia/mstts_experiments/paper/saved_models/exp8_vocoder/exp80_hifigan_vctk_sr16000"
hifi_gan = HIFIGAN.from_hparams(source=VOCODER_PATH,
                                run_opts={"device": DEVICE})

sampled_spk_embs_np, sampled_spk_embs_components = loaded_gmm.sample(200)
print("sampled_spk_embs_components: ", sampled_spk_embs_components)

# Processed the dataset one speaker at a time
# The following line works because the evaluation dataset is structured that way
for spk_dir in tqdm(glob.glob(f"{DATA_DIR}/*/*/*", recursive=True)):

    # Gets the ground truth input phoneme files:
    # Here, we already have the phoneme files computed using G2P
    # So we load phoneme files to avoid computing them every time
    tts_ground_truth_dir = os.path.join(spk_dir, "tts_ground_truth")
    tts_gt_wavs = get_all_files(tts_ground_truth_dir, match_and=[AUDIO_EXTENSION])
    
    for tts_gt_wav in tqdm(tts_gt_wavs):

      if tts_gt_wav.__contains__("_synthesized"):
        continue


      # Randomly sample a speaker embedding:
      idx = np.random.choice(sampled_spk_embs_np.shape[0], 1, replace=False)
      sampled_spk_emb_np = sampled_spk_embs_np[idx]
      ref_spk_emb = torch.from_numpy(sampled_spk_emb_np)
      ref_spk_emb = ref_spk_emb.float().to(DEVICE)
      
      # 2 Map text-to-speech - text for GT audio, spk emb for reference audio
      # 2.0 Get the phoneme input file

      # Manipulates path to get the uttid
      gt_path_parts = tts_gt_wav.split(os.path.sep)
      gt_uttid, _ = os.path.splitext(gt_path_parts[-1])

      phoneme_text_path = os.path.join(
          *gt_path_parts[:-1], gt_uttid + ".phoneme.txt"
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
