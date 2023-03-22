# Import libraries
import pickle
from joblib import dump, load
import numpy
import torch
import sklearn
import sklearn.mixture
from speechbrain.pretrained import HIFIGAN
from spk_emb_pretrained_interfaces import MSTacotron2, MelSpectrogramEncoder
from speechbrain.pretrained import GraphemeToPhoneme
import random
import os
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


# General setup
DEVICE = "cuda:0"

# Setup visualization
TB_LOG_DIR = "/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp94_offline_gmm/gmm_tts_tensorboard"
if not os.path.exists(TB_LOG_DIR):
  os.mkdir(TB_LOG_DIR)
tb_writer = SummaryWriter(TB_LOG_DIR)
original_spk_emb_tensors = list()

# Load G2P model
g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", run_opts={"device":DEVICE})

# Load GMM
SAVED_GMM_PATH = "/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp94_offline_gmm/saved_gmm.joblib"
loaded_gmm = load(SAVED_GMM_PATH)

# Load TTS model
tacotron2_ms = MSTacotron2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/TTS/exp7_loss_variations/exp74_spk_emb_loss/exp742_triplet_loss/exp7421_tacotron2_ecapa/exp7421_tacotron2_ecapa_libritts_e22",
                                        hparams_file="/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp94_offline_gmm/tacotron2_inf_hparams.yaml",
                                        run_opts={"device": DEVICE})

# Load vocoder
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                run_opts={"device": DEVICE})

# Predict components for original speaker embeddings (LibriTTS dev-clean for now)
# Parse speaker embedding files/generate embeddings
SPK_EMB_FILES = ["train_speaker_embeddings.pickle", "valid_speaker_embeddings.pickle"]
spk_embs = list()
# We do not need labels for clustering. Storing them to double check the output later.
spk_embs_labels = list()
for spk_emb_file in SPK_EMB_FILES:
  with open(spk_emb_file, "rb") as speaker_embeddings_file:
    speaker_embeddings = pickle.load(speaker_embeddings_file)

    for (spk_emb_label, spk_emb) in speaker_embeddings.items():
      spk_id = spk_emb_label.split("_")[0]
      spk_embs.append(spk_emb.tolist())
      spk_embs_labels.append(spk_id)

      original_spk_emb_tensors.append(spk_emb) # For visualization

spk_embs = numpy.array(spk_embs)

# Sample new speaker embeddings from GMM
sampled_spk_embs_np, sampled_spk_embs_labels_np = loaded_gmm.sample(5)

# For visualization
original_spk_emb_t = torch.stack(original_spk_emb_tensors)
sampled_spk_embs_t = torch.from_numpy(sampled_spk_embs_np)
sampled_spk_emb_gmm_labels = sampled_spk_embs_labels_np.tolist()

# Get components for original speaker embeddings
original_spk_emb_gmm_labels = loaded_gmm.predict(spk_embs).tolist()

# For every sampled speaker embedding, pick 5 random original embeddings with the same component number
PHRASE = "Mary had a little lamb"
common_phrase_phoneme_list = g2p(PHRASE)
common_phrase = " ".join(common_phrase_phoneme_list)
common_phrase = "{" + common_phrase + "}"
print(common_phrase)

n_original_spk_embs = 10

tb_embs_list, tb_embs_labels_list = list(), list()

for (i, sampled_spk_emb) in enumerate(sampled_spk_embs_t):
  valid_original_spk_embs_idx = [idx for idx in range(len(original_spk_emb_gmm_labels)) if sampled_spk_emb_gmm_labels[i] == original_spk_emb_gmm_labels[idx]]
  original_spk_embs_idx = random.sample(valid_original_spk_embs_idx, k=n_original_spk_embs)

  texts = [common_phrase for j in range(len(original_spk_embs_idx) + 1)]
  tts_spk_embs = torch.cat((sampled_spk_emb.unsqueeze(0), original_spk_emb_t[original_spk_embs_idx]))

  tb_embs_list.append(tts_spk_embs)
  tb_embs_labels_list.append(str(sampled_spk_emb_gmm_labels[i]) + "_sampled")
  tb_embs_labels_list.extend([str(sampled_spk_emb_gmm_labels[i]) + "_original" for x in range(n_original_spk_embs)])
  # import pdb; pdb.set_trace()
  # Generate speech for the common phrase "Mary had a little lamb"
  mel_output_ms, mel_length_ms, alignment_ms = tacotron2_ms.encode_batch(texts, tts_spk_embs)
  waveform_cp = hifi_gan.decode_batch(mel_output_ms)

  for (a, waveform) in enumerate(waveform_cp):

    tb_writer.add_audio(
        str(sampled_spk_emb_gmm_labels[i]) + f"_{a}.wav",
        waveform.squeeze(1),
        sample_rate=16000
      )

# import pdb; pdb.set_trace()
tb_writer.add_embedding(
  torch.cat(tb_embs_list, dim=0),
  metadata=tb_embs_labels_list,
  tag="combined_spk_embs"
)


# Generate a few other samples with using speaker embeddings computed using ECAPA TDNN encoder (use LibriTTS test-clean)
# This is not required but make sure everything is working 