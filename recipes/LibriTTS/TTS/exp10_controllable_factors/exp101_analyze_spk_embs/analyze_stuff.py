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
TB_LOG_DIR = "/content/speechbrain/recipes/LibriTTS/TTS/exp10_controllable_factors/exp101_analyze_spk_embs/tensoboard"
if not os.path.exists(TB_LOG_DIR):
  os.mkdir(TB_LOG_DIR)
tb_writer = SummaryWriter(TB_LOG_DIR)

# Load G2P model
g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", run_opts={"device":DEVICE})

# Load TTS model
tacotron2_ms = MSTacotron2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/TTS/exp7_loss_variations/exp74_spk_emb_loss/exp742_triplet_loss/exp7421_tacotron2_ecapa/exp7421_tacotron2_ecapa_libritts_e32",
                                        hparams_file="/content/speechbrain/recipes/LibriTTS/TTS/exp10_controllable_factors/exp101_analyze_spk_embs/tacotron2_inf_hparams.yaml",
                                        run_opts={"device": DEVICE})

# Load vocoder
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                run_opts={"device": DEVICE})

# Predict components for original speaker embeddings (LibriTTS dev-clean for now)
# Parse speaker embedding files/generate embeddings
SPK_EMB_FILES = ["valid_speaker_embeddings.pickle"]
spk_embs_dict = dict()
# We do not need labels for clustering. Storing them to double check the output later.
spk_embs_labels = list()
for spk_emb_file in SPK_EMB_FILES:
  with open(spk_emb_file, "rb") as speaker_embeddings_file:
    speaker_embeddings = pickle.load(speaker_embeddings_file)

    for (spk_emb_label, spk_emb) in speaker_embeddings.items():
      spk_id = spk_emb_label.split("_")[0]
      
      if spk_id not in spk_embs_dict.keys():
        spk_embs_dict[spk_id] = list()
      spk_embs_dict[spk_id].append(spk_emb)


# For every sampled speaker embedding, pick 5 random original embeddings with the same component number
PHRASE = "Mary had a little lamb"
common_phrase_phoneme_list = g2p(PHRASE)
common_phrase = " ".join(common_phrase_phoneme_list)
common_phrase = "{" + common_phrase + "}"
print(common_phrase)


# import pdb; pdb.set_trace()

#spk_ids = random.sample(spk_embs_dict.keys(), k=2)
# spk_ids = ["5895(F)", "8297(M)"]
spk_ids = ["5895", "8297"]
spk_emb_1 = random.sample(spk_embs_dict[spk_ids[0]], k=1)[0]
spk_emb_2 = random.sample(spk_embs_dict[spk_ids[1]], k=1)[0]

distance = spk_emb_2 - spk_emb_1
steps = 10

gradual_spk_embs = list()
gradual_spk_embs_labels = list()

gradual_spk_embs.append(spk_emb_1)
gradual_spk_embs_labels.append(spk_ids[0])

for i in range(steps):
  spk_emb_grad = spk_emb_1 + (i/steps) * distance
  gradual_spk_embs.append(spk_emb_grad)

  gradual_spk_embs_labels.append(spk_ids[0] + f"_step_{i}")

gradual_spk_embs.append(spk_emb_2)
gradual_spk_embs_labels.append(spk_ids[1])

gradual_spk_embs = torch.stack(gradual_spk_embs)

phoneme_seqs = [common_phrase for i in range(gradual_spk_embs.shape[0])]
mel_output_ms, mel_length_ms, alignment_ms = tacotron2_ms.encode_batch(phoneme_seqs, gradual_spk_embs)
waveform_cp = hifi_gan.decode_batch(mel_output_ms)

for (a, waveform) in enumerate(waveform_cp):

  tb_writer.add_audio(
      f"{gradual_spk_embs_labels[a]}.wav",
      waveform.squeeze(1),
      sample_rate=16000
    )

tb_writer.add_embedding(
  gradual_spk_embs,
  metadata=gradual_spk_embs_labels,
  tag="gradual_spk_embs"
)


# Do the same thing with pitch
# Take 2 utterances
# Compute spk_embs and pitch for both
# Compute pitch difference
# Gradually change pitch
# Plot results and see what happens