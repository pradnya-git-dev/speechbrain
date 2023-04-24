# Import libraries
import pickle
from joblib import dump, load
import numpy as np
import torch
import sklearn
import sklearn.mixture
from speechbrain.pretrained import HIFIGAN
from spk_emb_pretrained_interfaces import MSTacotron2, MelSpectrogramEncoder
from speechbrain.pretrained import GraphemeToPhoneme
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# General setup
DEVICE = "cuda:0"

# Setup visualization
TB_LOG_DIR = "/content/speechbrain/recipes/LibriTTS/TTS/exp10_controllable_factors/exp102_pca/tensoboard"
if not os.path.exists(TB_LOG_DIR):
  os.mkdir(TB_LOG_DIR)
tb_writer = SummaryWriter(TB_LOG_DIR)

# Load G2P model
g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", run_opts={"device":DEVICE})

# Load TTS model
tacotron2_ms = MSTacotron2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/TTS/exp7_loss_variations/exp74_spk_emb_loss/exp742_triplet_loss/exp7421_tacotron2_ecapa/exp7421_tacotron2_ecapa_libritts_e54",
                                        hparams_file="/content/speechbrain/recipes/LibriTTS/TTS/exp10_controllable_factors/exp102_pca/tacotron2_inf_hparams.yaml",
                                        run_opts={"device": DEVICE})

# Load vocoder
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                run_opts={"device": DEVICE})

# Predict components for original speaker embeddings (LibriTTS dev-clean for now)
# Parse speaker embedding files/generate embeddings
SPK_EMB_FILES = ["train_speaker_embeddings.pickle", "valid_speaker_embeddings.pickle"]
spk_embs_dict = dict()
# We do not need labels for clustering. Storing them to double check the output later.
for spk_emb_file in SPK_EMB_FILES:
  with open(spk_emb_file, "rb") as speaker_embeddings_file:
    speaker_embeddings = pickle.load(speaker_embeddings_file)

    for (spk_emb_label, spk_emb) in speaker_embeddings.items():
      spk_id = spk_emb_label.split("_")[0]
      
      if spk_id not in spk_embs_dict.keys():
        spk_embs_dict[spk_id] = list()
      spk_embs_dict[spk_id].append(spk_emb)

log_spk_embs = list()
log_spk_embs_labels = list()
num_spks = 5
for spk_id in spk_embs_dict.keys():

  num_samples = len(spk_embs_dict[spk_id])
  if num_samples < 100:
    continue
  log_spk_embs.extend(spk_embs_dict[spk_id])
  log_spk_embs_labels.extend([int(spk_id)] * num_samples)
  num_spks -= 1
  if num_spks < 1:
    break

log_spk_embs = torch.stack(log_spk_embs)

tb_writer.add_embedding(
  log_spk_embs,
  metadata=log_spk_embs_labels,
  tag="spk_embs_before_pca"
)

N_PRINCIPAL_COMPONETS = 2
original_spk_embs = log_spk_embs.numpy()

print("Max number of principal components to try: ", N_PRINCIPAL_COMPONETS)
for N_COMPONENTS in range(1, N_PRINCIPAL_COMPONETS+1):

  print("Current number of principal components: ", N_COMPONENTS)

  pca = PCA(n_components=N_COMPONENTS) # estimate only 2 PCs
  pca_spk_embs = pca.fit_transform(original_spk_embs)

  transformed_spk_embs = torch.from_numpy(pca_spk_embs)

  tb_writer.add_embedding(
    transformed_spk_embs,
    metadata=log_spk_embs_labels,
    tag=f"spk_embs_after_{N_COMPONENTS}_pca"
  )

  # fig, axes = plt.subplots(1,2)
  # axes[0].scatter(original_spk_embs[:,0], original_spk_embs[:,1], c=log_spk_embs_labels)
  # axes[0].set_xlabel('x1')
  # axes[0].set_ylabel('x2')
  # axes[0].set_title('Before PCA')
  # axes[1].scatter(transformed_spk_embs[:,0], transformed_spk_embs[:,1], c=log_spk_embs_labels)
  # axes[1].set_xlabel('PC1')
  # axes[1].set_ylabel('PC2')
  # axes[1].set_title('After PCA')
  # plt.show()

  # fig.savefig('pca.png')

  # import pdb; pdb.set_trace()
  print("Extracting PCs and recording the effects of taking steps wrt eigenvalues.")
  pca_vectors = pca.components_
  pca_eigenvalues = pca.singular_values_

  spk_ids = ["5895", "8297"]
  spk_emb_1 = random.sample(spk_embs_dict[spk_ids[0]], k=1)[0]
  spk_emb_2 = random.sample(spk_embs_dict[spk_ids[1]], k=1)[0]


  for pci in range(len(pca_vectors)):

    print(f"Working with PC {pci + 1}")

    step_size = pca_eigenvalues[pci]
    num_of_steps = 10

    while step_size > 1:

      gradual_spk_embs = list()
      gradual_spk_embs_labels = list()

      gradual_spk_embs.append(spk_emb_1)
      gradual_spk_embs_labels.append(spk_ids[0])

      print(f"Setting step size to {step_size}")

      for i in range(num_of_steps):
        pc_change = pca_vectors[pci] + (i * step_size)

        spk_emb_grad_1 = spk_emb_1 + pc_change
        gradual_spk_embs.append(spk_emb_grad_1)
        gradual_spk_embs_labels.append(spk_ids[0] + f"_pc_{pci}_step_{step_size}_{i}")

        spk_emb_grad_2 = spk_emb_2 + pc_change
        gradual_spk_embs.append(spk_emb_grad_2)
        gradual_spk_embs_labels.append(spk_ids[1] + f"_tpc_{N_COMPONENTS}_cpc_{pci}_step_{step_size}_{i}")

      gradual_spk_embs = torch.stack(gradual_spk_embs)

      PHRASE = "Mary had a little lamb"
      common_phrase_phoneme_list = g2p(PHRASE)
      common_phrase = " ".join(common_phrase_phoneme_list)
      common_phrase = "{" + common_phrase + "}"
      print(common_phrase)

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
        tag=f"grad_se_tpc_{N_COMPONENTS}_cpc_{pci}_step_size_{step_size}"
      )

      step_size = step_size / 2

      print(f"Reduced step size to {step_size}")
      print("Running exp with the new step size")



print("DONE!")