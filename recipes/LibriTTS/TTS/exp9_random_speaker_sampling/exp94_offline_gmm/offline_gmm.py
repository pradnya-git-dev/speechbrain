# Import libraries
import pickle
from joblib import dump, load
import numpy
import torch
import sklearn
import sklearn.mixture
import os
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Setup visualization
TB_LOG_DIR = "/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp94_offline_gmm/offline_gmm_tensorboard"
if not os.path.exists(TB_LOG_DIR):
  os.mkdir(TB_LOG_DIR)
tb_writer = SummaryWriter(TB_LOG_DIR)
original_spk_emb_tensors = list()

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

# Train GMM
gmm = sklearn.mixture.GaussianMixture(n_components=40, n_init=5, random_state=0).fit(spk_embs)
original_spk_emb_gmm_labels = gmm.predict(spk_embs).tolist()

# Sample speaker embeddings from the GMM
sampled_spk_embs_np, sampled_spk_embs_labels_np = gmm.sample(50)

# Visualization
original_spk_emb_t = torch.stack(original_spk_emb_tensors)
sampled_spk_embs_t = torch.from_numpy(sampled_spk_embs_np)
sampled_spk_emb_gmm_labels = sampled_spk_embs_labels_np.tolist()

tb_writer.add_embedding(
  original_spk_emb_t,
  metadata=original_spk_emb_gmm_labels,
  tag="original_spk_emb_gmm_clusters"
)

tb_writer.add_embedding(
  sampled_spk_embs_t,
  metadata=sampled_spk_emb_gmm_labels,
  tag="sampled_spk_emb_gmm_clusters"
)

combined_spk_embs = torch.cat((original_spk_emb_t, sampled_spk_embs_t))
combined_spk_emb_gmm_labels = [str(label) + "_original" for label in original_spk_emb_gmm_labels]
combined_spk_emb_gmm_labels.extend([str(label) + "_sampled" for label in sampled_spk_emb_gmm_labels])

tb_writer.add_embedding(
  combined_spk_embs,
  metadata=combined_spk_emb_gmm_labels,
  tag="combined_spk_emb_gmm_clusters"
)

# Save the trained GMM model
SAVED_GMM_PATH = "/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp94_offline_gmm/saved_gmm.joblib"
dump(gmm, SAVED_GMM_PATH) 

# Test speech synthesis with the new GMM