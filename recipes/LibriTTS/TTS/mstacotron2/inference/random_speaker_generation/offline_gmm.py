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
TB_LOG_DIR = "default_mean_offline_gmm_tensorboard"
if not os.path.exists(TB_LOG_DIR):
  os.mkdir(TB_LOG_DIR)
tb_writer = SummaryWriter(TB_LOG_DIR)

# Parse speaker embedding files/generate embeddings
SPK_EMB_FILES = ["train_speaker_embeddings.pickle", "valid_speaker_embeddings.pickle"]
train_spk_embs = list()
validation_spk_embs = list()
# We do not need labels for clustering. Storing them to double check the output later.
train_spk_embs_labels = list()
validation_spk_embs_labels = list()

spk_embs_dict = dict()
for spk_emb_file in SPK_EMB_FILES:
  with open(spk_emb_file, "rb") as speaker_embeddings_file:
    speaker_embeddings = pickle.load(speaker_embeddings_file)

    for (spk_emb_label, spk_emb) in speaker_embeddings.items():
      spk_id = spk_emb_label.split("_")[0]

      if spk_id not in spk_embs_dict.keys():
        spk_embs_dict[spk_id] = list()
      spk_embs_dict[spk_id].append(spk_emb.numpy())

custom_means = list()
custom_means_labels = list()
n_components = 0

for spk_id in spk_embs_dict.keys():
  mean_spk_emb = numpy.mean(numpy.array(spk_embs_dict[spk_id]), axis=0)
  custom_means.append(mean_spk_emb)
  custom_means_labels.append(spk_id + "_mean")
  num_valid_embs = 5 if 15 < len(spk_embs_dict[spk_id]) else 0
  if num_valid_embs > 0:
    validation_spk_embs.extend(spk_embs_dict[spk_id][:num_valid_embs])
    validation_spk_embs_labels.extend([spk_id + "_valid" for i in range(num_valid_embs)])

  num_train_embs = len(spk_embs_dict[spk_id]) - num_valid_embs
  if num_train_embs >= 1:
    n_components += 1
  train_spk_embs.extend(spk_embs_dict[spk_id][num_valid_embs:])
  train_spk_embs_labels.extend([spk_id + "_train" for i in range(num_train_embs)])
  
train_spk_embs = numpy.array(train_spk_embs)
validation_spk_embs = numpy.array(validation_spk_embs)
custom_means = numpy.array(custom_means)

print("Number of components: ", n_components)

# Train GMM
# gmm = sklearn.mixture.GaussianMixture(n_components=40, n_init=5, means_init=custom_means, random_state=0, verbose=2).fit(train_spk_embs)
gmm = sklearn.mixture.GaussianMixture(n_components=n_components, n_init=5, random_state=0, verbose=2).fit(train_spk_embs)
custom_means_gmm_labels = gmm.predict(custom_means).tolist()
train_spk_emb_gmm_labels = gmm.predict(train_spk_embs).tolist()
validation_spk_embs_gmm_labels = gmm.predict(validation_spk_embs).tolist()

for (i, label) in enumerate(custom_means_labels):
  custom_means_labels[i] = custom_means_labels[i] + f"_gmm_{custom_means_gmm_labels[i]}"

for (i, label) in enumerate(train_spk_embs_labels):
  train_spk_embs_labels[i] = train_spk_embs_labels[i] + f"_gmm_{train_spk_emb_gmm_labels[i]}"

for (i, label) in enumerate(validation_spk_embs_labels):
  validation_spk_embs_labels[i] = validation_spk_embs_labels[i] + f"_gmm_{validation_spk_embs_gmm_labels[i]}"

# Sample speaker embeddings from the GMM
sampled_spk_embs_np, sampled_spk_embs_labels_np = gmm.sample(50)

# Visualization
custom_means_t = torch.from_numpy(custom_means)
train_spk_emb_t = torch.from_numpy(train_spk_embs)
validation_spk_emb_t = torch.from_numpy(validation_spk_embs)
sampled_spk_embs_t = torch.from_numpy(sampled_spk_embs_np)
sampled_spk_emb_gmm_labels = sampled_spk_embs_labels_np.tolist()


tb_writer.add_embedding(
  custom_means_t,
  metadata=custom_means_labels,
  tag="train_spk_emb_custom_means"
)

tb_writer.add_embedding(
  train_spk_emb_t,
  metadata=train_spk_embs_labels,
  tag="train_spk_emb_gmm_clusters"
)

tb_writer.add_embedding(
  validation_spk_emb_t,
  metadata=validation_spk_embs_labels,
  tag="validation_spk_emb_gmm_clusters"
)

tb_writer.add_embedding(
  sampled_spk_embs_t,
  metadata=sampled_spk_emb_gmm_labels,
  tag="sampled_spk_emb_gmm_clusters"
)

combined_train_valid = torch.cat((custom_means_t, train_spk_emb_t, validation_spk_emb_t))
combined_train_valid_labels = list()
combined_train_valid_labels.extend(custom_means_labels)
combined_train_valid_labels.extend(train_spk_embs_labels)
combined_train_valid_labels.extend(validation_spk_embs_labels)
tb_writer.add_embedding(
  combined_train_valid,
  metadata=combined_train_valid_labels,
  tag="combined_train_valid_gmm_clusters"
)


combined_spk_embs = torch.cat((custom_means_t, train_spk_emb_t, validation_spk_emb_t, sampled_spk_embs_t))
combined_spk_emb_gmm_labels = list()
combined_spk_emb_gmm_labels.extend(custom_means_labels)
combined_spk_emb_gmm_labels.extend(train_spk_embs_labels)
combined_spk_emb_gmm_labels.extend(validation_spk_embs_labels)
combined_spk_emb_gmm_labels.extend([str(label) + "_sampled" for label in sampled_spk_emb_gmm_labels])

tb_writer.add_embedding(
  combined_spk_embs,
  metadata=combined_spk_emb_gmm_labels,
  tag="combined_spk_emb_gmm_clusters"
)

# Save the trained GMM model
SAVED_GMM_PATH = "saved_gmm.joblib"
dump(gmm, SAVED_GMM_PATH) 

# Test speech synthesis with the new GMM