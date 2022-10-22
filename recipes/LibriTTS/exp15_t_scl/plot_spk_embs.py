from speechbrain.lobes.features import Fbank
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.speech_augmentation import Resample
import matplotlib.pyplot as plt
import torch
import json
from torchaudio import transforms
from custom_pretrained import EncoderClassifier
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter

device = "cuda:0"

fbank_maker = Fbank(n_mels=24).to(device)

audio_to_mel = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=24,
    norm="slaney",
    mel_scale="slaney"
).to(device)

xvector_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={"device": device})

writer = SummaryWriter("/content/speechbrain/recipes/LibriTTS/exp15_t_scl/spk_emb_tensorboard")

samples_file = open("train.json")
utt_dict = json.load(samples_file)


emb_list = list()
label_list = list()
for uttid, utt_info in utt_dict.items():
  print(uttid, utt_info)
  SIGNAL_PATH = utt_dict[uttid]["wav"]

  signal = read_audio(SIGNAL_PATH).unsqueeze(0).to(device)

  fbanks = fbank_maker(signal)
  fbanks_mel_spec = fbanks.squeeze(0).t().to(device)

  mel = audio_to_mel(signal)
  torchaudio_mel_spec = torch.log(torch.clamp(mel, min=1e-5) * 1)
  
  fbanks_embeddings = xvector_encoder.encode_batch_mel_specs(fbanks_mel_spec).squeeze()
  torchaudio_embeddings = xvector_encoder.encode_batch_mel_specs(torchaudio_mel_spec).squeeze()
  
  emb_list.append(fbanks_embeddings)
  label_list.append("F_" + utt_dict[uttid]["spk_id"])
  emb_list.append(torchaudio_embeddings)
  label_list.append("T_" + utt_dict[uttid]["spk_id"])

emb_tensor = torch.stack(emb_list)

writer.add_embedding(emb_tensor, metadata=label_list)

writer.close()
