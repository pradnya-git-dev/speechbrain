import torch
import torchaudio
from fs2_pretrained_interfaces import HIFIGAN
from fs2_pretrained_interfaces import MSFastSpeech2
from spk_emb_pretrained_interfaces import MelSpectrogramEncoder
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files
import os
import torchaudio
from torch import nn
from speechbrain.pretrained import GraphemeToPhoneme
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

DEVICE = "cuda:0"

EXP_AUDIO_SR = 16000
SPK_EMB_SR = 16000
PHONEME_INPUT = False
AUDIO_OUTPUT_DIR = "/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp92_two_stage/exp925_fastspeech2_ecapa_frozen_z_sample/audio_output_b100"
if not os.path.exists(AUDIO_OUTPUT_DIR):
  os.mkdir(AUDIO_OUTPUT_DIR)

TB_LOG_DIR = "/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp92_two_stage/exp925_fastspeech2_ecapa_frozen_z_sample/inf_tensorboard_b100"
if not os.path.exists(TB_LOG_DIR):
  os.mkdir(TB_LOG_DIR)
tb_writer = SummaryWriter(TB_LOG_DIR)

g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", run_opts={"device":DEVICE})

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
fastspeech2_ms = MSFastSpeech2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/FastSpeech2/exp9_random_speaker_sampling/exp92_two_stage/exp925_fastspeech2_ecapa_frozen_z_sample_b100_ldc_sub_e300",
                                        hparams_file="/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp92_two_stage/exp925_fastspeech2_ecapa_frozen_z_sample/fastspeech2_inf_hparams.yaml",
                                        run_opts={"device": DEVICE})

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                run_opts={"device": DEVICE})

embs_list = list()
embs_labels_list = list()
common_phrase = "Mary had a little lamb."
for i in range(512):
    if PHONEME_INPUT:
      print(common_phrase)
      common_phrase_phoneme_list = g2p(common_phrase)
      common_phrase = " ".join(common_phrase_phoneme_list)
      common_phrase = "{" + common_phrase + "}"
      print(common_phrase)

    mel_output_cp, durations, pitch, energy, z_spk_emb = fastspeech2_ms.encode_text(common_phrase)
    waveform_cp = hifi_gan.decode_batch(mel_output_cp)
    cp_audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"synthesized_common_phrase{i}.wav")
    torchaudio.save(cp_audio_path, waveform_cp.squeeze(1).cpu(), EXP_AUDIO_SR)

    # print(fastspeech2_ms.hparams.random_sampler.infer())

    embs_list.append(z_spk_emb.squeeze())
    embs_labels_list.append(f"z_spk_embs_{i}")

    tb_writer.add_audio(
      f"synthesized_common_phrase_{i}.wav",
      waveform_cp,
      sample_rate=EXP_AUDIO_SR
    )

combined_embs = torch.stack(embs_list)
tb_writer.add_embedding(
  combined_embs,
  metadata=embs_labels_list,
)