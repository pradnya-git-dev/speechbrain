import torch
import torchaudio
from speechbrain.pretrained import HIFIGAN
from spk_emb_pretrained_interfaces import MSTacotron2, MelSpectrogramEncoder
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files
import os
import torchaudio
from torch import nn
from speechbrain.utils.train_logger import plot_spectrogram
from speechbrain.pretrained import GraphemeToPhoneme
import torch
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

DEVICE = "cuda:0"

INF_SAMPLE_DIR = "/content/libritts_data/test-clean/LibriTTS/test-clean"
ORIGINAL_AUDIO_SR = 24000
EXP_AUDIO_SR = 16000
SPK_EMB_SR = 16000
PHONEME_INPUT = True

TB_LOG_DIR = "/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp92_two_stage/exp922_tacotron2_ecapa_frozen_z_sample/exp924_pt_ldc_sub_e150_inf_tensorboard"
if not os.path.exists(TB_LOG_DIR):
  os.mkdir(TB_LOG_DIR)
tb_writer = SummaryWriter(TB_LOG_DIR)

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectogram_exp(
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


g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", run_opts={"device":DEVICE})

spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source="/content/drive/MyDrive/ecapa_tdnn/vc12_mel_spec_80",
                                                 run_opts={"device": DEVICE})
                                                 
exp_resampler = Resample(orig_freq=ORIGINAL_AUDIO_SR, new_freq=EXP_AUDIO_SR)

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2_ms = MSTacotron2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/TTS/exp9_random_speaker_sampling/exp92_two_stage/exp924_tacotron2_ecapa_frozen_z_triplet/exp924_tacotron2_ecapa_frozen_z_triplet_ldc_sub_e150",
                                        hparams_file="/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp92_two_stage/exp922_tacotron2_ecapa_frozen_z_sample/tacotron2_inf_hparams.yaml",
                                        run_opts={"device": DEVICE})

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                run_opts={"device": DEVICE})

extension = [".wav"]
wav_list = get_all_files(INF_SAMPLE_DIR, match_and=extension)

cos_sim_score = nn.CosineSimilarity()
COS_SIM_SCORES_FILE = INF_SAMPLE_DIR + "/cos_sim_scores.txt"
s_pre_spk_embs, s_z_spk_embs, s_post_spk_embs, s_z_means, s_mlp_outs = [], [], [], [], []
s_pre_spk_embs_labels, s_z_spk_embs_labels, s_post_spk_embs_labels, s_z_means_labels, s_mlp_outs_labels = [], [], [], [], []

same_spk_diff_phrases = [
  "This is phrase one",
  "Another random sentence for phase two",
  "Coming up with new samples is not very easy for some reason",
  "Roses are red",
  "Violets are blue",
  "Can you please think of something new"
]

processed_spk_ids = []

with open(COS_SIM_SCORES_FILE, "w+") as cs_f:
  sample_counter = 0

  for wav_file in wav_list:

    sample_counter += 1
    if sample_counter > 1000:
      break

    if wav_file.__contains__("oracle_spec") or wav_file.__contains__("synthesized"):
      continue

    path_parts = wav_file.split(os.path.sep)
    uttid, _ = os.path.splitext(path_parts[-1])

    spk_id = uttid.split("_")[0]
    # if spk_id in processed_spk_ids:
    #   continue
    # processed_spk_ids.append(spk_id)

    signal, fs = torchaudio.load(wav_file)

    if fs != EXP_AUDIO_SR:
      signal = exp_resampler(signal)


    phoneme_text_path = os.path.join(
        "/", *path_parts[:-1], uttid + ".normalized_phoneme.txt"
    )
    if os.path.exists(phoneme_text_path):
      with open(phoneme_text_path) as f:
        original_text = f.read()
    else:
      original_text_path = os.path.join("/", *path_parts[:-1], uttid + ".normalized.txt")
      with open(original_text_path) as f:
        original_text = f.read()
        if original_text.__contains__("{"):
          original_text.replace("{", "")
        if original_text.__contains__("}"):
          original_text.replace("}", "")

      if PHONEME_INPUT:
        print(original_text)
        original_text_phoneme_list = g2p(original_text)
        original_text = " ".join(original_text_phoneme_list)
        original_text = "{" + original_text + "}"
        print(original_text)

    oracle_mel_spec = mel_spectogram_exp(EXP_AUDIO_SR,
      256,
      1024,
      1024,
      80,
      0.0,
      8000.0,
      1,
      False,
      "slaney",
      "slaney",
      True,
      signal.squeeze(),)
    
    oracle_spec_wav = hifi_gan.decode_batch(oracle_mel_spec)
    oracle_spec_wav_path = os.path.join("/", *path_parts[:-1], uttid + "_oracle_spec.wav")
    torchaudio.save(oracle_spec_wav_path, oracle_spec_wav.squeeze(1).cpu(), EXP_AUDIO_SR)

    spk_emb = spk_emb_encoder.encode_mel_spectrogram(oracle_mel_spec)
    spk_emb = spk_emb.squeeze(0)

    print("Speaker embedding shape: ", spk_emb.shape)

    mel_output_ms, mel_length_ms, alignment_ms, z_spk_embs, z_mean, mlp_out = tacotron2_ms.encode_text(original_text, spk_emb)
    waveform_ms = hifi_gan.decode_batch(mel_output_ms)
    print("mel_output_ms.shape: ", mel_output_ms.shape)
    synthesized_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized.wav")
    torchaudio.save(synthesized_audio_path, waveform_ms.squeeze(1).cpu(), EXP_AUDIO_SR)

    tb_writer.add_audio(
        uttid + f"_synthesized.wav",
        waveform_ms.squeeze(1),
        sample_rate=EXP_AUDIO_SR
      )

    post_spk_emb = spk_emb_encoder.encode_mel_spectrogram(mel_output_ms).squeeze(0)

    s_pre_spk_embs.append(spk_emb.squeeze())
    s_pre_spk_embs_labels.append(spk_id)

    s_mlp_outs.append(mlp_out.squeeze())
    s_mlp_outs_labels.append(spk_id)

    s_z_means.append(z_mean.squeeze())
    s_z_means_labels.append(spk_id)

    s_z_spk_embs.append(z_spk_embs.squeeze())
    s_z_spk_embs_labels.append(spk_id)

    s_post_spk_embs.append(post_spk_emb.squeeze())
    s_post_spk_embs_labels.append(spk_id)

    """
    for (i, phrase) in enumerate(same_spk_diff_phrases):
      if PHONEME_INPUT:
        print(phrase)
        phrase_phoneme_list = g2p(phrase)
        phrase = " ".join(phrase_phoneme_list)
        phrase = "{" + phrase + "}"
        print(phrase)

      mel_output_ms, mel_length_ms, alignment_ms, z_spk_embs = tacotron2_ms.encode_text(phrase, spk_emb)
      waveform_ms = hifi_gan.decode_batch(mel_output_ms)
      print("mel_output_ms.shape: ", mel_output_ms.shape)
      synthesized_audio_path = os.path.join("/", *path_parts[:-1], spk_id + f"_phrase{i}_synthesized.wav")
      torchaudio.save(synthesized_audio_path, waveform_ms.squeeze(1).cpu(), EXP_AUDIO_SR)


      tb_writer.add_audio(
        spk_id + f"_phrase{i}_synthesized.wav",
        waveform_ms.squeeze(1),
        sample_rate=EXP_AUDIO_SR
      )

      post_spk_emb = spk_emb_encoder.encode_mel_spectrogram(mel_output_ms).squeeze(0)

      s_pre_spk_embs.append(spk_emb.squeeze())
      s_pre_spk_embs_labels.append(spk_id)

      s_z_spk_embs.append(z_spk_embs.squeeze())
      s_z_spk_embs_labels.append(spk_id)

      s_post_spk_embs.append(post_spk_emb.squeeze())
      s_post_spk_embs_labels.append(spk_id)
    """
    
# import pdb; pdb.set_trace()
combined_s_pre_spk_embs = torch.stack(s_pre_spk_embs)
combined_s_mlp_outs = torch.stack(s_mlp_outs)
combined_s_z_means = torch.stack(s_z_means)
combined_s_z_spk_embs = torch.stack(s_z_spk_embs)
combined_s_post_spk_embs = torch.stack(s_post_spk_embs)

tb_writer.add_embedding(
  combined_s_pre_spk_embs,
  metadata=s_pre_spk_embs_labels,
  tag="same_speaker_pre_spk_embs"
)

tb_writer.add_embedding(
  combined_s_mlp_outs,
  metadata=s_mlp_outs_labels,
  tag="same_speaker_mlp_outs"
)

tb_writer.add_embedding(
  combined_s_z_means,
  metadata=s_z_means_labels,
  tag="same_speaker_z_means"
)

tb_writer.add_embedding(
  combined_s_z_spk_embs,
  metadata=s_z_spk_embs_labels,
  tag="same_speaker_z_spk_embs"
)

tb_writer.add_embedding(
  combined_s_post_spk_embs,
  metadata=s_post_spk_embs_labels,
  tag="same_speaker_post_spk_embs"
)

combined_z_mean_z_spk_embs = torch.cat((combined_s_z_means, combined_s_z_spk_embs))
s_z_means_labels = [label + "_z_mean" for label in s_z_means_labels]
s_z_spk_embs_labels = [label + "_z_spk_embs" for label in s_z_spk_embs_labels]
s_z_means_labels.extend(s_z_spk_embs_labels)
combined_z_mean_z_spk_embs_labels = s_z_means_labels

tb_writer.add_embedding(
  combined_z_mean_z_spk_embs,
  metadata=combined_z_mean_z_spk_embs_labels,
  tag="same_speaker_z_mean_z_spk_embs_relative"
)