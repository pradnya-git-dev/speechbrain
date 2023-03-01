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

INF_SAMPLE_DIR = "/content/libritts_test_clean_subset_sr24000"
ORIGINAL_AUDIO_SR = 24000
EXP_AUDIO_SR = 16000
SPK_EMB_SR = 16000
PHONEME_INPUT = False
TB_LOG_DIR = "/content/speechbrain/recipes/LibriTTS/TTS/exp9_random_speaker_sampling/exp92_two_stage/exp925_fastspeech2_ecapa_frozen_z_sample/inf_tensorboard"
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
                                                 
spk_emb_resampler = Resample(orig_freq=ORIGINAL_AUDIO_SR, new_freq=SPK_EMB_SR)
mel_spec_resampler = Resample(orig_freq=ORIGINAL_AUDIO_SR, new_freq=EXP_AUDIO_SR)

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
fastspeech2_ms = MSFastSpeech2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/TTS/exp7_loss_variations/exp74_spk_emb_loss/exp742_triplet_loss/exp7423_fastspeech2_ecapa/exp7423_fastspeech2_ecapa_ldc_sub_e300",
                                        hparams_file="/content/speechbrain/recipes/LibriTTS/TTS/exp7_loss_variations/exp74_spk_emb_loss/exp742_triplet_loss/exp7423_fastspeech2_ecapa/fastspeech2_inf_hparams.yaml",
                                        run_opts={"device": DEVICE})

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                run_opts={"device": DEVICE})

extension = [".wav"]
wav_list = get_all_files(INF_SAMPLE_DIR, match_and=extension)

cos_sim_score = nn.CosineSimilarity()
COS_SIM_SCORES_FILE = INF_SAMPLE_DIR + "/cos_sim_scores.txt"

embs_list = list()
embs_labels_list = list()
with open(COS_SIM_SCORES_FILE, "w+") as cs_f:

  for wav_file in wav_list:

    if wav_file.__contains__("oracle_spec") or wav_file.__contains__("synthesized"):
      continue

    path_parts = wav_file.split(os.path.sep)
    uttid, _ = os.path.splitext(path_parts[-1])

    signal, fs = torchaudio.load(wav_file)

    if ORIGINAL_AUDIO_SR != SPK_EMB_SR:
      spk_emb_signal = spk_emb_resampler(signal)

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

    if ORIGINAL_AUDIO_SR != EXP_AUDIO_SR:
      mel_spec_signal = mel_spec_resampler(signal)

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
      mel_spec_signal.squeeze(),)
    
    oracle_spec_wav = hifi_gan.decode_batch(oracle_mel_spec)
    oracle_spec_wav_path = os.path.join("/", *path_parts[:-1], uttid + "_oracle_spec.wav")
    torchaudio.save(oracle_spec_wav_path, oracle_spec_wav.squeeze(1).cpu(), EXP_AUDIO_SR)

    spk_emb = spk_emb_encoder.encode_mel_spectrogram(oracle_mel_spec)
    spk_emb = spk_emb.squeeze(0)

    print("Speaker embedding shape: ", spk_emb.shape)

    mel_output_ms, durations, pitch, energy, z_spk_emb = fastspeech2_ms.encode_text(original_text, spk_emb)
    waveform_ms = hifi_gan.decode_batch(mel_output_ms)
    print("mel_output_ms.shape: ", mel_output_ms.shape)
    synthesized_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized.wav")
    torchaudio.save(synthesized_audio_path, waveform_ms.squeeze(1).cpu(), EXP_AUDIO_SR)

    embs_list.append(spk_emb.squeeze())
    embs_labels_list.append("spk_emb_" + uttid.split("_")[0])
    embs_list.append(z_spk_emb.squeeze())
    embs_labels_list.append("z_spk_emb_" + uttid.split("_")[0])

    common_phrase = "Mary had a little lamb."
    if PHONEME_INPUT:
      print(common_phrase)
      common_phrase_phoneme_list = g2p(common_phrase)
      common_phrase = " ".join(common_phrase_phoneme_list)
      common_phrase = "{" + common_phrase + "}"
      print(common_phrase)

    mel_output_cp, durations, pitch, energy, z_spk_emb = fastspeech2_ms.encode_text(common_phrase)
    waveform_cp = hifi_gan.decode_batch(mel_output_cp)
    print("mel_output_ms.shape: ", mel_output_ms.shape)
    cp_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized_common_phrase.wav")
    torchaudio.save(cp_audio_path, waveform_cp.squeeze(1).cpu(), EXP_AUDIO_SR)



    target_emb = spk_emb
    synthesized_emb = spk_emb_encoder.encode_mel_spectrogram(mel_output_ms).squeeze(0)
    cp_synthesized_emb = spk_emb_encoder.encode_mel_spectrogram(mel_output_cp).squeeze(0)


    cs_synthesized = cos_sim_score(target_emb, synthesized_emb).item()
    cs_cp_syntheseized = cos_sim_score(target_emb, cp_synthesized_emb).item()
    # import pdb; pdb.set_trace()

    print("{} {:.3f} {:.3f}\n".format(uttid, cs_synthesized, cs_cp_syntheseized))
    cs_f.write("{} {:.3f} {:.3f}\n".format(uttid, cs_synthesized, cs_cp_syntheseized))

combined_embs = torch.stack(embs_list)
tb_writer.add_embedding(
  combined_embs,
  metadata=embs_labels_list,
)