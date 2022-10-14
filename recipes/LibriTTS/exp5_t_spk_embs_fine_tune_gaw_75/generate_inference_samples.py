import torch
import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
from pretrained_model import Tacotron2MS
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files
import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier

INF_SAMPLE_DIR = "/content/libritts_inference_samples"
RESAMPLE_FILES = False
ORIGINAL_SR = 22050
NEW_SR = 16000

spk_emb_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
resampler = Resample(orig_freq=ORIGINAL_SR, new_freq=NEW_SR)

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
tacotron2_ms = Tacotron2MS.from_hparams(source="/content/drive/MyDrive/tacotron2/sb_exp6_t_fine_tune/libritts_fine_tune_gaw_75/epoch_50",
                                        hparams_file="/content/speechbrain/recipes/LibriTTS/exp5_t_spk_embs_fine_tune_gaw_75/inf_hparams.yaml")

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

extension = [".wav"]
wav_list = get_all_files(INF_SAMPLE_DIR, match_and=extension)

text_list = list()


for wav_file in wav_list:

  path_parts = wav_file.split(os.path.sep)
  uttid, _ = os.path.splitext(path_parts[-1])

  signal, fs =torchaudio.load(wav_file)

  if RESAMPLE_FILES:
    signal = resampler(signal)

  spk_embs_list = list()
  xv_emb = spk_emb_encoder.encode_batch(signal)
  spk_embs_list.append(xv_emb.squeeze())
  spk_embs = torch.stack((spk_embs_list))

  original_text_path = os.path.join("/", *path_parts[:-1], uttid + ".original.txt")
  with open(original_text_path) as f:
    original_text = f.read()
    if original_text.__contains__("{"):
      original_text.replace("{", "")
    if original_text.__contains__("}"):
      original_text.replace("}", "")
    text_list.append(original_text)

  print("len(text_list): ", len(text_list))
  print("spk_embs.shape: ", spk_embs.shape)


  # Running the TTS
  mel_output_ss, mel_length_ss, alignment_ss = tacotron2.encode_text(original_text)
  mel_output_ms, mel_length_ms, alignment_ms = tacotron2_ms.encode_text(original_text, spk_embs)

  print("mel_output_ss.shape: ", mel_output_ss.shape)
  print("mel_output_ms.shape: ", mel_output_ms.shape)

  # Running Vocoder (spectrogram-to-waveform)
  # waveform_ss = hifi_gan.decode_batch(mel_output_ss)
  waveform_ms = hifi_gan.decode_batch(mel_output_ms)

  synthesized_audio_path = os.path.join("/", *path_parts[:-1], uttid + "_synthesized.wav")
  torchaudio.save(synthesized_audio_path, waveform_ms.squeeze(1), 22050)