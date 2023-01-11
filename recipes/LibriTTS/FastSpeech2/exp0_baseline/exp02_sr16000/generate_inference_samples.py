import torchaudio
from fs2_pretrained_interfaces import FastSpeech2
from fs2_pretrained_interfaces import HIFIGAN
import tgt
import numpy as np
import os

def get_alignment(tier, sampling_rate, hop_length):
  sil_phones = ["sil", "sp", "spn"]

  phones = []
  durations = []
  start_time = 0
  end_time = 0
  end_idx = 0
  for t in tier._objects:
      s, e, p = t.start_time, t.end_time, t.text

      # Trim leading silences
      if phones == []:
          if p in sil_phones:
              continue
          else:
              start_time = s

      if p not in sil_phones:
          # For ordinary phones
          phones.append(p)
          end_time = e
          end_idx = len(phones)
      else:
          # For silent phones
          phones.append(p)

      durations.append(
          int(
              np.round(e * sampling_rate / hop_length)
              - np.round(s * sampling_rate / hop_length)
          )
      )

  # Trim tailing silences
  phones = phones[:end_idx]
  durations = durations[:end_idx]

  return phones, durations, start_time, end_time


# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
fastspeech2 = FastSpeech2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/FastSpeech2/sr_16000",
                                     hparams_file="/content/speechbrain/recipes/LibriTTS/FastSpeech2/exp0_baseline/exp02_sr16000/fs2_inference_hparams.yaml")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir_vocoder")

# Running the TTS
LJSPEECH_TEXTGRID_FOLDER = "/content/ljspeech_textgrids/TextGrid/LJSpeech"
TEXTGRID_PATHS = [
  os.path.join(LJSPEECH_TEXTGRID_FOLDER, "LJ050-0084.TextGrid"),
  os.path.join(LJSPEECH_TEXTGRID_FOLDER, "LJ050-0100.TextGrid"),
  os.path.join(LJSPEECH_TEXTGRID_FOLDER, "LJ050-0145.TextGrid"),
  os.path.join(LJSPEECH_TEXTGRID_FOLDER, "LJ050-0247.TextGrid"),
]

for entry in TEXTGRID_PATHS:
  # TEXTGRID_PATH = "/content/LJ050-0247.TextGrid"
  TEXTGRID_PATH = entry
  textgrid = tgt.io.read_textgrid(TEXTGRID_PATH)
  phone, duration, start, end = get_alignment(
      textgrid.get_tier_by_name("phones"),
      16000,
      256
  )
  label = " ".join(phone)

  mel_output, durations, pitch, energy = fastspeech2.encode_text(label)

  # Running Vocoder (spectrogram-to-waveform)
  waveforms = hifi_gan.decode_batch(mel_output)

  # Save the waverform
  utt_textgrid = TEXTGRID_PATH.split(os.path.sep)[-1]
  utt_wav = utt_textgrid.replace(".TextGrid", "_pace1_175.wav")
  torchaudio.save(os.path.join("/content", utt_wav),waveforms.squeeze(1), 16000)
