import torchaudio
from fs2_pretrained_interfaces import FastSpeech2
from fs2_pretrained_interfaces import HIFIGAN
import tgt
import numpy as np
import os
import re
import string

# Load models required for Text-to-Speech
# g2p = G2p()

fastspeech2 = FastSpeech2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/FastSpeech2/exp4_predict_spn/fastspeech_starter_sr_20050_e500",
                                     hparams_file="/content/speechbrain/recipes/LibriTTS/FastSpeech2/exp4_predict_spn/exp42_ssim_sr22050/fs2_inference_hparams.yaml")

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

# Take input text
# import pdb; pdb.set_trace()
input_text = "Were the leaders in this luckless change, though our own Baskerville, who was at work some years before them, went much on the same lines."

print(input_text)
# Convert text label into a phoneme label
# phoneme_seq = g2p(input_text)
# print(phoneme_seq)
# phoneme_seq = " ".join(phoneme_seq)

# Pass phoneme label through the TTS pipeline
mel_output, durations, pitch, energy = fastspeech2.encode_text(input_text)
waveforms = hifi_gan.decode_batch(mel_output)
torchaudio.save(os.path.join("/content", "fs2_site_1.wav"),waveforms.squeeze(1), 22050)

