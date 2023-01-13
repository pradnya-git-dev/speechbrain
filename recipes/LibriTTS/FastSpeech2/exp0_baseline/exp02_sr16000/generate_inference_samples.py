import torchaudio
from fs2_pretrained_interfaces import FastSpeech2
from fs2_pretrained_interfaces import HIFIGAN
from g2p_en import G2p
import tgt
import numpy as np
import os
import re
import string

# Load models required for Text-to-Speech
# g2p = G2p()

fastspeech2 = FastSpeech2.from_hparams(source="/content/drive/MyDrive/mstts_saved_models/FastSpeech2/sr_16000",
                                     hparams_file="/content/speechbrain/recipes/LibriTTS/FastSpeech2/exp0_baseline/exp02_sr16000/fs2_inference_hparams.yaml")

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir_vocoder")

# Take input text
# import pdb; pdb.set_trace()
input_text = "Testing the grapheme to phoneme model from the inference file."
print(input_text)
# Convert text label into a phoneme label
# phoneme_seq = g2p(input_text)
# print(phoneme_seq)
# phoneme_seq = " ".join(phoneme_seq)

# Pass phoneme label through the TTS pipeline
mel_output, durations, pitch, energy = fastspeech2.encode_text(input_text)
waveforms = hifi_gan.decode_batch(mel_output)
torchaudio.save(os.path.join("/content", "sample_audio.wav"),waveforms.squeeze(1), 16000)

