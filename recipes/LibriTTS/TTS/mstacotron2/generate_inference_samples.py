import torch
import torchaudio
from speechbrain.pretrained import HIFIGAN
from interfaces.pretrained import MSTacotron2, MelSpectrogramEncoder
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files
import os
import torchaudio
from torch import nn
import glob

# Load the evaluation dataset
DATA_DIR = "mstts_evaluation_dataset"
AUDIO_EXTENSION = ".wav"

# Load the required models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPK_EMB_ENCODER_PATH = "/content/drive/MyDrive/ecapa_tdnn/vc12_mel_spec_80"
MSTTS_MODEL_PATH = "/content/drive/MyDrive/2023/concordia/mstts_experiments/paper/saved_models/exp1_baseline/ltc_sub/exp1_baseline_mstacotron2_ltc_sub"
MSTTS_HPARAMS_PATH = "/content/speechbrain/recipes/LibriTTS/TTS/mstacotron2/hparams/exp1_baseline/inf_add_no_scl.yaml"

# Loads speaker embedding model
spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source=SPK_EMB_ENCODER_PATH,
                                                 run_opts={"device": DEVICE})

# Loads TTS model
tacotron2_ms = MSTacotron2.from_hparams(source=MSTTS_MODEL_PATH,
                                        hparams_file=MSTTS_HPARAMS_PATH,
                                        run_opts={"device": DEVICE})

# Loads Vocoder
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                run_opts={"device": DEVICE})


# Defines Cosine similarity
cos_sim_score = nn.CosineSimilarity()

# Processed the dataset one speaker at a time
# The following line works because the evaluation dataset is structured that way
for spk_dir in glob.glob(f"{DATA_DIR}/*/*/*", recursive=True):
    print(spk_dir)

    # Gets the reference waveforms - Here, we use only one
    spk_emb_ref_dir = os.path.join(spk_dir, "speaker_embedding_references")
    ref_wav_path = get_all_files(spk_emb_ref_dir, match_and=[AUDIO_EXTENSION])[0]

    # Computes the reference speaker embedding to use when generating audios for the speaker
    

