import json
from speechbrain.pretrained import EncoderClassifier
from speechbrain.processing.speech_augmentation import Resample
import torchaudio
import pickle
import torch



def compute_speaker_embeddings(input_filepaths, output_file_paths, data_folder, audio_sr, spk_emb_sr):
  """This function processes a JSON file to compute the speaker embeddings.
  Arguments
  ---------
  input_filepaths : list
  A list of paths to the JSON files to be processed
  output_file_path : list
  A list of paths to the output pickle files corrsponding to the input JSON files
  data_folder : str
  Path to the folder where LibriTTS data is stored
  audio_sr : int
  Sample rate of the audio files from the dataset
  spk_emb_sr : int
  Sample rate used by the speaker embedding encoder
  """



  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  spk_emb_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                                   run_opts={"device":device})
  resampler = None
  resample_audio = False
  if audio_sr != spk_emb_sr:
    resampler = Resample(orig_freq=audio_sr, new_freq=spk_emb_sr)
    resample_audio = True

  for i in range(len(input_filepaths)):
    speaker_embeddings = dict()
    json_file = open(input_filepaths[i])
    json_data = json.load(json_file)

    for utt_id, utt_data in json_data.items():
      utt_wav_path = utt_data["wav"]
      utt_wav_path = utt_wav_path.replace("{data_root}", data_folder)
      signal, sig_sr = torchaudio.load(utt_wav_path)
      if resample_audio:
        signal = resampler(signal)
      signal = signal.to(device)

      spk_emb = spk_emb_encoder.encode_batch(signal)
      spk_emb = spk_emb.squeeze()
      spk_emb = spk_emb.detach()

      if utt_data["spk_id"] not in speaker_embeddings.keys():
        speaker_embeddings[utt_data["spk_id"]] = list()

      speaker_embeddings[utt_data["spk_id"]].append(spk_emb)


    with open(output_file_paths[i], "wb") as output_file:
      pickle.dump(speaker_embeddings, output_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
  compute_speaker_embeddings(["train.json"], 
                             "/content/libritts_data_sr_16000", 
                             ["train_speaker_embeddings.pickle"],
                             16000, 
                             16000)