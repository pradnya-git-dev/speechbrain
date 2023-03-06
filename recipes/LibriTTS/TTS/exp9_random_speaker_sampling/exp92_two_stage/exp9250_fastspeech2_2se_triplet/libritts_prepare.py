from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.processing.speech_augmentation import Resample
import json
import os
import shutil
import random
import logging
import torchaudio
import torch
from speechbrain.dataio.dataio import load_pkl, save_pkl, load_data_json
import tgt
import numpy as np

logger = logging.getLogger(__name__)
# Change the entries in the following "LIBRITTS_SUBSETS" to modify the downloaded subsets for LibriTTS
# Used subsets ["dev-clean", "train-clean-100", "train-clean-360"]
LIBRITTS_SUBSETS = ["dev-clean"]
LIBRITTS_URL_PREFIX = "https://www.openslr.org/resources/60/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_libritts(
    data_folder,
    save_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    sample_rate,
    splits=["train", "valid"],
    split_ratio=[90, 10],
    model_name=None,
    seed=1234,
    pitch_n_fft=1024,
    pitch_hop_length=256,
    pitch_min_f0=65,
    pitch_max_f0=2093,
    skip_prep=False,
    use_custom_cleaner=False,
):
    """
    Prepares the json files for the LibriTTS dataset.
    Downloads the dataset if it is not found in the `data_folder` as expected.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the LibriTTS dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    sample_rate : int
        The sample rate to be used for the dataset
    split_ratio : list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # setting seeds for reproducible code.
    random.seed(seed)

    # Checks if this phase is already done (if so, skips it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return


    phoneme_alignments_folder = os.path.join(data_folder, "TextGrid")
    duration_folder = os.path.join(data_folder, "durations")
    if not os.path.exists(duration_folder):
      os.makedirs(duration_folder)
    pitch_folder = os.path.join(data_folder, "pitch")
    if not os.path.exists(pitch_folder):
      os.makedirs(pitch_folder)

    extension = [".wav"]  # The expected extension for audio files
    wav_list = list()  # Stores all audio file paths for the dataset

    # For every subset of the dataset, if it doesn't exist, downloads it and sets flag to resample the subset
    for subset_name in LIBRITTS_SUBSETS:

        subset_folder = os.path.join(data_folder, subset_name)
        subset_archive = os.path.join(subset_folder, subset_name + ".tar.gz")

        subset_data = os.path.join(subset_folder, "LibriTTS")
        if not check_folders(subset_data):
            logger.info(
                f"No data found for {subset_name}. Checking for an archive file."
            )
            if not os.path.isfile(subset_archive):
                logger.info(
                    f"No archive file found for {subset_name}. Downloading and unpacking."
                )
                subset_url = LIBRITTS_URL_PREFIX + subset_name + ".tar.gz"
                download_file(subset_url, subset_archive)
                logger.info(f"Downloaded data for subset {subset_name}.")
            else:
                logger.info(
                    f"Found an archive file for {subset_name}. Unpacking."
                )

            shutil.unpack_archive(subset_archive, subset_folder)

        # Collects all files matching the provided extension
        wav_list.extend(get_all_files(subset_folder, match_and=extension))
        # wav_list = wav_list[:64]
        
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    # if len(split_ratio) == 1:
    #   create_json(wav_list, save_json_train, sample_rate)
    #   return

    # Random split the signal list into train, valid, and test sets.
    # data_split = split_sets(wav_list, split_ratio)
    # import pdb; pdb.set_trace()

    # Creating json files
    random.shuffle(wav_list)

    # dev-clean train split - 34 speakers
    # train_spk_ids = [
    #   "7976", "6319", "1993", "2902", "174", "6241", "422", "1272", "6313", "2035", "1673", "7850", "3536", "5338", "2277", "3576", "3752", "652", "1462", "1988", "777", "3000", "6345", "3853", "2412", "2428", "251", "1919", "3170", "3081", "2086", "2078", "6295", "5694"
    # ]
    train_spk_ids = ["5895", "8297"]
    if "train" in splits:
      create_json(wav_list, 
                  train_spk_ids,
                  save_json_train,
                  phoneme_alignments_folder, 
                  sample_rate,
                  pitch_n_fft,
                  pitch_hop_length,
                  pitch_min_f0,
                  pitch_max_f0,
                  use_custom_cleaner)

    # dev-clean valid split - 6 speakers - 3M, 3F
    # valid_spk_ids = ["5895", "8297", "2803", "5536", "8842", "84" ]
    valid_spk_ids = ["5895", "8297"]
    if "valid" in splits:
      create_json(wav_list, 
                  valid_spk_ids,
                  save_json_valid,
                  phoneme_alignments_folder,
                  sample_rate,
                  pitch_n_fft,
                  pitch_hop_length,
                  pitch_min_f0,
                  pitch_max_f0,
                  use_custom_cleaner)
    
    # dev-clean test split - 0 speakers
    test_spk_ids = []
    if "test" in splits:
      create_json(wav_list, 
                  test_spk_ids,
                  save_json_test,
                  phoneme_alignments_folder,
                  sample_rate,
                  pitch_n_fft,
                  pitch_hop_length,
                  pitch_min_f0,
                  pitch_max_f0,
                  use_custom_cleaner)


def create_json(wav_list, 
                split_spk_ids,
                json_file,
                phoneme_alignments_folder,
                sample_rate,
                pitch_n_fft,
                pitch_hop_length,
                pitch_min_f0,
                pitch_max_f0,
                use_custom_cleaner=False):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    sample_rate : int
        The sample rate to be used for the dataset
    """

    json_dict = {}
    # Creates a resampler object with orig_freq set to LibriTTS sample rate (24KHz) and  new_freq set to SAMPLERATE
    resampler = Resample(orig_freq=24000, new_freq=sample_rate)

    # Processes all the wav files in the list
    for wav_file in wav_list:

        # Reads the signal
        signal, sig_sr = torchaudio.load(wav_file)
        # signal = signal.squeeze(0)

        duration = signal.shape[1] / sig_sr
        if duration > 10.10:
            # print(signal.shape, duration, wav_file)
            continue

        # Manipulates path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-6:])

        # Gets the speaker-id from the utterance-id
        spk_id = uttid.split("_")[0]

        if spk_id not in split_spk_ids:
          continue

        # Gets the path for the  text files and extracts the input text
        normalized_text_path = os.path.join(
            "/", *path_parts[:-1], uttid + ".normalized.txt"
        )
        with open(normalized_text_path) as f:
            normalized_text = f.read()
            if normalized_text.__contains__("{"):
                normalized_text = normalized_text.replace("{", "")
            if normalized_text.__contains__("}"):
                normalized_text = normalized_text.replace("}", "")
        label = normalized_text

        # Resamples the audio file if required
        if sig_sr != sample_rate:
            # signal = signal.unsqueeze(0)
            resampled_signal = resampler(signal)
            os.unlink(wav_file)
            torchaudio.save(wav_file, resampled_signal, sample_rate=sample_rate)

        # Gets the path for the  text files and extracts the input text
        textgrid_path = os.path.join(
            phoneme_alignments_folder, spk_id, f"{uttid}.TextGrid"
        )
        if not os.path.exists(textgrid_path):
          print("Skipping because this does not exist: ", textgrid_path)
          continue

        # Get alignments
        textgrid = tgt.io.read_textgrid(textgrid_path, include_empty_intervals=True)
        phone, duration, start, end = get_alignment(
            textgrid.get_tier_by_name("phones"),
            sample_rate,
            pitch_hop_length
        )
        label_phoneme = " ".join(phone)
        if start >= end:
            print(f"Skipping {uttid}")
            continue

        duration_file_path = os.path.join("/", *path_parts[:-1], uttid + "_durations.npy")
        np.save(duration_file_path, duration)

        signal = signal[:,
            int(sample_rate * start) : int(sample_rate * end)
        ]

        pitch_file_path = os.path.join("/", *path_parts[:-1], uttid + "_pitch.npy")
        if not os.path.isfile(pitch_file_path):
          pitch = torchaudio.functional.compute_kaldi_pitch(
                    waveform=signal,
                    sample_rate=sample_rate,
                    frame_length=(pitch_n_fft / sample_rate * 1000),
                    frame_shift=(pitch_hop_length / sample_rate * 1000),
                    min_f0=pitch_min_f0,
                    max_f0=pitch_max_f0,
                )[0, :, 0]
          pitch = pitch[: sum(duration)]
          np.save(pitch_file_path, pitch)


        # Creates an entry for the utterance
        json_dict[uttid] = {
            "uttid": uttid,
            "wav": relative_path,
            "spk_id": spk_id,
            "label": label,
            "label_phoneme": label_phoneme,
            "segment": True if "train" in json_file else False,
            "start": start,
            "end": end,
            "durations": duration_file_path,
            "pitch": pitch_file_path
        }
        
    # Writes the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def get_alignment(tier, sampling_rate, hop_length):
  """
  Returns phonemes, phoneme durations (in frames), start time (in seconds), end time (in seconds).
  This function is adopted from https://github.com/ming024/FastSpeech2/blob/master/preprocessor/preprocessor.py
  Arguments
  ---------
  tier : tgt.core.IntervalTier
      For an utterance, contains Interval objects for phonemes and their start time and end time in seconds
  sampling_rate : int
      Sample rate if audio signal
  hop_length : int
      Hop length for duration computation
  Returns
  -------
  (phones, durations, start_time, end_time) : tuple
      The phonemes, durations, start time, and end time for an utterance
  """

  sil_phones = ["sil", "sp", "spn", ""]

  phonemes = []
  durations = []
  start_time = 0
  end_time = 0
  end_idx = 0

  for t in tier._objects:
      s, e, p = t.start_time, t.end_time, t.text

      # Trims leading silences
      if phonemes == []:
          if p in sil_phones:
              continue
          else:
              start_time = s

      if p not in sil_phones:
          # For ordinary phones
          # Removes stress indicators
          if p[-1].isdigit():
            phonemes.append(p[:-1])
          else:
            phonemes.append(p)
          end_time = e
          end_idx = len(phonemes)
      else:
          # Uses a unique token for all silent phones
          phonemes.append("spn")

      durations.append(
          int(
              np.round(e * sampling_rate / hop_length)
              - np.round(s * sampling_rate / hop_length)
          )
      )

  # Trims tailing silences
  phonemes = phonemes[:end_idx]
  durations = durations[:end_idx]

  return phonemes, durations, start_time, end_time


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.

    Arguments
    ---------
    wav_list : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    # Random shuffles the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True

def custom_clean(text):
    import re
    from unidecode import unidecode
    _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
                    ('mrs', 'misess'),
                    ('mr', 'mister'),
                    ('dr', 'doctor'),
                    ('st', 'saint'),
                    ('co', 'company'),
                    ('jr', 'junior'),
                    ('maj', 'major'),
                    ('gen', 'general'),
                    ('drs', 'doctors'),
                    ('rev', 'reverend'),
                    ('lt', 'lieutenant'),
                    ('hon', 'honorable'),
                    ('sgt', 'sergeant'),
                    ('capt', 'captain'),
                    ('esq', 'esquire'),
                    ('ltd', 'limited'),
                    ('col', 'colonel'),
                    ('ft', 'fort'),
                    ]]
    text = unidecode(text.lower())
    text = re.sub("[:;]", " - ", text)
    text = re.sub("[)(\[\]\"]", " ", text)
    text = re.sub(' +', ' ', text)
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    text = text.strip().strip().strip('-')
    return text