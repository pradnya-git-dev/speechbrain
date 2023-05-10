from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.processing.speech_augmentation import Resample
import json
import os
import shutil
import random
import logging
import torchaudio
import torch
# from speechbrain.pretrained import GraphemeToPhoneme

logger = logging.getLogger(__name__)


def prepare_vctk(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    sample_rate,
    vctk_valid_spk_ids=[],
    vctk_test_spk_ids=[],
    seed=1234,
    append_data=False,
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
    if skip(save_json_train, save_json_valid, save_json_test) and not append_data:
        logger.info("Preparation completed in previous run, skipping.")
        return
    else:
      logger.info(f"Preparing VCTK data. Append mode = {append_data}")

    extension = [".wav"]  # The expected extension for audio files
    wav_list = get_all_files(data_folder, match_and=extension)  # Stores all audio file paths for the dataset
        
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    random.shuffle(wav_list)

    # Creating json files
    
    # train_spk_ids = ["5895", "8297"]
    create_json(
      wav_list,
      save_json_train, 
      vctk_valid_spk_ids,
      vctk_test_spk_ids, 
      sample_rate,
      append_data
    )

    create_json(
      wav_list,
      save_json_valid, 
      vctk_valid_spk_ids,
      vctk_test_spk_ids, 
      sample_rate,
      append_data
    )

    create_json(
      wav_list,
      save_json_test, 
      vctk_valid_spk_ids,
      vctk_test_spk_ids, 
      sample_rate,
      append_data
    )


def create_json(wav_list, json_file, vctk_valid_spk_ids, vctk_test_spk_ids, sample_rate, append_data):
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

    # wav_list = wav_list[:5]
    json_dict = {}

    if append_data:
      with open(json_file) as json_f:
        json_dict = json.load(json_f)
    
    # Processes all the wav files in the list
    for wav_file in wav_list:

      try:

        # Reads the signal
        signal, sig_sr = torchaudio.load(wav_file)

        # Manipulates path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-4:])
        uttid_parts = uttid.split("_")

        # Gets the speaker-id from the utterance-id
        spk_id = uttid_parts[0]

        if json_file.__contains__("train"):
          if (spk_id in vctk_valid_spk_ids) or (spk_id in vctk_test_spk_ids):
            continue
        if json_file.__contains__("valid"):
          if spk_id not in vctk_valid_spk_ids:
            continue
        if json_file.__contains__("test"):
          if spk_id not in vctk_test_spk_ids:
            continue 

        # Gets text file information for the audio clip
        text_path_parts = [path_part for path_part in path_parts]
        text_path_parts[-3] = "txt"
        text_file_id = uttid_parts[0] + "_" + uttid_parts[1]

        # Gets the path for the  text files and extracts the input text
        text_path = os.path.join(
            "/", *text_path_parts[:-1], text_file_id + ".txt"
        )
        with open(text_path) as f:
            text = f.read()
            if text.__contains__("{"):
                text = text.replace("{", "")
            if text.__contains__("}"):
                text = text.replace("}", "")

        phoneme_text_path = os.path.join(
            "/", *text_path_parts[:-1], text_file_id + ".phoneme.txt"
        )

        if not os.path.exists(phoneme_text_path):
          logger.info(f"No phoneme seqence found for {wav_file}. Skipping.")
          continue
        else:
          with open(phoneme_text_path) as ph_f:
            label_phoneme = ph_f.read()
            

        # Resamples the audio file if required
        if sig_sr != sample_rate:
            resampled_signal = torchaudio.functional.resample(signal, sig_sr, sample_rate)
            os.unlink(wav_file)
            torchaudio.save(wav_file, resampled_signal, sample_rate=sample_rate)

      except Exception as ex:
        logger.info(f"Skipping {wav_file} because of the following exception: {ex}")
        continue

      # Creates an entry for the utterance
      json_dict[uttid] = {
          "uttid": uttid,
          "wav": relative_path,
          "spk_id": spk_id,
          "label": text,
          "label_phoneme": label_phoneme,
          "segment": True if "train" in json_file else False,
      }

    # Writes the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


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