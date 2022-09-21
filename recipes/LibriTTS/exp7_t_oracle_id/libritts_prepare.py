from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.speech_augmentation import Resample
import json
import os, shutil
import random
import logging
import torchaudio

logger = logging.getLogger(__name__)
LIBRITTS_DATASET_URL = "https://www.openslr.org/resources/60/dev-clean.tar.gz"
SAMPLERATE = 22050


def prepare_libritts(
        data_folder,
        save_json_train,
        save_json_valid,
        save_json_test,
        split_ratio=[80, 10, 10],
):
    """
    Prepares the json files for the Mini Librispeech dataset.
    Downloads the dataset if it is not found in the `data_folder`.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    train_folder = os.path.join(data_folder, "LibriTTS", LIBRITTS_DATASET_URL.split(os.path.sep)[-1].split(".")[0])
    # train_folder = os.path.join(data_folder, "LibriTTS", "dev-clean")
    resample_audio = False
    if not check_folders(data_folder):
        download_mini_libritts(data_folder)
        resample_audio = True

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".wav"]
    speaker_counter = 1

    wav_list = list()

    for speaker_folder in os.listdir(train_folder):
      speaker_folder_path = os.path.join(train_folder, speaker_folder)
      if os.path.isdir(speaker_folder_path):
          speaker_samples = get_all_files(speaker_folder_path, match_and=extension)
          if len(speaker_samples) > 100:
            wav_list.extend(speaker_samples)
            logger.info(f"Using data for speaker: {speaker_folder}")
            speaker_counter = speaker_counter - 1
            if speaker_counter == 0:
              break


    # wav_list = get_all_files(train_folder, match_and=extension)
    logger.info(f"Total number of samples: {len(wav_list)}")


    # Random split the signal list into train, valid, and test sets.
    data_split = split_sets(wav_list, split_ratio)
    # Creating json files
    create_json(data_split["train"], save_json_train, resample_audio)
    create_json(data_split["valid"], save_json_valid, resample_audio)
    create_json(data_split["test"], save_json_test, resample_audio)


def create_json(wav_list, json_file, resample_audio=False):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    """

    # wav: /content/libritts_data/LibriTTS/dev-clean/6313/66125/6313_66125_000011_000000.wav
    # original_text: /content/libritts_data/LibriTTS/dev-clean/6313/66125/84_121123_000007_000001.original.txt
    # normalized_text: /content/libritts_data/LibriTTS/dev-clean/6313/66125/84_121123_000007_000001.normalized.txt

    # Processing all the wav files in the list
    json_dict = {}
    resampler = Resample(orig_freq=24000, new_freq=SAMPLERATE)
    spk_ids = dict()
    speaker_counter = 0

    for wav_file in wav_list:

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        if not resample_audio:
            uttid = "_".join(uttid.split("_")[:-1])

        relative_path = os.path.join("{data_root}", *path_parts[-5:])

        original_text_path = os.path.join("/", *path_parts[:-1], uttid + ".original.txt")

        with open(original_text_path) as f:
            original_text = f.read()
            if original_text.__contains__("{") or original_text.__contains__("}"):
              continue

        normalized_text_path = os.path.join("/", *path_parts[:-1], uttid + ".normalized.txt")

        with open(normalized_text_path) as f:
            normalized_text = f.read()

        if resample_audio:
            signal = signal.unsqueeze(0)
            resampled_signal = resampler(signal)

            resampled_path = os.path.join("/", *path_parts[:-1], uttid + "_resampled.wav")
            torchaudio.save(resampled_path, resampled_signal, sample_rate=SAMPLERATE)

            duration = resampled_signal.shape[1] / SAMPLERATE

            resampled_path_parts = resampled_path.split(os.path.sep)
            relative_path = os.path.join("{data_root}", *resampled_path_parts[-5:])

            os.unlink(wav_file)

        # Getting speaker-id from utterance-id
        spk_id = uttid.split("_")[0]
        if not spk_id in spk_ids:
          spk_ids[spk_id] = speaker_counter
          speaker_counter = speaker_counter + 1

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "spk_id": spk_ids[spk_id],
            "original_text": original_text,
            "normalized_text": normalized_text,
            "segment": True if "train" in json_file else False
        }

    # Writing the dictionary to the json file
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
    wav_lst : list
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
    # Random shuffle of the list
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


def download_mini_libritts(destination):
    """Downloads the dataset and unpacks it.
    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    train_archive = os.path.join(destination, LIBRITTS_DATASET_URL.split(os.path.sep)[-1])
    download_file(LIBRITTS_DATASET_URL, train_archive)
    shutil.unpack_archive(train_archive, destination)


if __name__ == "__main__":
    prepare_libritts("/content/libritts_train_clean_100_resampled",
                     "train.json", "valid.json", "test.json")
