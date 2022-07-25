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
SAMPLERATE = 16000


def prepare_mini_librispeech(
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
    train_folder = os.path.join(data_folder, "LibriTTS", "dev-clean")
    if not check_folders(train_folder):
        download_mini_libritts(data_folder)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".wav"]
    wav_list = get_all_files(train_folder, match_and=extension)

    # Random split the signal list into train, valid, and test sets.
    data_split = split_sets(wav_list, split_ratio)
    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)


def create_json(wav_list, json_file):
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
    resampler = Resample(orig_freq=24000, new_freq=16000)

    for wav_file in wav_list:

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        resampled_signal = resampler(signal)
        

        

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])


        resampled_path = os.path.join(*path_parts[-5:-1], uttid + "resampled.wav")
        torchaudio.save(resampled_path, resampled_signal, sample_rate=16000)

        duration = resampled_signal.shape[0] / SAMPLERATE


        relative_path = os.path.join("{data_root}", resampled_path)
        original_text_path = os.path.join("{data_root}", *path_parts[-5:-1], uttid + ".original.txt")
        # with open(original_text_path, "r") as orig_f:
        #   original_text = orig_f.read()
        #   orig_f.close()

        normalized_text_path = os.path.join("{data_root}", *path_parts[-5:-1], uttid + ".normalized.txt")
        # with open(normalized_text_path, "r") as norm_f:
        #   normalized_text = norm_f.read()
        #   norm_f.close()

        # Getting speaker-id from utterance-id
        spk_id = uttid.split("_")[0]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "spk_id": spk_id,
            "original_text": original_text_path,
            "normalized_text": normalized_text_path
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
    train_archive = os.path.join(destination, "dev-clean.tar.gz")
    download_file(LIBRITTS_DATASET_URL, train_archive)
    shutil.unpack_archive(train_archive, destination)


if __name__ == "__main__":
    prepare_mini_librispeech("/content/libritts_data", "train.json", "valid.json", "test.json")