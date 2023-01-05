"""
LJspeech data preparation.
Download: https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

Authors
 * Yingzhi WANG 2022
 * Sathvik Udupa 2022
"""

import os
import csv
import json
import random
import logging
import torchaudio
import numpy as np
from tqdm import tqdm
from speechbrain.utils.data_utils import download_file
from speechbrain.dataio.dataio import load_pkl, save_pkl, load_data_json
import tgt

logger = logging.getLogger(__name__)
OPT_FILE = "opt_ljspeech_prepare.pkl"
METADATA_CSV = "metadata.csv"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
TEST_JSON = "test.json"
WAVS = "wavs"
DURATIONS = "durations"

logger = logging.getLogger(__name__)
OPT_FILE = "opt_ljspeech_prepare.pkl"


def prepare_ljspeech(
    data_folder,
    save_folder,
    splits=["train", "valid"],
    split_ratio=[90, 10],
    seed=1234,
    duration_link=None,
    duration_folder=None,
    compute_pitch=False,
    pitch_folder=".",
    pitch_n_fft=1024,
    pitch_hop_length=256,
    pitch_min_f0=65,
    pitch_max_f0=2093,
    create_symbol_list=False,
    skip_prep=False,
    use_custom_cleaner=False,
):
    """
    Prepares the csv files for the LJspeech datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LJspeech dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    splits : list
        List of splits to prepare.
    split_ratio : list
        Proportion for train and validation splits.
    seed : int
        Random seed
    duration_link: link
        URL where to download the duration links (needed by fastspeech2).
    duration_folder: path
        Folder where to store the durations downloaded from duration_link.
    compute_pitch: bool
        If True, it computes the pitch (needed by fastspeech2)
    pitch_folder:
        Folder where to store the pitch of each audio.
    pitch_n_fft: int
        Number of fft points for pitch computation.
    pitch_hop_length: int
        Hop length for pitch computation.
    pitch_min_f0: int
        Minimum f0 for pitch compuation.
    pitch_max_f0:
        Max f0 for pitch computation.
    skip_prep: Bool
            If True, skip preparation.

    Example
    -------
    >>> from recipes.LJSpeech.TTS.ljspeech_prepare import prepare_ljspeech
    >>> data_folder = 'data/LJspeech/'
    >>> save_folder = 'save/'
    >>> splits = ['train', 'valid']
    >>> split_ratio = [90, 10]
    >>> seed = 1234
    >>> prepare_ljspeech(data_folder, save_folder, splits, split_ratio, seed)
    """
    # setting seeds for reproducible code.
    random.seed(seed)

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seed": seed,
    }
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if compute_pitch:
        if not os.path.exists(pitch_folder):
            os.makedirs(pitch_folder)

    # Setting ouput files
    meta_csv = os.path.join(data_folder, METADATA_CSV)
    wavs_folder = os.path.join(data_folder, WAVS)

    save_opt = os.path.join(save_folder, OPT_FILE)
    save_json_train = os.path.join(save_folder, TRAIN_JSON)
    save_json_valid = os.path.join(save_folder, VALID_JSON)
    save_json_test = os.path.join(save_folder, TEST_JSON)

    """
    # Download duration folder
    if duration_link is not None:
        if not os.path.exists(os.path.join(duration_folder, "durations/LJ001-0001.npy")):
            logger.info("Downloading durations for fastspeech training")
            download_file(
                duration_link, duration_folder + "/durations.zip", unpack=True,
            )
        duration_folder = duration_folder + "/durations"
    """

    duration_folder = os.path.join(data_folder, "durations")
    if not os.path.exists(duration_folder):
      os.makedirs(duration_folder)
    phoneme_alignments_folder = os.path.join(data_folder, "TextGrid", "LJSpeech")

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional check to make sure metadata.csv and wavs folder exists
    assert os.path.exists(meta_csv), "metadata.csv does not exist"
    assert os.path.exists(wavs_folder), "wavs/ folder does not exist"

    msg = "Creating json file for ljspeech Dataset.."
    logger.info(msg)

    data_split, meta_csv = split_sets(data_folder, splits, split_ratio)
    json_files = list()

    if "train" in splits:
        prepare_json(
            data_split["train"],
            save_json_train,
            wavs_folder,
            meta_csv,
            phoneme_alignments_folder,
            duration_folder,
            compute_pitch,
            pitch_folder,
            pitch_n_fft,
            pitch_hop_length,
            pitch_min_f0,
            pitch_max_f0,
            use_custom_cleaner,
        )
        json_files.append(save_json_train)
    if "valid" in splits:
        prepare_json(
            data_split["valid"],
            save_json_valid,
            wavs_folder,
            meta_csv,
            phoneme_alignments_folder,
            duration_folder,
            compute_pitch,
            pitch_folder,
            pitch_n_fft,
            pitch_hop_length,
            pitch_min_f0,
            pitch_max_f0,
            use_custom_cleaner,
        )
        json_files.append(save_json_valid)
    if "test" in splits:
        prepare_json(
            data_split["test"],
            save_json_test,
            wavs_folder,
            meta_csv,
            phoneme_alignments_folder,
            duration_folder,
            compute_pitch,
            pitch_folder,
            pitch_n_fft,
            pitch_hop_length,
            pitch_min_f0,
            pitch_max_f0,
            use_custom_cleaner,
        )
        json_files.append(save_json_test)
    if create_symbol_list:
        create_symbol_file(save_folder, json_files)
    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """
    Detects if the ljspeech data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    skip = True

    split_files = {
        "train": TRAIN_JSON,
        "valid": VALID_JSON,
        "test": TEST_JSON,
    }

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False
    return skip


def split_sets(data_folder, splits, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples for each session.

    Arguments
    ---------
    wav_list : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respectively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    meta_csv = os.path.join(data_folder, METADATA_CSV)
    csv_reader = csv.reader(
        open(meta_csv), delimiter="|", quoting=csv.QUOTE_NONE
    )

    meta_csv = list(csv_reader)

    index_for_sessions = []
    session_id_start = "LJ001"
    index_this_session = []
    for i in range(len(meta_csv)):
        session_id = meta_csv[i][0].split("-")[0]
        if session_id == session_id_start:
            index_this_session.append(i)
            if i == len(meta_csv) - 1:
                index_for_sessions.append(index_this_session)
        else:
            index_for_sessions.append(index_this_session)
            session_id_start = session_id
            index_this_session = [i]

    session_len = [len(session) for session in index_for_sessions]

    data_split = {}
    for i, split in enumerate(splits):
        data_split[split] = []
        for j in range(len(index_for_sessions)):
            if split == "train":
                random.shuffle(index_for_sessions[j])
                n_snts = int(session_len[j] * split_ratio[i] / sum(split_ratio))
                data_split[split].extend(index_for_sessions[j][0:n_snts])
                del index_for_sessions[j][0:n_snts]
            if split == "valid":
                if "test" in splits:
                    random.shuffle(index_for_sessions[j])
                    n_snts = int(
                        session_len[j] * split_ratio[i] / sum(split_ratio)
                    )
                    data_split[split].extend(index_for_sessions[j][0:n_snts])
                    del index_for_sessions[j][0:n_snts]
                else:
                    data_split[split].extend(index_for_sessions[j])
            if split == "test":
                data_split[split].extend(index_for_sessions[j])

    return data_split, meta_csv


def prepare_json(
    seg_lst,
    json_file,
    wavs_folder,
    csv_reader,
    phoneme_alignments_folder,
    durations_folder,
    compute_pitch,
    pitch_folder,
    pitch_n_fft,
    pitch_hop_length,
    pitch_min_f0,
    pitch_max_f0,
    use_custom_cleaner=False,
):
    """
    Creates json file given a list of indexes.

    Arguments
    ---------
    seg_list : list
        The list of json indexes of a given data split.
    json_file : str
        Output json path
    wavs_folder : str
        LJspeech wavs folder
    csv_reader : _csv.reader
        LJspeech metadata
    duration_folder: path
        Folder where to store the durations downloaded from duration_link.
    compute_pitch: bool
        If True, it computes the pitch (needed by fastspeech2)
    pitch_folder:
        Folder where to store the pitch of each audio.
    pitch_n_fft: int
        Number of fft points for pitch computation.
    pitch_hop_length: int
        Hop length for pitch computation.
    pitch_min_f0: int
        Minimum f0 for pitch compuation.
    pitch_max_f0:
        Max f0 for pitch computation.

    Returns
    -------
    None
    """

    # seg_lst = seg_lst[:50]

    print("preparing %s..." % (json_file))
    if compute_pitch:
        print("Computing pitch as well. This takes several minutes...")
    json_dict = {}
    for index in tqdm(seg_lst):
        id = list(csv_reader)[index][0]
        wav = os.path.join(wavs_folder, f"{id}.wav")
        label = list(csv_reader)[index][2]
        if use_custom_cleaner:
            label = custom_clean(label)

        audio, fs = torchaudio.load(wav)

        textgrid_path = os.path.join(
            phoneme_alignments_folder, f"{id}.TextGrid"
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(textgrid_path)
        phone, duration, start, end = get_alignment(
            textgrid.get_tier_by_name("phones"),
            fs,
            pitch_hop_length
        )
        label = " ".join(phone)
        if start >= end:
            print(f"Skipping {id}")
            continue

        duration_file_path = os.path.join(durations_folder, f"{id}.npy")
        np.save(duration_file_path, duration)

        audio = audio[:,
            int(fs * start) : int(fs * end)
        ]

        """
        if durations_folder is not None:
            duration_path = os.path.join(durations_folder, id + ".npy")
            json_dict[id].update({"durations": duration_path})
        """
        
        # Pitch Computation
        if compute_pitch:
            pitch_file = wav.replace(".wav", ".npy").replace(
                wavs_folder, pitch_folder
            )
            if not os.path.isfile(pitch_file):
                
                pitch = torchaudio.functional.compute_kaldi_pitch(
                    waveform=audio,
                    sample_rate=fs,
                    frame_length=(pitch_n_fft / fs * 1000),
                    frame_shift=(pitch_hop_length / fs * 1000),
                    min_f0=pitch_min_f0,
                    max_f0=pitch_max_f0,
                )[0, :, 0]
                pitch = pitch[: sum(duration)]
                np.save(pitch_file, pitch)

        
        json_dict[id] = {
            "uttid": id,
            "wav": wav,
            "label": label,
            "segment": True if "train" in json_file else False,
            "start": start,
            "end": end,
            "durations": duration_file_path,
            "pitch": pitch_file
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")

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


def create_symbol_file(save_folder, json_files):
    lexicon_path = os.path.join(save_folder, "lexicon")
    if os.path.exists(lexicon_path):
        logger.info("Symbols file present")
    else:
        logger.info("Symbols file not present, creating from training data.")
        char_set = set()

        for json_file in json_files:
          data = load_data_json(json_file)
          for id in data:
              line = data[id]["label"]
              char_set.update(line.split())

        with open(lexicon_path, "w") as f:
            f.write("\t".join(char_set))

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
