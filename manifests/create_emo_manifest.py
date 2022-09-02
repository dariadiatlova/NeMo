import glob
import json
from typing import Dict

from tqdm import tqdm


def write_json(d: Dict, path: str) -> None:
    with open(path, 'a') as f:
        json.dump(d, f)
        f.write("\n")


def main(wav_dir_path, train_manifest_path, val_manifest_path, val_size=8):
    all_wav_paths = glob.glob(f"{wav_dir_path}/*.wav")
    val_filenames = all_wav_paths[-val_size:]
    for wav_path in tqdm(all_wav_paths):
        dict_to_write = {"audio_filepath": wav_path, "mel_filepath": None, "duration": None}
        if wav_path in val_filenames:
            write_json(dict_to_write, val_manifest_path)
        else:
            write_json(dict_to_write, train_manifest_path)


if __name__ == "__main__":
    wav_dir_path = "/root/storage/dasha/data/emo-data/english_esd/preprocessed/trimmed_wav"
    train_manifest_path = "/root/storage/dasha/data/emo-data/english_esd/hifi/train_manifest.json"
    val_manifest_path = "/root/storage/dasha/data/emo-data/english_esd/hifi/val_manifest.json"
    main(wav_dir_path, train_manifest_path, val_manifest_path)
