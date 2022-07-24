import glob
import os
from tqdm import tqdm


def main(wav_txt_dir_path: str, val_set_dataset_path: str, train_manifest_path: str, val_manifest_path: str):
    val_filenames = [f"{i}.wav" for i in open(val_set_dataset_path).readlines()[0].split("|")]
    all_wav_paths = glob.glob(f"{wav_txt_dir_path}/*.wav")
    for wav_path in tqdm(all_wav_paths):
        _, wav_filename = os.path.split(wav_path)
        text = open(os.path.join(wav_txt_dir_path, f"{wav_filename[:-4]}.txt")).readlines()[0]
        if wav_filename in val_filenames:
            for _ in range(8):
                with open(val_manifest_path, 'a') as f:
                    f.write(f"{wav_path}|{text}")
        else:
            with open(train_manifest_path, 'a') as f:
                f.write(f"{wav_path}|{text}")


if __name__ == "__main__":
    wav_txt_dir_path = "/root/storage/dasha/data/emo-data/etts/vk_etts_data/copied_wavs"
    val_set_dataset_path = "/root/storage/dasha/data/emo-data/etts/vk_etts_data/copied_wavs_val_paths.txt"
    train_manifest_path = "/root/storage/dasha/data/emo-data/etts/vk_etts_data/train_manifest.txt"
    val_manifest_path = "/root/storage/dasha/data/emo-data/etts/vk_etts_data/val_manifest.txt"
    main(wav_txt_dir_path, val_set_dataset_path, train_manifest_path, val_manifest_path)
