import glob
import os
import shutil
import json
from collections import defaultdict
from tqdm import tqdm


def write_json(d: str, path: str) -> None:
    with open(path, 'a') as f:
        json.dump(d, f)
        f.write("\n")


def write_txt(text: str, path: str) -> None:
    with open(path, 'a') as f:
        f.write(text + "\n")


def main(dataset_path, target_dir, manifests_path):
    speaker_encodings = ["us-ascii", "utf-16le", "utf-16le", "utf-16le", "us-ascii", "iso-8859-1", "iso-8859-1",
                         "utf-16le", "utf-16le", "us-ascii"]
    speakers = ["0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]
    emotions_dict = dict(zip(["Neutral", "Angry",  "Happy",  "Sad",  "Surprise"], [0, 1, 2, 3, 4]))
    text_dictionary = defaultdict(lambda: "absent")
    written_wavs_path = set()

    for i, speaker in tqdm(enumerate(speakers)):
        counter = 0

        data = open(f"{dataset_path}/{speaker}/{speaker}.txt", encoding=speaker_encodings[i]).readlines()
        if speaker != "0014":
            assert len(data) % 5 == 0, f"Not all sentences are aligned to all emotions for speaker {speaker}."

        for sample in data:
            preprocessed = sample.split("\t")
            if len(preprocessed) == 3:
                audio_id, sentence, emotion = preprocessed
                emotion = emotion[:-1]
                if emotion in emotions_dict.keys():
                    test_wav_paths = glob.glob(f"{dataset_path}/{speaker}/{emotion}/test/*.wav")
                    test_file_names = [os.path.split(i)[1][:-4] for i in test_wav_paths]
                    val_wav_paths = glob.glob(f"{dataset_path}/{speaker}/{emotion}/evaluation/*.wav")
                    val_file_names = [os.path.split(i)[1][:-4] for i in val_wav_paths]

                    if len(audio_id) == 12:
                        audio_id = audio_id[1:]

                    emotion_id = emotions_dict[emotion]

                    if text_dictionary[sentence] == "absent":
                        counter += 1
                        text_dictionary[sentence] = str(counter)
                    sentence_id = text_dictionary[sentence]

                    if audio_id in test_file_names:
                        previous_wav_path = f"{dataset_path}/{speaker}/{emotion}/test/{audio_id}.wav"
                        manifest_path = f"{manifests_path}/test_manifest.json"
                    elif audio_id in val_file_names:
                        previous_wav_path = f"{dataset_path}/{speaker}/{emotion}/evaluation/{audio_id}.wav"
                        manifest_path = f"{manifests_path}/train_manifest.json"
                    else:
                        previous_wav_path = f"{dataset_path}/{speaker}/{emotion}/train/{audio_id}.wav"
                        manifest_path = f"{manifests_path}/train_manifest.json"

                    if os.path.exists(previous_wav_path) and previous_wav_path not in written_wavs_path:
                        written_wavs_path.add(previous_wav_path)
                        write_txt(sentence, f"{target_dir}/{audio_id[2:4]}_{sentence_id}_{emotion_id}.txt")
                        shutil.copyfile(previous_wav_path,
                                        f"{target_dir}/{audio_id[2:4]}_{sentence_id}_{emotion_id}.wav")

                        dict_to_write = {
                            "audio_filepath": f"{target_dir}/{audio_id[2:4]}_{sentence_id}_{emotion_id}.wav",
                            "mel_filepath": None,
                            "duration": None
                        }

                        write_json(dict_to_write, manifest_path)

                        if manifest_path == f"{manifests_path}/test_manifest.json":
                            for _ in range(7):
                                write_json(dict_to_write, manifest_path)


if __name__ == "__main__":
    dataset_path = "/root/storage/dasha/data/emo-data/ESD"
    target_dir = "/root/storage/dasha/data/emo-data/json/wavs"
    manifests_path = "/root/storage/dasha/data/emo-data/json"
    main(dataset_path, target_dir, manifests_path)
