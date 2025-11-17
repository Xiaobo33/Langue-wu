import os
import json
from glob import glob

import torchaudio

# Chemin racine du corpus
DATA_ROOT = "/home/sibel/Langue-wu/Data/Shanghai_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence"

WAV_DIR = os.path.join(DATA_ROOT, "WAV")
TRANS_FILE = os.path.join(DATA_ROOT, "UTTRANSINFO.txt")

OUT_TRAIN = os.path.join(DATA_ROOT, "train.jsonl")
OUT_DEV   = os.path.join(DATA_ROOT, "dev.jsonl")
OUT_TEST  = os.path.join(DATA_ROOT, "test.jsonl")


def load_transcriptions(path):
    """Lit UTTRANSINFO.txt et renvoie un dict {utt_id: transcription_wu}."""
    id2text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5 or parts[0] == "CHANNEL":
                continue
            utt_id = parts[1].replace(".wav", "")   # ex. G0004_0_S0002
            text = parts[4].strip()                # colonne TRANSCRIPTION (Wu)
            id2text[utt_id] = text
    return id2text


def get_duration(path):
    """Durée de l'audio en secondes."""
    wav, sr = torchaudio.load(path)
    return wav.shape[1] / sr


def write_jsonl(data, path):
    """Écrit une liste de dicts au format JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    id2text = load_transcriptions(TRANS_FILE)

    examples = []
    for wav_path in glob(os.path.join(WAV_DIR, "G*/*.wav")):
        fname = os.path.basename(wav_path)        # G0004_0_S0002.wav
        utt_id = fname.replace(".wav", "")
        if utt_id not in id2text:
            continue
        text = id2text[utt_id]
        duration = round(get_duration(wav_path), 3)
        examples.append({
            "path": wav_path,
            "duration": duration,
            "text": text
        })

    examples = sorted(examples, key=lambda x: x["path"])
    n = len(examples)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)

    train_set = examples[:n_train]
    dev_set   = examples[n_train:n_train + n_dev]
    test_set  = examples[n_train + n_dev:]

    write_jsonl(train_set, OUT_TRAIN)
    write_jsonl(dev_set, OUT_DEV)
    write_jsonl(test_set, OUT_TEST)

    print("train:", len(train_set))
    print("dev  :", len(dev_set))
    print("test :", len(test_set))


if __name__ == "__main__":
    main()
