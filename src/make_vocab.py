import json
import re

files = ["train.jsonl", "dev.jsonl", "test.jsonl"]

chars = set()

# extraire tous les caracteres
for fp in files:
    for line in open(fp, "r", encoding="utf-8"):
        obj = json.loads(line)
        text = obj["text"]
        for ch in text:
            if ch.strip() != "":
                chars.add(ch)

# construire vocab
vocab = {
    "<pad>": 0, # padding token
    "<s>": 1, # start of sentence
    "</s>": 2, # end of sentence
    "<unk>": 3, # unknown token
    "|": 4 # égale à espace
}

idx = 5
for ch in sorted(chars):
    vocab[ch] = idx
    idx += 1

json.dump(vocab, open("vocab.json", "w", encoding="utf-8"), 
          ensure_ascii=False, indent=2)

print("Total chars:", len(chars))