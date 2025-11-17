from glob import glob
import os
import argparse

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("dir_wav", type=str, help="directory containing wav files")
parser.add_argument("lang", type=str, help="language for inference")
args = parser.parse_args()

from datasets import Dataset, Audio
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

liste_audios = glob(os.path.join(args.dir_wav, '*.wav'))
lang = args.lang

# Load Audio Dataset
audio_dataset = Dataset.from_dict({"audio": liste_audios, "path": liste_audios}).cast_column("audio", Audio(sampling_rate=16000))

# Loading model
model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
processor.tokenizer.set_target_lang(lang)
model.load_adapter(lang)


for i in range(len(liste_audios)):
    sample = audio_dataset[i]["audio"]
    array = sample["array"]

    # Process audio data
    inputs = processor(array, sampling_rate=16_000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    #print("Transcription : {}".format(transcription))
    filename = os.path.splitext(audio_dataset[i]["path"])[0]
    with open(filename + '.txt', 'w') as f:
        f.write(transcription)
