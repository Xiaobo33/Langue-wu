from glob import glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("dir_wav", type=str, help="directory containing wav files")
parser.add_argument("-o", "--output", help="output name (tsv file)", default="lang_pred_mms.tsv")
args = parser.parse_args()


from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
import librosa
import pandas as pd
from datasets import Dataset, Audio

"""

Applique le détecteur de langue MMS sur les échantillons audio du répertoire en paramètre.

"""

model_id = "facebook/mms-lid-4017"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)

df = pd.DataFrame()

# Liste des fichiers wav
audio_names = list(glob(os.path.join(args.dir_wav, '*.wav')))

# Conversion en Dataset
audio_dataset = Dataset.from_dict({"audio": audio_names}).cast_column("audio", Audio(sampling_rate=16000))

preds = []
for i in range(len(audio_names)):
        sample = audio_dataset[i]["audio"]
        array = sample["array"]
        inputs = processor(array, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
                outputs = model(**inputs).logits

        lang_id = torch.argmax(outputs, dim=-1)[0].item()
        detected_lang = model.config.id2label[lang_id]
        preds.append(detected_lang)

df = pd.DataFrame()
df['fichier'] = audio_names
df['prediction'] = preds
df['duree'] = [librosa.get_duration(path=x) for x in audio_names]

df.to_csv(args.output, sep='\t', index=False)


