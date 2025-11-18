# Langue peu dotée : la langue Wu

### Objectif
Ce projet vise à étudier et adapter un modèle de reconnaissance vocale (ASR) existant pour la langue Wu (shanghaïen), un dialecte chinois considérée comme langue peu dotée.
L'objectif principal est de tester les limites des modèles multilingues actuels (MMS, Whisper, Qwen3, à décider avec Simeng) et de proposer une première approche d'adaptation basée sur la normalisation et le fine-tuning du modèle MMS avec CTC.

### Corpus
Corpus ASR-SCShhiDiaDuSC

### Méthodologie

1. Préparer les données
On a extrait les paires audio–transcription du corpus.
Les fichiers audio sont organisés dans des sous-dossiers (G0004, G0005, etc.) sous WAV/, tandis que les transcriptions correspondantes se trouvent dans UTTRANSINFO.txt.
Du coup on a utilisé un script python `make_jsonl.py` pour parcourir automatiquement tous les fichiers .wav, récupérer leur durée, associer chaque enregistrement à sa transcription en Wu, puis générer trois fichiers au format JSONL : train.jsonl, dev.jsonl et test.jsonl (80% / 10% / 10%).

- train: 3855
- dev  : 481
- test : 483

2. Normalisation du Wu
On a écrit d'abord un script python `make_vocab.py` pour construire un tableau de vocabulaire en Wu， il extrait tous les caractères présents dans les transcriptions en Wu. Chaque caractère devient une entrée du vocabulaire utilisé par le modèle CTC.

Création d'un petit lexique de correspondances (ex. 吾 → 我, 伐 → 不, 侬 → 你).

3. Tests des modèles existants avec le fichier vocab.json
- MMS : ne reconnaît pas wuu, mais fonctionne partiellement avec cmn-script_simplified.
- Whisper (OpenAI) : produit un chinois standard fluide mais dévie du sens du Wu.
- Qwen3-ASR (Alibaba) : résultats plus naturels, sens global conservé.

4. Adaptation du modèle MMS