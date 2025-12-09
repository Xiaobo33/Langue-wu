# Projet traduction automatique (Traduction entre La langue Wu et le mandarin)

## Objectif :
Construction d'un modèle de traduction automatique, nous adaptons une architecture transformer, entrainée sur le corpus Shangaïen - Mandarin.

## Entrées (corpus) :

#### 1. Corpus : 
Corpus ASR-SCShhiDiaDuSC

#### 2. Disponibilité : 
Le corpus provient de MagicHub : ASR-SCShhiDiaDuSC: A Scripted Chinese Shanghai Dialect Daily-use Speech Corpus

#### 3. Adresse : 
https://magichub.com/datasets/shanghai-dialect-scripted-speech-corpus-daily-use-sentence/

#### 4. Licence d'utilisation : 
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

#### 5. Type de données : 
Corpus parallèle (Wu - Mandarin) de parole transcrite. Les textes alignés sont en format csv.

#### 6. Langues 
- langue source : Mandarin
- langue cible : Wu (Shanghaïen)

#### 7. Taille de corpus : 

- Train : 3,855 paires
- Dev : 481 paires
- Test : 483 paires

## Modèles

#### 1.Nom : 
Transformer

#### 2. Nombre de couches : 
- encodeur : 4 couches
- decodeur : 4 couches
- multi-head self-attentions

#### 3. Type d'encodage
Tokenisation caractères

#### 4. Type d'apprentissage
Apprentissage supervisé

## Scope couvert

#### 1. Collecte et préparation
- Télécharger les données 
- Extraction des paires de pharses
- Faire un simple nettoyage
- Transformer en csv
- Split 3 set train/dev/test

#### 2. Entraînement
Nous aimerions essayer le modèle de **Transformer** et puis fine-tuning le modèle : 

- Entraîner une baseline Mandarin -> Wu sur sur le jeu d’entraînement de notre corpus parallèle (3,855 paires de phrases)
- Comparer les performances de ces trois modèles et sélectionner l’architecture la plus performante
- Pour l’architecture choisie, mettre en place une back-translation :
  entraîner un modèle Wu → Mandarin,
  traduire 500 phrases monolingues en Wu,
  créer un petit corpus pseudo-parallèle.
- Réentraîner un modèle amélioré Mandarin -> Wu en combinant les données parallèles initiales et le corpus pseudo-parallèle, et ajuster les hyperparamètres.

#### 3. Évaluation
On va évaluer manuellement, c'est à dire donner les commentaires pour les sorties, si on a encore le temps, on va peut-être faire un BLEU sur le set de test.


### Architecture du Modèle : Transformer

#### Baseline

Encodeur (Mandarin) -> Decodeur (Wu) -> Output Layer

#### Hyperparamètres :

| Hyperparamètre | Valeur | Description |
|----------------|--------|-------------|
| `D_MODEL` | 128 | Dimension des embeddings |
| `N_ENC` | 4 | Nombre de couches d'encodeur |
| `N_DEC` | 4 | Nombre de couches de décodeur |
| `N_HEADS` | 8 | Nombre de têtes d'attention |
| `DFF` | 512 | Dimension du feed-forward |
| `DROP` | 0.1 | Taux de dropout |
| `MAX_SRC_LEN` | 50 | Longueur max source |
| `MAX_TGT_LEN` | 50 | Longueur max cible |
| `MAX_VOCAB_SIZE` | 4000 | Taille max du vocabulaire |

#### Callbacks

1. **ModelCheckpoint** :
   - Sauvegarde le meilleur modèle selon `val_loss`
   - Permet de récupérer le modèle optimal

2. **EarlyStopping** :
   - `patience=5` : arrête si pas d'amélioration pendant 5 epochs
   - `restore_best_weights=True` : restaure les meilleurs poids
   - Évite le sur-apprentissage

#### Résultats

| Mandarin | Wu | Evaluation |
|----------|----|------------|
|你好 | 侬好叫吾老 |
|你今天吃饭了吗 | 侬今朝吃饭了伐 |
|我不会说上海话 | 吾伐会的讲呢 |
|今天天气真好 | 今朝天气真呃好 |
|请你帮我一下 | 请侬帮吾一下 |
|祝您开心 | 祝侬开心心 |