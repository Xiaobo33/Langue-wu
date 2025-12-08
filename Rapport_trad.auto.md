# Projet traduction automatique (Traduction entre La langue Wu et le mandarin)

## Objectif :
Construction d'un modèle de traduction automatique, nous adaptons une architecture seq2seq, entrainée sur le corpus Shangaïen - Mandarin.

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
- langue source : Wu (Shanghaïen)
- langue cible : Mandarin

#### 7. Taille de corpus : 

- Train : 3,855 paires
- Dev : 481 paires
- Test : 483 paires

## Modèles

#### 1.Nom : 
seq2seq encodeur-decodeur

#### 2. Nombre de couches : 
- encodeur : 2-4 couches
- decodeur : 2-4 couches
- couche d'attention (pas encore décidé)

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
Nous aimerions comparer trois architectures de modèles de traduction, chacune prise en charge par un membre du groupe :
Xiaobo -> Transformers, Simeng -> BiLSTM, Hongying -> BiGRU
- Entraîner une baseline Mandarin -> Wu sur sur le jeu d’entraînement de notre corpus parallèle (3,855 paires de phrases)
- Comparer les performances de ces trois modèles et sélectionner l’architecture la plus performante
- Pour l’architecture choisie, mettre en place une back-translation :
  entraîner un modèle Wu → Mandarin,
  traduire 500 phrases monolingues en Wu,
  créer un petit corpus pseudo-parallèle.
- Réentraîner un modèle amélioré Mandarin -> Wu en combinant les données parallèles initiales et le corpus pseudo-parallèle, et ajuster les hyperparamètres.

#### 3. Évaluation
On va évaluer manuellement, c'est à dire donner les commentaires pour les sorties, si on a encore le temps, on va peut-être faire un BLEU sur le set de test.
