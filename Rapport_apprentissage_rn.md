# Apprentissage, réseaux de neurones profonds, modèles de langues
## Objectif : 
Construction d'un modèle de traduction automatique, nous adaptons une architecture transformer, entrainée sur le corpus Shangaïen - Mandarin. La traduction est de Mandarin vers la langue Wu.

## Données

1. Corpus : 

Corpus ASR-SCShhiDiaDuSC

Le corpus provient de MagicHub : ASR-SCShhiDiaDuSC: A Scripted Chinese Shanghai Dialect Daily-use Speech Corpus
[Lien vers le corpus](https://magichub.com/datasets/shanghai-dialect-scripted-speech-corpus-daily-use-sentence/)

2. Licence d'utilisation : 

Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

3. Type de données : 

Corpus parallèle (Wu - Mandarin) de parole transcrite. Les textes alignés sont en format csv.

4. Langues 

- langue source : Mandarin
- langue cible : Wu (Shanghaïen)

5. Taille de corpus : 

- Train : 3,855 paires
- Dev : 481 paires
- Test : 483 paires

## Modèles : Transformer

1. Nombre de couches : 

- encodeur : 2-4 couches
- decodeur : 2-4 couches
- couche d'attention (pas encore décidé)

2. Type d'encodage

Tokenisation caractères

3. Type d'apprentissage

Apprentissage supervisé

## Scope couvert

1. Collecte et préparation

- Télécharger les données 
- Extraction des paires de pharses
- Faire un simple nettoyage
- Transformer en csv
- Split 3 set train/dev/test

2. Entraînement

Nous aimerions essayer le modèle de **Transformer** et puis fine-tuning le modèle : 

- Entraîner une baseline Mandarin -> Wu sur sur le jeu d’entraînement de notre corpus parallèle (3,855 paires de phrases)
- Comparer les performances de ces trois modèles et sélectionner l’architecture la plus performante
- Pour l’architecture choisie, mettre en place une back-translation :
  entraîner un modèle Wu → Mandarin,
  traduire 500 phrases monolingues en Wu,
  créer un petit corpus pseudo-parallèle.
- Réentraîner un modèle amélioré Mandarin -> Wu en combinant les données parallèles initiales et le corpus pseudo-parallèle, et ajuster les hyperparamètres.

3. Évaluation

On va évaluer à la fois automatiquement (score BLEU ou ROUGE sur le set de test) et manuellement, c'est à dire donner les commentaires pour les sorties.