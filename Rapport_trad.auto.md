# Projet traduction automatique (Traduction entre La langue Wu et le mandarin)

## Objectif
Construction d'un modèle de traduction automatique, nous adaptons une architecture seq2seq, entrainée sur le corpus Shangaïen - Mandarin.

## Méthodologie
Corpus : Corpus ASR-SCShhiDiaDuSC
Nous prenons la partie de texte aligné avec le mandarin pour entraîner le modèle.

Type de données : corpus parallèle (Wu - Mandarin) de parole transcrite.

Format : texte aligné en format csv.

Taille de corpus : 

- Train : 3,855 paires
- Dev : 481 paires
- Test : 483 paires
