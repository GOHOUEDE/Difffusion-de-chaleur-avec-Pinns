
---

# Simulation de la difusion de chaleur avec les  Réseaux de Neuronesinformé par la physique (Pinns).

Ce projet implémente un réseau de neurones profond informé par la physique  pour résoudre une équation différentielle partielle (PDE) de la diffussion de chaleur  à l'aide de la méthode des réseaux de neurones basés sur la physique (Physics-Informed Neural Networks - PINNs). Nous utilisons PyTorch pour l'entraînement et la résolution  et inclut les méthodes d'optimisation  Adam et L-BFGS pour la minimisation de la perte.

## Installation

Assurez-vous d'avoir les dépendances nécessaires installées :

```bash
pip install torch matplotlib scikit-learn pyDOE scipy
```

## Structure du projet

Voici les principaux fichiers du projet :

- **Chaleur_pinns.ipynb** : Le notebook qui présente le travail(code ,chargement et graphiques).
- - **code.py** : Contient l'intégralité des codes python pour la simulation de la difussion de chaleur.
- **README.md** : Documentation du projet.

  
## Description du modèle

Le modèle utilise un réseau de neurones fully connected (FCN) pour approximer la solution d'une équation différentielle partielle donnée. Le réseau est entraîné à l'aide de données de conditions aux limites et de points de collocation générés de manière aléatoire à l'aide de l'échantillonnage hypercube latin.

Le réseau suit les étapes suivantes :

1. **Conditions aux limites (BC)** : Le modèle est entraîné pour satisfaire les conditions aux limites définies sur les bords du domaine.
2. **Équation différentielle partielle (PDE)** : La perte est calculée en fonction de la satisfaction de l'EDP à l'intérieur du domaine en utilisant la dérivée de la solution par rapport aux variables spatiales et temporelles.

## Variables et Hyperparamètres

- **steps** : Le nombre d'itérations d'entraînement.
- **lr** : Le taux d'apprentissage pour l'optimisation.
- **layers** : La structure du réseau de neurones (par exemple, [2, 32, 32, 1] pour un réseau avec 2 entrées, deux couches cachées de 32 neurones et une sortie).
- **Nu** : Le nombre de points d'entraînement issus des conditions aux limites.
- **Nf** : Le nombre de points de collocation pour l'EDP.
- **x_min, x_max, t_min, t_max** : Les bornes spatiales et temporelles pour la simulation.
  
## Utilisation

1. **Préparation des données** : Le script génère des données d'entrée à partir de points uniformément espacés dans un domaine défini.
2. **Entraînement du modèle** : Le réseau de neurones est formé sur les données de conditions aux limites et les points de collocation pour minimiser la perte.
3. **Visualisation** : Les résultats sont tracés en 3D à l'aide de Matplotlib pour observer l'évolution de la solution dans le domaine spatial et temporel.

### Exemple de génération de données et entraînement


## Résultats



