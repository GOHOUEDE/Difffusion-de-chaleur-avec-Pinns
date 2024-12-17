
---

# Projet de Réseau de Neurones pour Résoudre une Équation Différentielle Partielle (PDE)

Ce projet implémente un réseau de neurones profond pour résoudre une équation différentielle partielle (PDE) à l'aide de la méthode des réseaux de neurones basés sur la physique (Physics-Informed Neural Networks - PINNs). Il utilise PyTorch pour l'entraînement et la résolution de l'EDP et inclut des méthodes d'optimisation comme Adam et L-BFGS pour la minimisation de la perte.

## Installation

Assurez-vous d'avoir les dépendances nécessaires installées. Vous pouvez utiliser `pip` pour installer les bibliothèques suivantes :

```bash
pip install torch matplotlib scikit-learn pyDOE scipy
```

## Structure du projet

Voici les principaux fichiers du projet :

- **main.py** : Script principal où le modèle est défini et entraîné.
- **README.md** : Documentation du projet.
- **utils.py** : (Optionnel) Contient des fonctions utilitaires comme le traçage des courbes 3D.
  
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

```python
# Charger les bibliothèques nécessaires
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

# Définir la fonction réelle de la PDE
def f_real(x, t):
    return torch.exp(-t) * torch.sin(np.pi * x)

# Définir le modèle PINN
class FCN(nn.Module):
    def __init__(self, layers):
        super(FCN, self).__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        a = x.float()
        for i in range(len(self.linears) - 1):
            a = self.activation(self.linears[i](a))
        return self.linears[-1](a)

# Initialisation et entraînement...
```

## Résultats

Une fois le modèle formé, les résultats peuvent être visualisés en 3D pour observer la solution de l'EDP dans le domaine spatial et temporel. 


