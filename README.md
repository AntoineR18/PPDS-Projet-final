# Marathon de Paris 2026

**Elèves**: Antoine Rustenholz, Jasmin Neveu, Aziz Seghaier

## Données

- **Source**: API SportInnovation
- **Taille**: ~57 000 coureurs
- **Features**: caractéristiques athlète + 10 splits intermédiaires

## Modèles

- **Ridge**: régression régularisée L2
- **ElasticNet**: régression régularisée L1 + L2

Les hyperparamètres sont sélectionnés par validation croisée à 5 folds (critère : RMSE).

## Installation

```bash
# Création de l'environnement virtuelle
uv venv

# Activation
source .venv/bin/activate

# Installation des dépendances
uv sync
```

## Structure du projet

```
.
├── data/
│   ├── donnees_finales.parquet   # Jeu de données final
│   ├── fetch_data.py             # Script de collecte API
│   └── pre_process.py            # Preprocessing
├── src/model/
│   ├── bootstrap.py             # Intervalles de confiance bootstrap
│   ├── config.py                # Configuration globale
│   ├── data_engineering.py      # Pipeline de préparation des données
│   ├── train.py                 # Entraînement Ridge et ElasticNet
│   └── visualization.py         # Graphiques
├── main.ipynb                   # Notebook principal
└── pyproject.toml              # Dépendances Python
```

## Utilisation

Ouvrir `main.ipynb` et exécuter les cellules dans l'ordre :

1. Import des libraries et configuration
2. Construction du dataset (`data_engineering`)
3. Entraînement des modèles (`train`)
4. Évaluation sur le jeu de test
5. Bootstrap pour les intervalles de confiance des coefficients
6. Visualisations

Pour changer le nombre de splits utilisés, modifier `N_SPLITS` dans `src/model/config.py`.
