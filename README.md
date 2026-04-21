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

# Installation des dépendances

```uv sync```

# Importation des données

1. Devoir Gitclone depuis une instance SSPCloud pour avoir accès à la bucket s3
2. Lancer le fichier retreive_data_from_sspcloud.py avec la commande ```uv run data/retreive_data_from_sspcloud.py```

## Utilisation

Ouvrir `main.ipynb` et exécuter les cellules dans l'ordre :

1. Import des libraries et configuration
2. Construction du dataset (`data_engineering`)
3. Entraînement des modèles (`train`)
4. Évaluation sur le jeu de test
5. Bootstrap pour les intervalles de confiance des coefficients
6. Visualisations

Pour changer le nombre de splits utilisés, modifier `N_SPLITS` dans `src/model/config.py`.
