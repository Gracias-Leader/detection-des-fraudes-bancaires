# Détection de la Fraude Bancaire par Machine Learning

## 📌 Présentation du Projet
Ce projet implémente une solution d'intelligence artificielle dédiée à la détection précoce des transactions frauduleuses dans le secteur bancaire. L'objectif est de minimiser les pertes financières de l'institution tout en préservant l'expérience client grâce à une double approche de modélisation (supervisée et non supervisée).

Le jeu de données comprend **100 000 transactions** simulées, reflétant les dynamiques des flux financiers (virements, retraits, Mobile Money) avec une problématique majeure de **fort déséquilibre des classes** (~1% de fraudes).

## 🚀 Fonctionnalités Clés & Feature Engineering
* **Analyse Temporelle :** Extraction de l'heure (`heure_transaction`) et du jour de la semaine (`jour_semaine`) pour capturer les anomalies comportementales (ex: transactions frauduleuses nocturnes).
* **Gestion des Données Manquantes :** Traitement stratégique de la colonne `marchand` (conservation de l'information sous la catégorie `'Inconnu'`).
* **Encodage & Normalisation :** Traitement des variables catégorielles par One-Hot Encoding (`pd.get_dummies`) pour une compatibilité algorithmique optimale.
* **Traitement du Class Imbalance :** Utilisation du paramètre dynamique `scale_pos_weight` pour pénaliser les erreurs sur la classe minoritaire (fraude).

## 🛠️ Technologies & Librairies utilisées
* **Langage :** Python 3.x
* **Manipulation & Visualisation :** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning Supervisé :** Scikit-Learn (`RandomForestClassifier`), XGBoost (`XGBClassifier`)
* **Machine Learning Non Supervisé :** Scikit-Learn (`IsolationForest`)

## 📊 Stratégie de Modélisation
Pour maximiser l'efficacité de la détection, deux approches complémentaires ont été déployées :
1. **Approche Supervisée (XGBoost & Random Forest) :** Entraînés sur les données historiques pour reconnaître les patterns de fraude connus.
2. **Approche Non Supervisée (Isolation Forest) :** Utilisée pour isoler les comportements atypiques et détecter d'éventuelles nouvelles techniques de fraude non étiquetées.

### Métrique d'Évaluation Critique
En détection de fraude, l'exactitude globale (*Accuracy*) est trompeuse. Ce projet priorise le **Rappel (Recall)** pour la classe fraude ($1$), car le coût métier d'un *Faux Négatif* (une fraude non détectée) est largement supérieur au coût d'un *Faux Positif* (une transaction légitime bloquée pour vérification).

## 📈 Résultats et Performance
*Les métriques seront complétées après la phase finale d'entraînement.*

| Modèle | Précision (Classe 1) | Rappel / Recall (Classe 1) | F1-Score |
| :--- | :--- | :--- | :--- |
| **XGBoost** (avec `scale_pos_weight`) | *À compléter %* | *À compléter %* | *À compléter %* |
| **Random Forest** | *À compléter %* | *À compléter %* | *À compléter %* |
| **Isolation Forest** | *À compléter %* | *À compléter %* | *À compléter %* |

## 📁 Structure du Répertoire
```text
├── farudes_bancaires.ipynb   # Notebook principal (Data Prep, EDA, Modélisation)
├── README.md                 # Description du projet et livrables
└── donnees/                  # Répertoire contenant le dataset (si applicable)
