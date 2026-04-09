import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Chargement des données (à remplacer par votre DataFrame complet)
df = pd.read_csv('data/cv_dataset.csv')


# 3. Feature Engineering
# Transformation des colonnes textuelles (skills, certifications) en variables numériques (nombre d'éléments)
df['nb_skills'] = df['skills'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
df['nb_certifications'] = df['certifications'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

# 4. Séparation des caractéristiques (X) et de la cible (y)
# On exclut 'cv_id' qui n'est pas prédictif, ainsi que les colonnes textuelles brutes
X = df.drop(columns=['cv_id', 'passed_next_stage', 'skills', 'certifications'])
y = df['passed_next_stage'].astype(int) # On s'assure que la cible est un entier (0 ou 1)

# 5. Définition des groupes de colonnes pour le prétraitement
num_features = [
    'age', 'distance_ville_haute_km', 'total_experience_years',
    'total_gap_months', 'nb_gaps', 'education_score',
    'number_of_experiences', 'lang_fr', 'lang_en', 'lang_de',
    'lang_es', 'lang_it', 'lang_other_score_sum',
    'nb_skills', 'nb_certifications'
]

cat_features = [
    'target_role', 'education_degree', 'education_field', 'education_school'
]

# 6. Création du pipeline de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            # Remplace les NaN par la médiane de la colonne (très utile pour distance_ville_haute_km)
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', Pipeline(steps=[
            # Remplace les NaN catégoriels par 'missing'
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            # Encodage One-Hot pour les variables catégorielles
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

# 7. Création du pipeline final avec l'algorithme de classification
# Le Random Forest est performant, résiste bien au surapprentissage et donne l'importance des variables.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 8. Séparation en jeu d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 9. Entraînement du modèle
print("Entraînement du modèle en cours...")
model.fit(X_train, y_train)

# 10. Évaluation du modèle
y_pred = model.predict(X_test)
print("\n--- Rapport de Classification ---")
print(classification_report(y_test, y_pred))
print(f"Précision Globale (Accuracy) : {accuracy_score(y_test, y_pred):.2%}")