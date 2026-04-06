import joblib                                        # pour sauvegarder le modèle
from pathlib import Path                             # pour gérer les chemins
from sklearn.ensemble import RandomForestClassifier  # le modèle ML
from sklearn.model_selection import cross_val_score  # pour évaluer le modèle
from core.features import load_features              # charge les features et labels


# charge les features et les labels depuis les JSONs et le CSV
X, y = load_features("data/processed", "data/labels.csv")

# affiche un aperçu des données
print("Données chargées :")
print(X)
print("\nLabels :", y.values)

# crée le modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# évalue le modèle par cross-validation (sur nos 5 CVs c'est limité mais utile)
scores = cross_val_score(model, X, y, cv=2, scoring="accuracy")
print(f"\nPrécision moyenne : {scores.mean():.2f}")

# entraîne le modèle sur toutes les données
model.fit(X, y)

# crée le dossier models/ s'il n'existe pas
Path("models").mkdir(exist_ok=True)

# sauvegarde le modèle entraîné
joblib.dump(model, "models/model.pkl")
print("\nModèle sauvegardé dans models/model.pkl")