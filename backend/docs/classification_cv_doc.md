# Documentation du Notebook : Classification de CVs

## 1. Objectif Général du Notebook
Ce notebook a pour but de construire un modèle de Machine Learning capable de prédire la variable `passed_next_stage`, c'est-à-dire déterminer si un candidat passera à l'étape suivante du processus de recrutement.

**L'enjeu métier principal** défini ici est de **maximiser la précision**. En d'autres termes, on veut être absolument certain de la qualité du CV avant de le recommander (éviter de faire perdre du temps aux recruteurs avec des faux positifs), quitte à être plus sévère et à écarter quelques bons profils.

Le pipeline complet suit un flux de travail rigoureux : 
Chargement et exploration ➔ Feature Engineering ➔ Preprocessing ➔ Modélisation (LR L1) avec validation croisée ➔ Optimisation du seuil sur la métrique F0.5 ➔ Évaluation ➔ Sauvegarde.

---

## 2. L'enjeu critique : Le Dataset Déséquilibré (Imbalanced Data)
Lors de l'exploration, on constate que le jeu de données est fortement déséquilibré : la classe cible (`passed_next_stage = 1`) ne représente qu'environ **25.5 %** des CVs (51 sur 200). 

**Pourquoi faire attention ?** Si un modèle se contentait de dire "Refusé" à tous les CVs, il aurait une "Exactitude" (Accuracy) d'environ 75%, ce qui paraîtrait bien sur le papier, mais serait totalement inutile en réalité. Ce déséquilibre oblige à adapter toute la suite du pipeline : 
* Utilisation du paramètre `stratify` lors de la séparation des données.
* Utilisation de la méthode SMOTE pour générer des données synthétiques de la classe minoritaire.
* Ajout de `class_weight='balanced'` dans le modèle.
* Choix de métriques adaptées.

---

## 3. Variables utilisées

### Variables brutes (issues du dataset original)

| Variable | Type         | Description |
|---|--------------|---|
| `age` | Numérique    | Âge du candidat |
| `distance_ville_haute_km` | Numérique    | Distance entre le domicile et la ville du poste (en km) |
| `total_experience_years` | Numérique    | Nombre total d'années d'expérience |
| `nb_gaps` | Numérique    | Nombre de trous dans le parcours professionnel |
| `total_gap_months` | Numérique    | Durée totale des trous (en mois) — utilisée dans le feature engineering, non retenue directement en V4 |
| `education_score` | Numérique    | Score numérique associé au niveau de formation |
| `lang_fr`, `lang_en`, `lang_de`, `lang_es`, `lang_it` | Numérique    | Indicateurs de maîtrise des langues |
| `lang_other_score_sum` | Numérique    | Score agrégé des autres langues déclarées |
| `target_role` | Catégorielle | Poste visé par le candidat |
| `education_degree` | Catégorielle | Diplôme obtenu |
| `education_field` | Catégorielle | Domaine d'études |
| `skills` | Texte        | Liste des compétences déclarées (séparées par des virgules) |
| `certifications` | Texte        | Liste des certifications obtenues (séparées par des virgules) |

### Variables créées par Feature Engineering

Pour aider le modèle à capter de meilleurs signaux, le dataset original a été enrichi avec des variables calculées :

| Variable | Formule | Objectif |
|---|---|---|
| `avg_gap_duration` | `total_gap_months / (nb_gaps + 1)` | Durée moyenne d'un trou. *Présente dans V1 uniquement — écartée en V4 car fortement colinéaire avec `nb_gaps` (VIF élevé).* |
| `gap_ratio` | `total_gap_months / (total_experience_years * 12 + total_gap_months + 1)` | Part de la carrière passée en inactivité |
| `skills_count` | Comptage des éléments de la liste `skills` | Nombre total de compétences déclarées |
| `certif_count` | Comptage des éléments de la liste `certifications` | Nombre total de certifications obtenues |
| `has_certif` | `1 si certif_count > 0, sinon 0` | Indicateur binaire : le candidat a-t-il au moins une certification ? |

**Jeu de variables final retenu (V4) :** toutes les variables brutes sauf `total_gap_months` (remplacée par `gap_ratio`), et sans `avg_gap_duration` (colinéarité). Enrichi de `certif_count` et `has_certif`.

---

## 4. Choix du jeu de données (La sélection des Features)
Le notebook garde la trace de **plusieurs configurations de variables (V1 à V4)** testées empiriquement. On a trouvé le "bon" jeu de données (la version V4 active) après plusieurs itérations :
* **La version V1** incluait `avg_gap_duration`, mais cette variable causait un fort problème de **colinéarité** (mesuré via le VIF - Variance Inflation Factor) avec le nombre de trous (`nb_gaps`), ce qui perturbait la régression logistique.
* **Les V2 et V3** étaient des "baselines" (jeux de référence) allégées.
* **Le jeu final (V4)** a été retenu car, expurgé des variables trop corrélées et enrichi de `has_certif` et `certif_count`, il offre le meilleur équilibre d'informations pour la prédiction.

---

## 5. Explication des étapes de Preprocessing
La préparation des données est gérée de manière très robuste par un `ColumnTransformer` :

* **`stratify=y` (dans le `train_test_split`)** : Ce paramètre est capital à cause du déséquilibre abordé plus haut. Il s'assure que vos jeux d'entraînement et de test contiennent exactement la même proportion de CV sélectionnés (25,5%). Sans cela, par malchance, le jeu de test pourrait ne contenir aucun CV positif.
* **`StandardScaler`** : Centre (moyenne = 0) et réduit (écart-type = 1) les variables numériques. C'est **obligatoire** pour que la régression logistique (qui utilise une pénalité L1) puisse comparer les variables équitablement. Sinon, une variable avec de grandes valeurs (comme la distance en km) écraserait une variable avec de petites valeurs (comme les années d'expérience).
* **`OneHotEncoder`** : Transforme les variables catégorielles (ex: le diplôme ou le rôle visé) en plusieurs colonnes binaires (0 ou 1) que le modèle mathématique peut interpréter.
* **`TfidfVectorizer`** : Transforme les champs textuels (Skills et Certifications) en scores mathématiques (Term Frequency-Inverse Document Frequency) donnant plus de poids aux mots importants et rares. 
  * **`max_features`** : On l'utilise (ex: max_features=20 pour les skills) pour imposer une limite au vocabulaire. Cela évite "la malédiction de la dimensionnalité" (avoir plus de colonnes que de lignes dans le jeu de données), limitant ainsi considérablement le risque de surapprentissage (overfitting).

---

## 6. Choix du Modèle : Régression Logistique avec Validation Croisée

### Pourquoi ce modèle ?
Le modèle retenu est le **`LogisticRegressionCV`**. Les expériences menées dans la V1 du notebook ont montré que les modèles plus complexes (comme les Random Forest ou le Gradient Boosting) n'apportaient pas de meilleures performances et surapprenaient face au faible volume de données (200 lignes).

La régression logistique est ici le point de départ naturel : interprétable, robuste sur peu de données, et compatible avec une pénalité L1 qui agit comme un filtre de variables automatique.

### Paramètres du modèle

| Paramètre | Valeur | Justification |
|---|---|---|
| `penalty` | `'l1'` | Sélectionne automatiquement les variables en mettant les coefficients inutiles à 0 (contrairement à L2 qui les réduit sans les éliminer) |
| `solver` | `'liblinear'` | Seul solver compatible avec la pénalité L1 pour de petits datasets |
| `Cs` | `10` | Grille de 10 valeurs de C testées en interne par `LogisticRegressionCV` |
| `cv` | `StratifiedKFold(n_splits=5)` | Validation croisée interne stratifiée pour sélectionner le meilleur C |
| `scoring` | `'roc_auc'` | Critère d'optimisation interne de C |
| `class_weight` | `'balanced'` | Compense le déséquilibre de classes en pondérant automatiquement les observations |
| `max_iter` | `1000` | Assure la convergence même sur des données difficiles |
| `random_state` | `42` | Reproductibilité |

### Gestion du déséquilibre : SMOTE

En amont du classifieur dans la pipeline, un **SMOTE** (`k_neighbors=5`, `random_state=42`) est appliqué sur le jeu d'entraînement. Il génère des exemples synthétiques de la classe minoritaire (CVs sélectionnés) par interpolation entre voisins proches, pour équilibrer la distribution avant l'apprentissage.

### Validation croisée externe

Une cross-validation externe (5 folds stratifiés, `StratifiedKFold`) est lancée sur l'ensemble du pipeline pour évaluer la stabilité des métriques :

| Métrique | Score moyen (CV) |
|---|---|
| ROC-AUC | ~affiché à l'exécution |
| Average Precision | ~affiché à l'exécution |
| F1 | ~affiché à l'exécution |

* **Pourquoi `StratifiedKFold` ?** Cette méthode de validation croisée découpe les données d'entraînement en 5 sous-ensembles (folds) tout en garantissant le ratio de 25,5% de positifs dans chaque fold. Cela permet de tester la stabilité du modèle et d'optimiser les hyperparamètres de manière fiable sur des petites données.
* **Pourquoi la pénalité `l1` (Lasso) ?** Les tests ont montré que L2 (Ridge) ou Elastic-Net étaient moins performants. C'est parce que la pénalité **L1 agit comme un sélecteur de variables naturel**. Elle "éteint" (met le coefficient à 0) purement et simplement les variables ou les mots issus du TF-IDF qui ne servent à rien, débruitant ainsi le modèle, là où la L2 se contenterait de les réduire.

---

## 7. Évaluation et Choix du Seuil (Threshold)

### L'utilisation de la métrique F0.5
Usuellement, on utilise le F1-Score qui est la moyenne harmonique exacte entre la **Précision** et le **Rappel (Recall)**. 
Cependant, ici, on a délibérément choisi le **F0.5-score**. Mathématiquement, cette métrique donne **deux fois plus de poids à la Précision qu'au Rappel**.
* **Pourquoi dans notre cas ?** Le métier veut recruter efficacement. Un "Faux Positif" (recommander un mauvais candidat) fait perdre un temps précieux en entretien aux équipes RH et aux managers. Un "Faux Négatif" (rejeter un bon candidat) est dommage, mais moins coûteux financièrement. Le F0.5 permet d'optimiser le modèle exactement sur ce besoin de sévérité et de fiabilité absolue.

### L'optimisation du Seuil
Par défaut, un modèle de classification binaire sépare les classes à une probabilité de **0.50** (50%).
Dans un contexte déséquilibré, et surtout avec la métrique métier F0.5 choisie, ce seuil standard n'est pas optimal.
À l'aide d'une courbe `Precision/Recall vs Seuil`, le notebook identifie et sélectionne le **seuil optimal** (qui se situe autour de **0.70** dans notre cas). 

Le modèle est donc volontairement "conservateur" : il n'envoie un CV à l'étape suivante que s'il est certain à plus de 70% de sa pertinence, maximisant ainsi la Précision au profit de la fiabilité métier. Heureuse nouvelle, le maximum de `precision` que l'on peut atteindre reste aligné avec une zone où le `recall` est encore raisonnable. Cela signifie qu'il est possible de trouver un compromis pertinent entre fiabilité des prédictions et couverture des cas positifs.

Sur la courbe **Precision/Recall vs Seuil**, on observe que :
- Lorsque le seuil augmente, la **precision** augmente globalement.
- En parallèle, le **recall** diminue progressivement.
- La métrique **F0.5**, qui favorise la précision, atteint son maximum autour du seuil **0.70**.

Ce point correspond donc au meilleur compromis selon l'objectif métier :  
Il faut donc **minimiser les faux positifs**, même si cela implique de rater certains cas pertinents.

### Analyse de la courbe Precision-Recall
La courbe PR (AP = 0.569) confirme ce choix :
- Le point correspondant au seuil **0.70** donne une **precision d'environ 0.60** pour un **recall d'environ 0.60**.
- Au-delà de ce seuil, les gains en précision deviennent faibles tandis que le recall chute fortement.
- En dessous de ce seuil, le recall augmente mais au prix d'une dégradation rapide de la précision.

### Conclusion sur le seuil
Le seuil de **0.70** permet donc :
- Une **meilleure qualité des prédictions** (moins de faux positifs)
- Une **cohérence avec la métrique F0.5**
- Un comportement adapté à un contexte métier où la **fiabilité prime sur l'exhaustivité**

Ce choix rend le modèle volontairement **conservateur**, ce qui est particulièrement pertinent dans un processus de tri de CV où chaque erreur positive a un coût élevé (temps humain, surcharge des recruteurs, etc.).

---

## 8. Matrice de Confusion et Évaluation sur le Jeu de Test
Une fois le seuil optimal (autour de 0.70) déterminé, le modèle est évalué de manière concrète sur le jeu de test. 
* Une matrice de confusion est générée pour identifier visuellement les Vrais Positifs (TP), Vrais Négatifs (TN), Faux Positifs (FP) et Faux Négatifs (FN).
* Cette étape permet de valider que l'objectif métier de minimiser drastiquement les faux positifs est bien respecté sur des données que le modèle n'a jamais vues lors de son entraînement.

---

## 9. Test sur de Nouveaux CVs (Données Unseen)
Pour simuler une mise en production, le pipeline est ensuite testé sur un tout nouveau lot de CVs. 
* Le modèle applique les transformations et calcule la probabilité de sélection pour chaque nouveau candidat.
* En appliquant le seuil optimisé, le modèle émet une décision claire (`Sélectionné` ou `Refusé`).
* Les prédictions et les probabilités associées sont présentées dans un tableau, confirmant la viabilité du modèle à classer correctement de nouveaux profils.

---

## 10. Sauvegarde du Modèle
La dernière étape du notebook permet de persister le travail accompli pour un usage futur.
* Le script crée automatiquement un dossier `../models` s'il n'existe pas déjà.
* Afin d'assurer une reproductibilité parfaite et de simplifier le déploiement, **la pipeline complète** (incluant l'imputation, la standardisation et l'encodage TF-IDF) ainsi que **le seuil optimal** sont regroupés dans un même dictionnaire.
* L'ensemble est sauvegardé via la librairie `joblib` sous le fichier `model_classification_cv.joblib`. Cela permet à une application de charger un seul fichier pour transformer et prédire de nouveaux CVs.