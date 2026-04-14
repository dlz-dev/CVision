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

## 3. Création de Variables (Feature Engineering)
Pour aider le modèle à capter de meilleurs signaux, le dataset original a été enrichi avec de nouvelles variables calculées (Feature Engineering) :
* **`avg_gap_duration`** : Durée moyenne d'un trou dans le CV (`total_gap_months / (nb_gaps + 1)`).
* **`gap_ratio`** : Le ratio de la carrière passée en période d'inactivité (trous par rapport au temps global d'expérience).
* **`skills_count`** : Le nombre total de compétences listées.
* **`certif_count`** et **`has_certif`** : Le nombre total de certifications, ainsi qu'une variable binaire (0 ou 1) indiquant simplement si le candidat possède au moins une certification.

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
Le modèle retenu est le **`LogisticRegressionCV`**. Les expériences menées dans la V1 du notebook ont montré que les modèles plus complexes (comme les Random Forest ou le Gradient Boosting) n'apportaient pas de meilleures performances et surapprenaient face au faible volume de données (200 lignes).

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

Ce point correspond donc au meilleur compromis selon l’objectif métier :  
Il faut donc **minimiser les faux positifs**, même si cela implique de rater certains cas pertinents.

### Analyse de la courbe Precision-Recall
La courbe PR (AP = 0.569) confirme ce choix :
- Le point correspondant au seuil **0.70** donne une **precision d’environ 0.60** pour un **recall d’environ 0.60**.
- Au-delà de ce seuil, les gains en précision deviennent faibles tandis que le recall chute fortement.
- En dessous de ce seuil, le recall augmente mais au prix d’une dégradation rapide de la précision.

### Conclusion sur le seuil
Le seuil de **0.70** permet donc :
- Une **meilleure qualité des prédictions** (moins de faux positifs)
- Une **cohérence avec la métrique F0.5**
- Un comportement adapté à un contexte métier où la **fiabilité prime sur l’exhaustivité**

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
* L'ensemble est sauvegardé via la librairie `joblib` sous le fichier `modele_classification_cv.joblib`. Cela permet à une application de charger un seul fichier pour transformer et prédire de nouveaux CVs.