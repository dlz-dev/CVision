# Audit d'équité — Modèle de classification de CVs

> **CVision** · Documentation des méthodes · B2 IA · HELMo
> Notebook de référence : [`backend/notebooks/fairness_audit.ipynb`](../notebooks/fairness_audit.ipynb)

---

## Table des matières

1. [Introduction & contexte légal](#introduction--contexte-légal)
2. [Méthodologie générale de l'audit](#méthodologie-générale-de-laudit)
3. [Données & setup](#données--setup)
4. [Attributs sensibles & groupes protégés](#attributs-sensibles--groupes-protégés)
5. [Métriques d'équité — choix et justification](#métriques-déquité--choix-et-justification)
6. [Tests statistiques — Chi², Fisher exact & bootstrap](#tests-statistiques--chi-fisher-exact--bootstrap)
7. [Analyse par attribut & intersectionnalité](#analyse-par-attribut--intersectionnalité)
8. [Disparités opérationnelles (rôle visé)](#disparités-opérationnelles-rôle-visé)
9. [Explicabilité du modèle](#explicabilité-du-modèle)
10. [Stratégie corrective retenue](#stratégie-corrective-retenue)
11. [Comparaison ancien vs nouveau modèle](#comparaison-ancien-vs-nouveau-modèle)
12. [Explication individuelle d'une décision](#explication-individuelle-dune-décision)
13. [Conclusion](#conclusion)
14. [Glossaire](#glossaire)

---

## Introduction & contexte légal

Le projet CVision développe un modèle qui trie automatiquement les CV pour décider si un candidat passe à l'étape suivante (entretien) ou non. Quelques mois après le déploiement, l'équipe de conformité RH de LuxTalent a observé que certains profils étaient invités à des taux significativement différents. Le modèle ayant été entraîné sur des décisions historiques, il pouvait avoir absorbé des biais implicites — l'audit ci-dessous a été commandé pour le démontrer (ou l'écarter) chiffres à l'appui, et corriger le tir si nécessaire.

La question centrale : **est-ce que le modèle traite tout le monde de façon équitable ?** En particulier, est-ce qu'à *qualification égale* les candidats sont traités identiquement quel que soit leur âge, leur origine présumée (langues, distance), ou leur école ?

### Cadre éthique — *AI4People* (Floridi et al.)

> **Source de ce cadre :** la conférence sur l'éthique des systèmes algorithmiques, dont la synthèse est disponible dans [`ethique_synthese.md`](ethique_synthese.md), a présenté le projet *AI4People* de Luciano Floridi comme référence conceptuelle centrale. Les cinq principes définis dans ce cadre (§1 de la synthèse) ont directement orienté les choix techniques de cet audit. Le tableau ci-dessous les relie explicitement à chaque décision prise.

| Principe (*AI4People*) | Question concrète | Pourquoi ce principe s'applique ici | Traduction technique dans cet audit |
|---|---|---|---|
| **Bienfaisance** | L'IA fait-elle le bien ? | Un outil de recrutement automatisé influence des trajectoires professionnelles réelles. Mal calibré, il peut systématiquement exclure des profils capables. | Modèle équitable proposé (Stratégie 1, §10) |
| **Non-malfaisance** | L'IA évite-t-elle de nuire ? | Présenter des métriques calculées sur les données d'entraînement ou sur un seul ré-échantillon gonflerait artificiellement les résultats et donnerait à LuxTalent une fausse impression de sécurité. | Tests stats **out-of-sample uniquement**, IC bootstrap pour quantifier l'incertitude plutôt que la masquer |
| **Autonomie** | L'humain garde-t-il le contrôle ? | L'AI Act (Art. 14) et le RGPD (Art. 22) exigent qu'un candidat refusé puisse obtenir une explication et contester. Un recruteur qui ne comprend pas la décision ne peut ni l'expliquer ni la corriger. | Explication individuelle lisible par le recruteur (§12) |
| **Justice & Équité** | Traite-t-elle tout le monde pareil ? | La conférence a insisté sur le fait que l'équité doit être mesurée *à qualification égale*, pas sur des groupes aux niveaux de compétence différents — d'où le rejet de la DP brute au profit de l'EOD. | EOD (Equal Opportunity Difference) comme métrique principale (§5) |
| **Explicabilité** | Peut-on comprendre ses décisions ? | Sans explicabilité, l'effet "boîte noire" s'installe (cf. [`ethique_synthese.md`](ethique_synthese.md) §1). Le recruteur ne peut pas détecter une erreur ni justifier un refus. | Triple couche L1 + SHAP + log-odds (§9, §12) |

### Cadre légal — pourquoi cet audit est une obligation

L'**Annexe III de l'AI Act** classe les outils automatisés de présélection de CV comme **systèmes à haut risque**. Cela impose à LuxTalent les obligations suivantes :

| Article AI Act | Obligation | Couvert dans cet audit |
|---|---|---|
| **Art. 9** | Système de gestion des risques documenté | Document entier |
| **Art. 10** | Gouvernance des données, détection et correction des biais | §4 à §11 |
| **Art. 13** | Transparence envers les déployeurs (informer le client) | §9 (SHAP global) |
| **Art. 14** | Supervision humaine effective (le RH garde la main) | §12 (explication individuelle) |
| **Art. 15** | Exactitude et robustesse | §6 (bootstrap), §11 (perf avant/après) |

> Autres références juridiques : Directive 2000/43/CE (origine ethnique), Directive 2000/78/CE (âge), RGPD Art. 9 (données sensibles) et Art. 22 (décisions automatisées), Loi belge du 10/05/2007 contre la discrimination.

---

## Méthodologie générale de l'audit

L'audit suit **8 étapes** explicites, chacune justifiée méthodologiquement et tracée dans le notebook.

| Étape | Ce qui est fait | Pourquoi c'est nécessaire |
|---|---|---|
| **1.** Reproduction exacte du split train/test original | Recalcul de `train_test_split(test_size=0.2, random_state=42)` | Seules les métriques out-of-sample sont fiables. Évaluer sur le train gonfle les TPR et masque les biais — c'est la règle de non-malfaisance (§1 ci-dessus). |
| **2.** Définition des attributs sensibles et bucketisation | Groupes protégés définis avant tout calcul | Si les groupes sont définis *après* avoir vu les écarts, on risque de cherry-picker les découpages favorables. Ils sont fixés a priori, conformément aux directives légales. |
| **3.** Calcul des métriques d'équité par groupe | Selection rate · DP Gap · DI · TPR/FPR · EOD | Aucune métrique seule n'est suffisante. Leur combinaison révèle des biais que chacune prise isolément manquerait. |
| **4.** Validation statistique de chaque écart | Fisher exact (2×2), Chi² (3×2), IC bootstrap 95% | Un écart visible sur un graphique peut être dû au hasard avec n=100. On ne tire de conclusions que sur des écarts statistiquement significatifs. |
| **5.** Analyse intersectionnelle | Heatmaps Âge × Francophonie, Âge × Géographie | La conférence (cf. [`ethique_synthese.md`](ethique_synthese.md) §1 — *Focus biais et intersectionnalité*) a insisté : un modèle peut être équitable sur chaque attribut isolé et discriminer sur leur combinaison. |
| **6.** Disparités opérationnelles | Analyse par `target_role` | Répondre à la question 3 du cahier des charges : les écarts sont-ils liés au poste (légitimes) ou à des attributs protégés (suspects) ? |
| **7.** Explicabilité | Coefficients L1 + SHAP + log-odds individuels | Obligation légale (Art. 13 et 14 AI Act) et principe d'Autonomie : le recruteur doit pouvoir comprendre et contester chaque décision. |
| **8.** Stratégie corrective + comparaison avant/après | Suppression features, split anti-leakage 60/20/20, seuil F-beta sur validation | Corriger sans créer de leakage. Documenter le coût en performance pour que LuxTalent prenne une décision éclairée sur le trade-off équité/performance. |

### Trois règles d'or appliquées dans tout l'audit

**1. Out-of-sample uniquement.**
Toutes les métriques d'équité sont calculées sur le test set (100 candidats) — jamais sur le train. *Pourquoi :* évaluer un modèle sur ses données d'entraînement gonfle artificiellement les TPR et FPR, et masque les biais. C'est la règle de **Non-malfaisance** en pratique : ne pas présenter des chiffres flatteurs qui donneraient à LuxTalent une fausse impression de sécurité.

**2. Pas de data leakage.**
Le seuil de décision du modèle FAIR est calibré sur un **set de validation séparé** (100 candidats), pas sur le test. Le test set n'est touché qu'une seule fois pour le rapport final. *Pourquoi :* si on optimise le seuil sur le test set, les métriques finales sont biaisées à la hausse — on aurait "appris" le test set sans le dire. C'est une forme de fraude méthodologique classique dans les projets IA appliqués.

**3. Honnêteté sur l'incertitude.**
Avec seulement 20 candidats positifs au test, certaines métriques (TPR Junior calculé sur 2 personnes) sont très bruitées. On le quantifie via un IC bootstrap au lieu de présenter des chiffres ponctuels comme s'ils étaient stables. *Pourquoi :* la conférence a rappelé que la **Non-malfaisance** inclut la *capability caution* — ne pas laisser croire que l'IA est plus certaine qu'elle ne l'est réellement. Un EOD point à 1.00 avec un IC [0.6, 1.0] ne raconte pas la même histoire qu'un EOD à 1.00 avec un IC [0.95, 1.0].

---

## Données & setup

| Élément | Valeur |
|---|---|
| Dataset | 500 candidats, 21 colonnes |
| Taux de sélection réel (`passed_next_stage = 1`) | 20% (déséquilibre 80/20) |
| Modèle audité | `model_classification_cv_cool.joblib` (LogisticRegression L1) |
| Seuil de décision optimal (modèle d'origine) | 0.1434 |

### Split train/test reproduit à l'identique

```
train_test_split(test_size=0.2, random_state=42, stratify=y)
↓
Train : 400 candidats  (jamais utilisés pour calculer des métriques d'équité)
Test  : 100 candidats  ← base de tout l'audit (out-of-sample)
```

Sur le test set, le modèle d'origine prédit un taux de sélection de **34%** (contre 20% en réalité). C'est un comportement attendu : avec `class_weight='balanced'` et un seuil bas (0.14), le modèle est calibré pour ne pas rater les positifs au prix d'un excès de faux positifs — utile pour le rappel mais accentue les disparités de selection rate.

### Conséquence du déséquilibre 80/20

Le test set ne contient que **20 candidats qualifiés** (positifs réels). Plusieurs métriques sont mécaniquement bruitées :

- TPR par groupe : Junior 2 qualifiés, Mid 12, Senior 6.
- Cellules intersectionnelles : certaines tombent à 0-3 individus.

C'est précisément pour cette raison que le §6 ajoute des **IC bootstrap** et que les Chi² sur les contingences 3×2 sont annotés "cellules attendues <5".

---

## Attributs sensibles & groupes protégés

Avant tout calcul, on définit les **groupes protégés** — les caractéristiques sur lesquelles le modèle ne devrait pas discriminer.

| Attribut | Groupes (effectifs) | Base légale |
|---|---|---|
| `age` | Junior ≤29 (181) · Mid 30–34 (178) · Senior ≥35 (141) | Directive 2000/78/CE · Loi belge 2007 |
| `distance_ville_haute_km` | Local <1 000 km (159) · Régional (130) · International >5 000 km (211) | RGPD Art. 9 — proxy d'origine |
| `lang_fr` | Francophone ≥4 (128) · Non-francophone (372) | Directive 2000/43/CE |
| `education_degree` | Master+ (209) · Bachelor ou moins (291) | AI Act Annexe III |
| `education_score` | École de prestige 4 (209) · École standard 3 (291) | AI Act Art. 10 |

> ⚠ `education_degree` et `education_score` sont **parfaitement corrélés** (291/209 dans les deux cas). Les métriques sont donc identiques — on les conserve pour la traçabilité mais ce sont effectivement **4 attributs distincts**, pas 5.

### Pourquoi ces attributs — la logique du proxy

Le dataset ne contient **pas** les attributs sensibles "purs" (genre, nationalité, origine ethnique). Mais l'AI Act et le RGPD considèrent qu'**une feature qui révèle indirectement un attribut protégé est elle-même sensible** — c'est le principe du *proxy*.

La conférence (cf. [`ethique_synthese.md`](ethique_synthese.md) §1 — *Focus sur l'Équité, les Biais et la Non-discrimination*) a insisté sur le fait que les discriminations algorithmiques transitent souvent par des données apparemment neutres. Cela justifie l'examen des features suivantes :

| Feature suspecte | Proxy de quoi ? | Pourquoi c'est un proxy | Justification de l'inclusion dans l'audit |
|---|---|---|---|
| `lang_de`, `lang_es`, `lang_it` | Nationalité / origine | Parler allemand corrèle avec être allemand. Favoriser ces langues = discriminer indirectement par nationalité. | Directive 2000/43/CE — discrimination sur l'origine nationale |
| `distance_ville_haute_km` | Origine géographique | Distance à Liège élevée ⇒ candidat non européen probablement. | RGPD Art. 9 — données révélant indirectement l'origine |
| `lang_fr` | Origine francophone | Parler français corrèle avec une origine belge ou française. | Cas **ambigu** : aussi une compétence métier légitime en Belgique (cf. §10 pour la décision retenue) |
| `education_score` | Statut socio-économique | École de prestige corrèle avec milieu social aisé. | AI Act Art. 10 — biais dans les données d'entraînement |

### `target_role` volontairement exclu

Le rôle visé (`Software Engineer`, `Data Scientist`, etc.) n'est **pas** un attribut protégé : c'est la définition du poste. Des écarts de selection rate entre rôles sont attendus (un poste senior demande plus d'expérience) et **légitimes**. On analyse cependant `target_role` en §8 comme "disparité opérationnelle" pour répondre à la question 3 du cahier des charges ("les disparités sont-elles justifiées par le poste ?").

---

## Métriques d'équité — choix et justification

On combine **trois métriques complémentaires** plus un test statistique systématique. Aucune métrique seule n'est suffisante — c'est leur combinaison qui donne un audit robuste.

### Les trois métriques

**1. Selection rate** — taux de candidats prédits comme sélectionnés dans un groupe.
```
selection_rate(a) = P(Ŷ=1 | A=a)
```
Sert de référence pour calculer les deux suivantes.

**2. Demographic Parity Gap (DP Gap)** — vue collective.
```
DP Gap = max(selection_rate) − min(selection_rate)
Seuil d'alerte : > 0.10
```
Mesure l'écart brut entre groupes. Question : *les groupes sont-ils sélectionnés à des taux comparables ?* **Trompeur si les base rates de qualification diffèrent réellement.**

**3. Disparate Impact Ratio (DI)** — version normée.
```
DI = min(selection_rate) / max(selection_rate)
Seuil d'alerte : < 0.80
```
Inspiré de la **règle des 80%** (jurisprudence US, reprise par l'AI Act) : un groupe sélectionné à moins de 80% du taux du groupe favorisé est présumé victime de discrimination. Même limite que DP : insensible aux base rates.

**4. Equal Opportunity Difference (EOD)** ← *notre métrique principale*
```
TPR(a) = P(Ŷ=1 | A=a, Y=1)
EOD Gap = max(TPR) − min(TPR)
Seuil d'alerte : > 0.10
```
Vue individuelle : *parmi les candidats **vraiment** qualifiés (`Y=1`), le modèle les détecte-t-il aussi bien dans tous les groupes ?* C'est l'**Equal Opportunity** de Hardt et al. (2016), recommandé dans la littérature fairness pour les tâches où les base rates diffèrent légitimement.

### Pourquoi l'EOD prime ici — arbre de décision

La conférence (cf. [`ethique_synthese.md`](ethique_synthese.md) §1 — principe de **Justice & Équité**) a rappelé que l'équité doit s'évaluer *à qualification égale*. Cela rend la DP brute inadaptée dès lors que les groupes ont des niveaux de qualification différents en réalité. L'arbre suivant formalise ce raisonnement :

```
1. Les base rates de qualification diffèrent-ils entre groupes ?
   → Oui (cf. tableau ci-dessous) → EOD obligatoire, DP/DI complémentaires
   → Non → DP suffirait

2. Si on observe DP Gap = 0.5 mais EOD Gap = 0.0
   → différence reflète seulement les qualifications réelles, pas un biais

3. Si on observe DP Gap petit mais EOD Gap = 1.0
   → biais caché : à qualification égale, un groupe est moins détecté
```

| Groupe | Base rate (taux réel qualifié) |
|---|---|
| Junior (≤29) | 10.5% |
| Mid (30–34) | 22.5% |
| Senior (≥35) | 29.1% |
| Bachelor− | 12.7% |
| Master+ | 30.1% |

Les écarts de base rate sont **importants** (Senior 3× plus qualifié que Junior). La DP brute condamnerait à tort le modèle pour avoir simplement reflété cette réalité. L'**EOD est la seule métrique qui isole le biais du modèle des qualifications réelles** — c'est notre métrique principale, les autres servent de complément descriptif.

> **Limite du DI** : le Disparate Impact Ratio est un rapport min/max de taux de sélection. Avec de petits effectifs (2 Juniors qualifiés au test), le dénominateur peut fluctuer beaucoup d'un ré-échantillonnage à l'autre. Cela explique pourquoi le DI min semble se dégrader dans le modèle FAIR (§11) même quand l'EOD s'améliore : c'est un artefact de variance d'échantillonnage, pas un vrai recul de l'équité. L'EOD est plus robuste sur petit n car il conditionne sur les positifs réels, un sous-ensemble plus stable.

---

## Tests statistiques — Chi², Fisher exact & bootstrap

Un écart visible sur un graphique peut venir du modèle (vrai biais) **ou du hasard d'échantillonnage** (n petit). On valide donc chaque disparité par un test formel.

### Chi² de Pearson et Fisher exact

Pour chaque attribut, **deux tests d'indépendance** :

| Test | Hypothèse nulle H₀ | Sur quel sous-ensemble |
|---|---|---|
| **DP** | La prédiction est indépendante du groupe | Test set entier (100) |
| **EO** | La prédiction est indépendante du groupe | Sous-ensemble `y_true = 1` (20) |

**Choix du test selon la forme de la table** :

| Forme | Test | Pourquoi |
|---|---|---|
| 2×2 (attribut binaire) | **Fisher exact** | Exact même avec effectifs <5. Standard sous α=0.05 quand n est petit. |
| 3×2 (attribut ternaire) | **Chi² Pearson** | Approximation asymptotique. Hypothèse : cellules attendues ≥ 5. Sinon on annote. |

> Pour les contingences 3×2 avec >20% de cellules attendues <5 (cas du test EO âge : 5/6 cellules concernées), le test idéal serait **Fisher-Freeman-Halton** (Fisher exact généralisé). Il n'est pas disponible dans scipy ; on garde Chi² avec annotation explicite. Le résultat (p=0.026) est indicatif mais à confirmer avec un dataset plus large.

### IC bootstrap 95% sur l'EOD

Le test set ne compte que **20 positifs**. La métrique EOD Âge calculée vaut 1.00, mais cette valeur repose sur **2 Juniors qualifiés** et **6 Seniors qualifiés**. Une métrique calculée sur 2 personnes a une marge d'erreur énorme.

Pour quantifier cette incertitude, on **bootstrappe** :

```
Pour i = 1 à 1000 :
    Ré-échantillonner le test set avec remise (n=100)
    Calculer l'EOD sur ce ré-échantillon
IC 95% = [percentile 2.5%, percentile 97.5%] des 1000 EOD obtenus
```

Interprétation :
- **IC étroit (largeur <0.2)** : la métrique est stable, on peut s'y fier.
- **IC large (largeur >0.4)** : variance d'échantillonnage forte — la conclusion repose sur trop peu d'observations. À traiter avec prudence.

C'est une honnêteté méthodologique importante : un EOD point à 1.00 avec un IC [0.6, 1.0] ne raconte pas la même histoire qu'un EOD à 1.00 avec un IC [0.95, 1.0]. Dans notre cas, les IC sont effectivement larges et confirment la nécessité d'un dataset plus grand pour un audit définitif.

### Calibration par groupe — la 3ᵉ jambe du fairness

La littérature fairness (Hardt et al. 2016 ; Pleiss et al. 2017) reconnaît **trois critères d'équité non simultanément satisfaisables** quand les base rates diffèrent :

1. **Demographic Parity** — taux de sélection égaux entre groupes (DP).
2. **Equal Opportunity** — TPR égaux à qualification égale (EOD).
3. **Calibration** — quand le modèle dit P=x dans deux groupes, la fraction réellement qualifiée est ≈ x dans les deux cas.

DP et EO ont été couverts par les métriques §5 et les tests §6.1–6.2. La calibration mesure une chose différente : **la fiabilité numérique des probabilités prédites entre groupes**.

**Pourquoi c'est important pour le recruteur** : si un Junior à P=0.5 est qualifié dans 20% des cas mais qu'un Senior à P=0.5 est qualifié dans 50% des cas, alors un même score de 0.5 ne *veut pas dire la même chose* selon le groupe. Le recruteur qui compare deux candidats à proba égale ferait un choix biaisé sans le savoir. C'est le type de biais que DP et EO ne détectent pas.

**Méthode appliquée** (volontairement simple) :

```
Pour chaque groupe g :
    mean_proba(g) = moyenne des probabilités prédites
    base_rate(g)  = moyenne des étiquettes réelles
    calibration_gap(g) = | mean_proba(g) − base_rate(g) |

Seuil d'alerte : gap > 0.10
```

Un gap > 0.10 signale une **sur- ou sous-confiance systématique** du modèle pour ce groupe.

**Lien légal** : AI Act Art. 15 — *« Les systèmes IA à haut risque sont conçus de manière à atteindre […] un niveau d'exactitude approprié »*. La calibration est une mesure directe de cette exactitude conditionnelle au groupe.

> Approche plus poussée (hors-scope ici) : calculer l'**Expected Calibration Error** (ECE) par groupe en découpant les probabilités en 10 bins. Notre méthode "moyenne globale par groupe" suffit pour détecter une décalibration grossière et reste lisible pour un RH.

---

## Analyse par attribut & intersectionnalité

### Âge — le cas le plus grave

| Groupe | n_test | qualifiés | Selection rate | **TPR** | Base rate test |
|---|---|---|---|---|---|
| Junior (≤29) | 36 | 2 | 11.1% | **0.0%** ❌ | 5.5% |
| Mid (30–34) | 38 | 12 | 36.8% | 66.7% | 31.6% |
| Senior (≥35) | 26 | 6 | 61.5% | **100%** | 23.1% |

- Corrélation Pearson `age` vs `proba` sur l'ensemble du dataset : **r = 0.543** — forte et positive.
- TPR Junior = 0%, TPR Senior = 100% → **EOD = 1.00** (maximum théorique).
- Chi² conditionnel à `y_true=1` : **p = 0.026** → significatif (Sig.).
- L'ancien modèle **ne détecte aucun candidat Junior pourtant qualifié**. Même avec n petit (2 Juniors qualifiés), c'est un signal alarmant.

### Distance géographique — pas de biais avéré

- Corrélation distance/proba : **r = 0.071** — très faible.
- Chi² DP : p = 0.53 → non significatif.
- Coefficient L1 *positif* (+0.04) : le modèle favorise même les candidats lointains.

→ La feature est **retirée par principe RGPD** (minimisation des données, risque de proxy d'origine), **pas pour corriger un biais avéré**. C'est de la prévention. La conférence (cf. [`ethique_synthese.md`](ethique_synthese.md) §1 — *Non-malfaisance*) a rappelé qu'une donnée qui *pourrait* révéler une origine ethnique ou nationale entre dans le champ du RGPD Art. 9 même si son coefficient est faible aujourd'hui — il peut croître avec plus de données.

### Francophonie — base rates quasi identiques, marginalement détectable

- Base rate Francophone 18.8% / Non-francophone 20.4% → quasi identiques.
- Chi² DP : p = 0.13 → non significatif.
- Mais l'analyse intersectionnelle (voir ci-dessous) montre que la cellule `Junior × Francophone` = 0% de sélection (n=8). À surveiller.

### Niveau d'éducation — biais avéré mais partiellement légitime

- Base rates très différents : Bachelor− 12.7% vs Master+ 30.1%.
- DP Gap = 0.56, DI = 0.17 — alarmant **en apparence**, mais reflète des qualifications réelles.
- EOD Gap = 0.55 → là c'est un vrai biais : à qualification égale, un Bachelor qualifié est moins détecté qu'un Master qualifié.

### Intersectionnalité — un point clé de la conférence

La conférence (cf. [`ethique_synthese.md`](ethique_synthese.md) §1 — *Focus sur l'Équité, les Biais et la Non-discrimination* : *« une vigilance particulière doit être portée sur l'intersectionnalité, car l'IA peut prendre des décisions basées sur une combinaison de caractéristiques protégées »*) a insisté : les discriminations sont souvent **croisées**. Un modèle peut être OK sur chaque attribut pris seul et discriminer sur leur combinaison.

On teste deux croisements pertinents :

- **`age_group × fr_speaker`** — révèle la cellule `Junior × Francophone` à 0% (n=8).
- **`age_group × geo_group`** — révèle des combinaisons à 0% selection.

> Les cellules avec effectif <5 sont annotées et traitées comme indicatives, pas conclusives. Un audit annuel avec un dataset plus large permettra de confirmer (ou infirmer) ces signaux intersectionnels.

---

## Disparités opérationnelles (rôle visé)

Cette section répond à la **question 3 du cahier des charges** : *« les disparités sont-elles justifiées par des caractéristiques liées au poste ? »*

`target_role` n'est pas un attribut protégé. Les écarts de selection rate entre rôles sont **attendus** : un poste de Data Scientist demande plus d'expérience qu'un poste junior, donc moins de candidats remplissent les critères. C'est de la **discrimination opérationnelle légitime**, pas de la discrimination juridique.

Sur notre test set, les rôles les plus sélectionnés sont Product Analyst, ML Engineer et DevOps Engineer (>45%) ; les moins sélectionnés sont QA, Cybersecurity, BI Developer (0% sur n petits). Ces écarts sont normaux : la composition des CV par rôle, le niveau d'expérience requis, et la rareté des compétences expliquent ces différences.

> ⚠ Cette analyse est **descriptive**, pas un audit légal. Aucune correction n'est appliquée — un recrutement *doit* avoir des critères différents selon le poste.

---

## Explicabilité du modèle

L'AI Act (Art. 13) et le RGPD (Art. 22) exigent que les décisions automatisées soient explicables. Le principe d'**Explicabilité** de la conférence (cf. [`ethique_synthese.md`](ethique_synthese.md) §1) ajoute une dimension humaine à cette obligation légale : éviter l'effet "boîte noire" où ni le candidat ni le recruteur ne comprennent pourquoi une décision a été prise. On combine **trois couches** complémentaires pour répondre à ces deux exigences.

### Couche 1 — Coefficients L1 (modèle d'origine)

La régression logistique L1 met automatiquement à zéro les features non informatives. **Sur 128 features, seulement 9 sont actives** dans le modèle d'origine :

| Feature | Coefficient | Lecture |
|---|---|---|
| `total_experience_years` | +0.7099 | Favorise les expérimentés (légitime) |
| `education_score` | +0.4238 | Favorise les écoles de prestige |
| `certif_count` | +0.2124 | Favorise les certifiés |
| `lang_de` | +0.1027 | ⚠ Favorise les germanophones |
| `lang_es` | +0.0854 | ⚠ Favorise les hispanophones |
| `lang_other_score_sum` | +0.0795 | ⚠ Favorise les multilingues "exotiques" |
| `gap_ratio` | −0.0707 | Pénalise les trous dans le CV |
| `distance_ville_haute_km` | +0.0396 | ⚠ Favorise les candidats lointains |
| `lang_it` | −0.0073 | ⚠ Pénalise les italophones |

Les coefficients marqués ⚠ sont des **proxies d'origine** — c'est sur cette base qu'ils ont été retirés du modèle FAIR (§10). *Pourquoi cette couche est utile :* les coefficients L1 donnent une image globale et parcimonieuse du modèle. Ils permettent de comprendre en un regard quelles features pilotent réellement les décisions, sans noyer le RH dans 128 variables.

### Couche 2 — SHAP (sur le modèle FAIR)

Les coefficients donnent l'impact *marginal* d'une feature. Les valeurs **SHAP** (SHapley Additive Explanations) donnent l'impact *réel* sur chaque prédiction, en tenant compte des interactions et de la distribution des données. On l'applique au modèle FAIR (celui qui sera déployé) sur 300 candidats.

*Pourquoi SHAP sur le modèle FAIR et non l'ancien :* le diagnostic des biais de l'ancien modèle a déjà été conduit via L1 (§8) et les corrélations âge/proba (§7). Ce qui importe pour le déploiement, c'est de comprendre le modèle FAIR — celui que LuxTalent va utiliser. SHAP valide que les features retirées n'ont pas été remplacées par des proxies inattendus.

**Résultats principaux (modèle FAIR)** :

| Feature | \|SHAP\| moyen | Direction |
|---|---|---|
| `education_score` | 0.71 | Favorise (+) |
| `lang_fr` | 0.24 | Pénalise (−) les non-francophones |
| `lang_en` | ~0.00 | Neutre |

Visualisations produites par le notebook :
- **Beeswarm** : importance globale + direction pour toutes les features.
- **Waterfall** : décomposition individuelle pour un candidat précis.

> ⚠ Le fait que `lang_fr` soit devenue la 2ᵉ feature la plus influente du modèle FAIR (|SHAP| = 0.24) est un signal à surveiller. En retirant les autres langues, le modèle a reporté du poids sur `lang_fr` — qui peut elle-même être un proxy d'origine francophone. Ce point est développé en §11.

### Couche 3 — Log-odds individuels (pour le recruteur)

Voir §12. C'est l'explication la plus opérationnelle : pour chaque décision, on liste les 5–10 features qui ont le plus pesé, avec leur contribution chiffrée. Conforme à l'**Art. 14 AI Act** (supervision humaine) et au **RGPD Art. 22** (droit à l'explication). *Pourquoi cette couche :* SHAP est puissant mais difficile à lire sans formation. Le log-odds individuel traduit la décision en langage naturel, accessible à un RH sans compétence en ML.

---

## Stratégie corrective retenue

### Stratégie 1 — Suppression des features sensibles (pre-processing) — RETENUE

**Principe :** si une feature pose problème, on la retire avant d'entraîner. Le nouveau modèle apprend sans avoir accès à ces signaux.

**Pourquoi cette approche plutôt qu'une autre :**
La conférence (cf. [`ethique_synthese.md`](ethique_synthese.md) §1 — principe de **Bienfaisance** et §2 — AI Act Art. 10) a souligné que le cadre européen privilégie la correction *en amont* des données plutôt qu'une manipulation *a posteriori* des prédictions. La suppression de features est la stratégie de pré-processing la plus défendable juridiquement car elle ne distingue jamais les candidats sur la base d'un attribut protégé au moment de la décision. C'est aussi la plus transparente : on peut expliquer à LuxTalent exactement quelles informations le modèle n'utilise plus et pourquoi.

**Features supprimées** (6) :
```
age · distance_ville_haute_km · lang_de · lang_es · lang_it · lang_other_score_sum
```

**Features conservées** :
- `lang_fr` : compétence professionnelle légitime dans le contexte belgo-luxembourgeois (LuxTalent opère dans un environnement bilingue FR/EN ; la maîtrise du français est un critère métier défendable).
- `lang_en` : lingua franca technique — neutre par construction (|SHAP| ≈ 0 dans le modèle FAIR).

**Pourquoi conserver `lang_fr` malgré le risque de proxy :**
C'est la décision la plus délicate de l'audit. La francophonie corrèle avec une origine luxembourgeoise, belge, française ou suisse. Supprimer `lang_fr` réduirait ce risque mais pénaliserait un critère métier réel pour un cabinet de recrutement au Luxembourg où de nombreux postes exigent une communication en français.

Le RGPD (Art. 5 — principe de **finalité**) autorise l'utilisation d'une donnée sensible si elle est proportionnée à la finalité poursuivie et si aucune alternative moins intrusive n'existe. Ici, `lang_fr` est proportionnée (un poste au Luxembourg peut légitimement exiger le français) et aucune donnée équivalente moins sensible n'est disponible dans le dataset. La décision est cependant **provisoire** : l'aggravation de l'EOD francophonie après correction (0.13 → 0.47, cf. §11) justifie de remettre ce choix en question lors du prochain audit.

**Méthodologie anti-data-leakage** — le point méthodologique le plus important de la refonte :

```
Dataset 500 candidats
├── Test    100 candidats  ← split original, intact, jamais touché
└── Train+val 400
    ├── Train 300 candidats  ← entraînement du modèle
    └── Val   100 candidats  ← tuning du seuil F-beta
```

Le seuil de décision est optimisé sur la validation par F-beta (β=0.5, privilégie la précision). Le test set n'est touché qu'une seule fois en toute fin pour produire les chiffres du §11.

### Justification du choix du seuil — une décision éthique, pas technique

Le choix de β=0.5 dans la F-beta n'est pas neutre. Le principe d'**Autonomie** de la conférence (cf. [`ethique_synthese.md`](ethique_synthese.md) §1) rappelle que l'humain doit garder le contrôle — y compris sur les paramètres qui définissent la politique de sélection. Ce choix doit donc être **validé avec LuxTalent**, pas décidé unilatéralement par l'équipe technique.

| β | Optimise | Conséquence pour le candidat | Conséquence pour LuxTalent |
|---|---|---|---|
| β = 0.5 (notre choix) | Précision > rappel | Moins de faux positifs : un candidat "sélectionné" l'est probablement vraiment | Moins d'entretiens à organiser, mais on rate plus de vrais talents |
| β = 1.0 (équilibré) | F1 équilibré | Standard | Compromis |
| β = 2.0 | Rappel > précision | Plus de chances d'être invité même si la conviction du modèle est faible | Plus d'entretiens, mais moins de vrais talents ratés |

Nous avons retenu **β=0.5** pour deux raisons :

1. **Précaution éthique côté faux positif** : un candidat à qui on dit "sélectionné" puis qu'on rejette à l'entretien est une mauvaise expérience humaine et juridique. Mieux vaut être prudent dans le « oui ».
2. **Coût opérationnel** : un faux positif coûte un entretien (RH mobilisé, candidat dérangé) ; un faux négatif coûte un talent raté. Pour LuxTalent en haut volume, le coût d'un entretien est plus tangible — donc privilégier précision.

> Ce choix est **contestable** et **doit l'être** : un client qui privilégierait l'inclusion (rappel maximum) demanderait β=2. La fonction `precision_recall_curve` étant déjà calculée, un changement de β ne demande pas de ré-entraînement, seulement un recalibrage du seuil. C'est une décision *à valider avec le donneur d'ordre*, pas un choix technique anodin.

Seuils obtenus :
- **Ancien modèle** : 0.1434 (très bas — privilégie le rappel) ← héritage du notebook de classification
- **Nouveau modèle FAIR** : 0.6352 (modéré — privilégie la précision)

L'écart vient en partie du fait que le nouveau modèle, ayant moins de features, produit des probabilités plus polarisées (moins de candidats en zone grise).

**Résultats du modèle FAIR sur le test set** :
```
Seuil optimal (calibré sur val) : 0.6352
ROC-AUC                         : 0.687
Accuracy                        : 75%
Précision (sélectionné)         : 0.41
Rappel    (sélectionné)         : 0.55
Rappel    (refusé)              : 0.80
```

### Stratégie 2 — Calibration de seuils par groupe (post-processing) — ÉTUDIÉE PUIS REJETÉE

L'idée : appliquer un seuil différent selon le groupe (par ex. 0.05 pour les Juniors, 0.70 pour les Seniors) de façon à égaliser les TPR. Mathématiquement, ça force l'**Equal Opportunity**.

**Pourquoi on l'abandonne** — argument juridique :
- Appliquer un standard de réussite différent selon une caractéristique protégée = **demographic norming** = **discrimination directe** sous :
  - Loi belge du 10 mai 2007 contre la discrimination (Art. 5)
  - Directive européenne 2000/78/CE (Art. 2)
- L'AI Act (Art. 10) et le RGPD (Art. 5) privilégient la suppression de la variable sensible **en amont**, pas son utilisation pour différencier les seuils à l'inférence.

**Pourquoi on l'abandonne** — argument technique :
- Pour égaliser le TPR Junior, le seuil devait tomber à 0.05, ce qui faisait exploser le taux de sélection Junior à **93%** (vs 20% réels). Inacceptable opérationnellement.

→ Seule la **Stratégie 1** est retenue et déployée. La Stratégie 2 est documentée à titre pédagogique.

---

## Comparaison ancien vs nouveau modèle

Comparaison sur le **même test set** (rigueur méthodologique).

### Performances brutes

| Métrique | Ancien | Nouveau (FAIR) |
|---|---|---|
| ROC-AUC | **0.706** | 0.687 |
| Accuracy | 74% | **75%** |
| Rappel (sélectionné) | **0.70** | 0.55 |
| Précision (sélectionné) | 0.41 | 0.41 |
| Rappel (refusé) | 0.75 | **0.80** |

Perte de **~2 points de ROC-AUC** et 15 points de rappel sur la classe positive. C'est le **trade-off équité/performance assumé**. La conférence (cf. [`ethique_synthese.md`](ethique_synthese.md) §1 — principe de **Bienfaisance**) pose explicitement cette question : un outil qui "fait le bien" pour une entreprise (performances prédictives maximales) ne fait pas nécessairement le bien pour les candidats si ses performances sont inégalement distribuées entre groupes.

### Équité — vue d'ensemble

| Attribut | EOD ancien | EOD nouveau | Δ | DI min ancien | DI min nouveau |
|---|---|---|---|---|---|
| Âge | 1.00 | 0.67 | **−0.33** ✅ | 0.18 | 0.11 |
| Francophonie | 0.13 | 0.47 | +0.34 ⚠ | 0.47 | 0.28 |
| Niveau éducation | 0.55 | 0.62 | +0.07 ⚠ | 0.17 | 0.12 |
| Distance géographique | 0.23 | 0.20 | −0.03 | 0.72 | 0.58 |

### Effets de bord — analyse honnête

**1. EOD francophonie qui s'aggrave (0.13 → 0.47) — le principal risque résiduel.**

C'est l'effet de bord le plus préoccupant de la correction. Quand on retire `lang_de`, `lang_es`, `lang_it` et `lang_other_score_sum`, le modèle perd des signaux prédictifs et reporte mécaniquement du poids sur les features restantes. Le SHAP du modèle FAIR confirme : `lang_fr` est devenue la 2ᵉ feature la plus influente (|SHAP| ≈ 0.24).

Ce que cela signifie concrètement : un candidat non francophone qualifié est maintenant *moins bien détecté* que dans l'ancien modèle, alors que l'ancien modèle ne l'était déjà qu'à 87% de la fréquence d'un candidat francophone. Autrement dit, **on a corrigé le biais sur l'âge au prix d'une aggravation du biais sur l'origine francophone**.

Ce résultat n'invalide pas la correction — l'EOD âge était à 1.00 (maximum absolu, inacceptable légalement) tandis que l'EOD francophonie passe de 0.13 à 0.47 (préoccupant, mais dans une plage où le Chi² n'est pas significatif sur ce dataset). Cependant, ce point doit être **communiqué sans ambiguïté à LuxTalent** : la version 2 du modèle n'est pas une version entièrement équitable — c'est une version moins discriminatoire sur le critère le plus urgent (l'âge), avec un risque accru sur un autre critère à surveiller.

*Quelle serait la prochaine étape :* tester une version où `lang_fr` est également retirée, mesurer l'impact sur les performances et l'EOD francophonie, et soumettre le choix à LuxTalent avec le tableau des trade-offs. Si la maîtrise du français est vérifiée en entretien de toute façon, retirer `lang_fr` du scoring automatique peut être pertinent.

**2. DI min qui se dégrade sur tous les attributs.**

```
DI min Âge      : 0.181 → 0.111  (−38%)
DI min Distance : 0.715 → 0.578  (−19%)
DI min Francopho: 0.473 → 0.284  (−40%)
DI min Éducation: 0.173 → 0.116  (−33%)
```

Cette dégradation générale semble alarmante mais s'explique par la nature mathématique du DI : c'est un rapport `min_rate / max_rate`. Avec de petits effectifs (2 Juniors qualifiés au test), un léger changement du taux de sélection du groupe le plus faible fait varier le DI disproportionnellement. Le modèle FAIR ayant un seuil plus élevé (0.635 vs 0.143), il sélectionne moins au total, ce qui comprime les taux des petits groupes.

C'est précisément pour cela que l'**EOD prime sur le DI** dans cet audit : l'EOD conditionne sur les vrais positifs et est plus robuste aux effets de seuil. Le DI brut, sans correction des base rates, mesure partiellement les différences de qualification réelle plutôt que le biais du modèle. Il reste utile comme indicateur légal (règle des 80%) mais ne doit pas être interprété isolément.

**3. EOD éducation qui augmente légèrement (+0.07).**

Dans la marge de variance d'échantillonnage (l'IC bootstrap est large à n=20 positifs). Le biais éducation n'est pas adressé par la stratégie 1 car `education_score` et `education_degree` sont conservés comme features légitimes (compétences réelles, pas proxies). C'est un compromis assumé : traiter ce biais demanderait soit de retirer le niveau d'éducation du modèle (ce qui dégraderait fortement les performances), soit de s'attaquer au déséquilibre des qualifications réelles dans le dataset (hors scope de l'audit).

> Dans un contexte AI Act haut risque, **l'EOD sur l'attribut le plus discriminé (l'âge) prime** sur les autres métriques. Le report partiel du poids sur `lang_fr` justifierait un second tour d'audit dès qu'un dataset plus large sera disponible.

---

## Explication individuelle d'une décision

Pour répondre à l'**Art. 14 de l'AI Act** (supervision humaine), le notebook fournit une fonction `explain_prediction_logistic` qui décompose chaque prédiction en contributions de chaque feature au log-odds.

**Exemple — candidat `cv_0001.txt`** :
```
Rôle visé           : Software Engineer
Expérience          : 2.4 ans
Score éducation     : 3

Probabilité prédite : 9.7%
Seuil de décision   : 63.5%
Décision            : ❌ REFUSÉ

Top contributions :
  total_experience_years  −0.97  (manque d'expérience)
  education_score         −0.61  (école standard)
  certif_count            −0.21  (peu de certifications)
  lang_fr                 +0.18  (francophone — léger boost)
```

Un graphique en barres horizontales accompagne chaque décision :
- 🟢 **Vert** → la feature pousse vers la sélection
- 🔴 **Rouge** → la feature pousse vers le refus

Le recruteur peut ainsi :
1. Comprendre la décision (Art. 14 AI Act).
2. Contester si une feature clé est manifestement erronée (RGPD Art. 22).
3. Faire un *override* humain si le contexte le justifie.

---

## Conclusion

### Résultats clés en une ligne

| Indicateur | Ancien | Nouveau | Δ |
|---|---|---|---|
| **EOD Âge** (métrique clé) | 1.00 | 0.67 | **−0.33** ✅ |
| EOD Francophonie | 0.13 | 0.47 | **+0.34** ⚠ |
| ROC-AUC | 0.706 | 0.687 | −0.019 |
| Recall sélectionné | 0.70 | 0.55 | −0.15 |

### Ce que l'audit a permis de faire

1. **Identifier** les biais du modèle d'origine — corrélation âge/proba r=0.543, TPR Junior = 0%, plusieurs langues agissant comme proxies d'origine.
2. **Quantifier** ces biais avec DP/DI/EOD + Chi²/Fisher + IC bootstrap, le tout out-of-sample.
3. **Distinguer** discrimination opérationnelle légitime (rôle visé) de discrimination protégée (âge).
4. **Corriger** le modèle en supprimant 6 features problématiques, avec un coût performance documenté.
5. **Expliquer** chaque décision (L1 global + SHAP + log-odds individuel).
6. **Documenter** le tout conformément aux Art. 9, 10, 13, 14 de l'AI Act.

### Réponses au cahier des charges WP2

1. *Le système traite-t-il les candidats comparables de manière égale ?* — **Non, pour l'âge** sur l'ancien modèle (EOD = 1.00). Corrigé partiellement sur le nouveau (0.67). La francophonie, elle, s'est aggravée.
2. *Y a-t-il des disparités mesurables ?* — Oui sur âge et éducation (Chi² significatif). Non sur distance ni francophonie marginale (avant correction).
3. *Sont-elles justifiées par le poste ?* — Partiellement pour l'éducation (Master+ plus qualifié) et les rôles techniques (cf. §8). L'écart de TPR par âge n'est *pas* justifié — c'est un vrai biais.
4. *Le modèle peut-il être amélioré ?* — Oui : EOD Âge réduit de 33 points avec le modèle FAIR, au prix de 2 points de ROC-AUC. Un second tour d'audit sur `lang_fr` est recommandé.
5. *Les décisions peuvent-elles être rendues plus transparentes ?* — Oui, via SHAP global + log-odds individuel (§9, §12), conforme aux Art. 13 et 14 AI Act.

### Limites assumées

- **Test set de 100 candidats (20 positifs)** → IC bootstrap larges, certaines cellules intersectionnelles à effectif <10.
- **Attributs sensibles "vrais"** (genre, nationalité, origine ethnique) **absents du dataset** — l'audit travaille sur des proxies.
- **Chi² 3×2 avec cellules <5** : l'idéal serait Fisher-Freeman-Halton, non disponible dans scipy. Résultat (p=0.026) reste indicatif.
- **L'amélioration EOD Âge pourrait être en partie due à la variance d'échantillonnage** — à confirmer avec un dataset plus large.
- **Le modèle FAIR n'est pas un modèle entièrement équitable** — il réduit le biais sur l'âge mais aggrave l'EOD francophonie. C'est un premier cycle de correction, pas un état final.

>Je le répète : nous avons dû faire un compromis entre l’âge et la maîtrise du français. Dans le cadre de cet audit, il nous semble cohérent qu’une entreprise luxembourgeoise considère la connaissance de la langue administrative du pays comme un critère pertinent.

### Recommandations pour LuxTalent

1. **Audit annuel** avec un test set d'au moins 500 candidats pour réduire les IC.
2. **Documenter les décisions contestées** (RGPD Art. 22 — droit à l'explication).
3. **Boucle de feedback humain** pour les cas limites (proba ∈ [seuil ± 5%]).
4. **Collecter des données démographiques anonymisées** pour mesurer les vrais attributs protégés (avec consentement explicite).
5. **Former les recruteurs** aux biais algorithmiques et à la lecture des explications individuelles.
6. **Valider le paramètre β** avec la direction : le choix de β=0.5 est défendable mais contestable — un client orienté inclusion demanderait β=2.

### Apports de la conférence intégrés à cet audit

La synthèse de la conférence est disponible dans [`ethique_synthese.md`](ethique_synthese.md). Les cinq principes *AI4People* définis en §1 ont guidé les choix techniques suivants :

| Principe (cf. [`ethique_synthese.md`](ethique_synthese.md) §1) | Choix technique dans l'audit | Pourquoi ce lien |
|---|---|---|
| **Justice & Équité** | EOD plutôt que DP seule comme métrique principale | La DP brute ignorerait les différences réelles de qualification entre groupes — "équité" ne signifie pas "taux égaux" mais "chances égales à mérite égal" |
| **Intersectionnalité** (focus biais) | Analyses croisées Âge × Francophonie et Âge × Géographie | Un modèle peut discriminer via la combinaison de deux attributs même s'il paraît équitable sur chacun isolément |
| **Explicabilité** (anti black-box) | Triple couche L1 + SHAP + log-odds individuel | Sans explicabilité, le recruteur ne peut ni comprendre ni contester une décision — Art. 13 et 14 AI Act |
| **Non-malfaisance** (*capability caution*) | IC bootstrap sur l'EOD, évaluation out-of-sample uniquement | Ne pas présenter des chiffres qui semblent certains quand le dataset est trop petit pour l'être |
| **Bienfaisance** | Suppression des features sensibles en pré-processing | La correction doit agir sur les données d'entraînement, pas contourner le problème par des ajustements a posteriori |
| **Durabilité / IA frugale** (§3 de la synthèse) | Régression logistique L1 préférée à un modèle deep | ~64 KB sauvegardé, pas de GPU requis, entraînement local en secondes — intrinsèquement explicable et à empreinte minimale |
| **Hauts risques AI Act** (§2 de la synthèse) | Ensemble du document | Notre outil entre dans l'Annexe III — toutes les obligations documentaires, de gouvernance et de supervision humaine s'appliquent |

**Un modèle performant n'est pas forcément un modèle équitable.** Nous avons choisi de sacrifier un peu de ROC-AUC pour avoir un système qui ne discrimine plus l'âge — choix légitime dans un contexte de recrutement automatisé classé haut risque par l'AI Act. Ce choix reste **incomplet** : la francophonie mérite un second cycle de correction, et le paramètre β doit être co-décidé avec LuxTalent.

---

## Glossaire

| Terme | Définition |
|---|---|
| **Base rate** | Taux *réel* de candidats qualifiés (`Y=1`) dans un groupe. À comparer au selection rate. |
| **Bootstrap** | Méthode de ré-échantillonnage (tirage avec remise) pour estimer la marge d'erreur d'une métrique. |
| **Demographic norming** | Pratique consistant à appliquer un standard d'évaluation différent selon un attribut protégé. Illégale en droit du travail européen. |
| **Demographic Parity (DP)** | Métrique d'équité : exige des selection rates égaux entre groupes. Sensible aux différences de base rate. |
| **Disparate Impact (DI)** | Ratio min/max des selection rates. Règle des 80% : DI ≥ 0.80 attendu. Sensible aux petits effectifs. |
| **EOD** | Equal Opportunity Difference : écart max-min des TPR entre groupes. Métrique principale ici. |
| **Equal Opportunity** | À qualification égale (`Y=1`), tous les groupes doivent avoir la même probabilité d'être détectés. |
| **Fisher exact** | Test d'indépendance exact pour tables 2×2, valide même avec petits effectifs. |
| **L1 (régularisation Lasso)** | Pénalisation qui force certains coefficients à zéro → modèle parcimonieux et explicable. |
| **Out-of-sample** | Évaluation sur des données jamais vues à l'entraînement. Garantit que les métriques sont fiables. |
| **Proxy** | Feature qui révèle indirectement un attribut protégé (ex. langue parlée → nationalité). |
| **SHAP** | SHapley Additive Explanations — méthode d'explicabilité fondée sur la théorie des jeux. |
| **TPR (Taux de Vrais Positifs / Recall)** | Parmi les `Y=1`, fraction correctement détectée par le modèle. Composante principale de l'EOD. |

---

> *Notebook de référence : [`backend/notebooks/fairness_audit.ipynb`](../notebooks/fairness_audit.ipynb)*
> *Document éthique de référence (synthèse de la conférence) : [`ethique_synthese.md`](ethique_synthese.md)*