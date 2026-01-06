# EL Amraoui Abir ENCG SETTAT Finance G2                    
# Apogé: 24010353
![Abir](https://github.com/elamraouiabir6963encg-rgb/24010353_ELAMRAOUI-ABIR/blob/main/Image/Abir.png?raw=true)

# 🏦 Analyse Exploratoire et Modélisation Prédictive  
## Dataset Bank Marketing (UCI)

---

## 1. Problématique, métier et objectifs de la mission

### Problématique du métier
Dans le secteur bancaire, les campagnes de télé-marketing constituent un levier stratégique pour promouvoir les dépôts à terme, un produit d’épargne essentiel pour assurer la liquidité et la stabilité financière des banques.

Cependant, ces campagnes présentent plusieurs limites opérationnelles et financières :
*   Coûts élevés liés aux centres d’appels.
*   Taux de conversion faible, car la majorité des clients contactés ne souscrivent pas au produit.
*   Ciblage inefficace, entraînant une surcharge des agents et une allocation non optimale du budget marketing. 
*   Perte d’opportunités commerciales,  lorsque des clients potentiellement intéressés ne sont pas identifiés à temps.

Ainsi, la banque se trouve confrontée à un défi majeur :
*   **Comment identifier, avant même l’appel, les clients ayant la plus forte probabilité de souscrire à un dépôt à terme, afin d’optimiser les ressources et maximiser le rendement des campagnes ?** 

### L’Enjeu décisionnel : Une matrice de coûts d’erreur asymétrique
Contrairement à des problématiques classiques de classification, ici toutes les erreurs n’ont pas le même impact :
*   **Faux Positif:** un client est prédit comme intéressé alors qu’il ne souscrira pas                                                    → Coût opérationnel inutile, perte de temps agent, saturation des lignes d’appels.
*   **Faux Négatif:** un client réellement intéressé est classé comme non pertinent
→ Manque à gagner commercial direct, perte potentielle de valeur client à long terme.

Dans ce contexte, la priorité stratégique n’est pas seulement la précision, mais surtout :
*   **Maximiser le rappel de la classe positive, afin de réduire au maximum les faux négatifs et ne pas laisser passer les clients intéressés.** 

###  Objectifs de la mission IA
La mission consiste donc à concevoir un Assistant IA d’aide au ciblage client, basé sur le machine learning, permettant de :
*   Analyser le comportement et le profil client.
*   Prédire la probabilité de souscription avant le contact téléphonique.
*   Optimiser le ciblage des campagnes marketing. 
*   Réduire les coûts opérationnels liés aux appels non pertinents.
*   Augmenter le taux de conversion et le rendement global des campagnes.
*   Fournir un outil scalable pouvant être intégré dans un système CRM bancaire.

### Les Données utilisées (L’Input du Modèle)
Pour répondre à cette mission, nous exploitons le dataset *Bank Marketing (UCI)*, qui contient des données historiques issues de campagnes réelles :
*   Profil socio-démographique : âge, profession, niveau d’éducation, situation familiale…
*   Indicateurs financiers : solde bancaire (balance)
*   Historique marketing : nombre de contacts lors de la campagne (campaign), résultat des campagnes précédentes (poutcome), nombre de contacts antérieurs (previous), délai depuis le dernier contact (pdays)…
*   Variable cible (y) : souscription au dépôt à terme → yes ou no
*   Taille et format : 45 211 observations, 17 variables, données tabulaires, mélangeant variables numériques et catégorielles
Ces données permettent au modèle d’apprendre à partir de tendances comportementales réelles, tout en tenant compte des contraintes économiques du métier.

![Aperçu des données](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Aper%C3%A7u%20des%20donn%C3%A9es.png?raw=true)
---

## 2. Le Code Python simplifié

```python
# INSTALLATION & IMPORTS
# pip install ucimlrepo seaborn matplotlib pandas numpy

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1) CHARGEMENT DES DONNÉES
# ============================================================
bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets

df = pd.concat([X, y], axis=1)


# ============================================================
# 2) ANALYSE DES VARIABLES NUMERIQUES
# ============================================================
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# HISTOGRAMMES
for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution de {col}")
    plt.show()

# BOXPLOTS
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

# HEATMAP
plt.figure(figsize=(10,6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()

# ============================================================
# 3) FEATURE ENGINEERING
# ============================================================
df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100],
                          labels=['Jeune', 'Adulte', 'Senior', 'Très senior'])
df['duration_min'] = df['duration'] / 60
df['total_contacts'] = df['campaign'] + df['previous']


# ============================================================
# 4) MODELISATION : LOGISTIC REGRESSION + RANDOM FOREST
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df['y'] = df['y'].map({'yes':1, 'no':0})

X = df.drop(columns=['y'])
y = df['y']

num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object','category']).columns

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ============================================================
# MODELE 1 : RÉGRESSION LOGISTIQUE
# ============================================================
log_reg_model = Pipeline(steps=[
    ('prep', preprocess),
    ('model', LogisticRegression(max_iter=1000))
])

log_reg_model.fit(X_train, y_train)
y_pred_log = log_reg_model.predict(X_test)



# ============================================================
# MODELE 2 : RANDOM FOREST
# ============================================================
rf_model = Pipeline(steps=[
    ('prep', preprocess),
    ('model', RandomForestClassifier(n_estimators=300, random_state=42))
])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

```
---
### **Interprétation du script Python**

Le code présenté ci-dessus illustre l’intégralité du pipeline de traitement et de modélisation appliqué au dataset **Bank Marketing (UCI)**. Il suit une approche méthodologique rigoureuse, structurée autour de quatre axes principaux :

1. **Chargement et consolidation des données**, permettant d’unifier les variables explicatives (`X`) et la variable cible (`y`) dans un même DataFrame pour faciliter l’analyse et le prétraitement.
2. **Analyse statistique des variables numériques (EDA)**, réalisée à l’aide d’histogrammes, de boxplots et d’une matrice de corrélation. Cette étape permet d’examiner :

   * la forme des distributions,
   * la dispersion des données,
   * ainsi que la présence de valeurs extrêmes pouvant influencer les performances des modèles prédictifs.
3. **Feature Engineering**, visant à enrichir le jeu de données par la création de nouvelles variables plus interprétables et pertinentes pour le métier bancaire, notamment :

   * `age_group` pour la segmentation client,
   * `duration_min` pour convertir la durée d’appel en minutes,
   * et `total_contacts` pour mesurer l’intensité du contact marketing.
4. **Modélisation supervisée**, basée sur deux algorithmes complémentaires :

   * **La Régression Logistique**, utilisée comme modèle linéaire de référence (baseline), adaptée à une première estimation de la probabilité de souscription.
   * **Le Random Forest**, modèle non-linéaire puissant, capable de capturer des interactions complexes entre les variables et de réduire le risque de surapprentissage grâce à l’agrégation d’arbres multiples.

L’encodage des variables catégorielles via `OneHotEncoder` et l’utilisation d’un `ColumnTransformer` garantissent un prétraitement optimal avant l’entraînement des modèles. Enfin, la division des données en ensembles d’apprentissage et de test (`train_test_split`) respecte la distribution de la classe cible (`stratify=y`), assurant une évaluation fiable des performances.

Ce pipeline technique constitue une base solide pour la prédiction client, tout en tenant compte de l’enjeu prioritaire du projet : **Minimiser les faux négatifs en maximisant le rappel de la classe positive, dans un contexte où le coût des erreurs est fortement asymétrique.**

---

# 5. RÉSULTATS DE L'ANALYSE EXPLORATOIRE (EDA)

## 5.1 Statistiques descriptives
![Statistiques descreptives](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Statistiques%20descreptives.png?raw=true)

📌La différence observée entre la moyenne et la médiane pour plusieurs variables indique des distributions asymétriques, souvent dues à la présence de valeurs extrêmes. 

---

# 5.2 Histogrammes – Distributions

### Age
![Distrtibution de Age](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20Age.png?raw=true)

📌 Interprétation - age :
Une concentration importante des clients dans les classes d’âge intermédiaires entre 30 et 60 ans. Cette distribution asymétrique suggère que la clientèle bancaire est majoritairement composée d’adultes actifs, ce qui est cohérent avec la cible des produits d’épargne à terme. La présence de clients très jeunes ou très âgés reste marginale, indiquant des segments spécifiques pouvant nécessiter des stratégies marketing différenciées.

### Balance
![Distribbution de Balance](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20Balance.png?raw=true)

📌 Interprétation - balance :
La distribution du solde bancaire est fortement asymétrique à droite, avec une majorité de clients présentant des soldes faibles ou proches de zéro, et quelques valeurs très élevées. Cette forte dispersion met en évidence la présence d’outliers, typique des données financières.

### Day of week
![Distribbution de pdays_of_week](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20day_of_week.png?raw=true)

📌 Interprétation - day_of_week :
L’histogramme du jour de la semaine montre une répartition relativement homogène des appels marketing sur les différents jours ouvrables. Cela suggère une organisation équilibrée des campagnes de contact, sans concentration excessive sur un jour spécifique, ce qui limite les biais temporels dans les données.

### Duration
![Distribbution de duration](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20duration.png?raw=true)

📌 Interprétation - duration :
La durée des appels présente une distribution très asymétrique, avec une majorité d’appels courts et un faible nombre d’appels très longs. Cette variable est particulièrement discriminante, car les appels plus longs sont souvent associés à un intérêt plus marqué du client. Toutefois, son utilisation doit être prudente, car elle peut introduire une fuite de données si elle est observée après la décision de souscription.

### Campaign
![Distribbution de campaign](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20campaign.png?raw=true)

📌 Interprétation - campaign :
La majorité des clients est contactée un nombre limité de fois durant une campagne, tandis qu’un petit nombre de clients fait l’objet de sollicitations répétées. Cette asymétrie reflète une stratégie de relance ciblée, mais peut également indiquer un risque de saturation ou de rejet de la part des clients fortement sollicités.

### Pdays
![Distribbution de pdays](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20pdays.png?raw=true)

📌 Interprétation - pdays :
La distribution de pdays met en évidence une forte concentration sur certaines valeurs spécifiques, traduisant des périodes standardisées de relance commerciale. Les valeurs élevées indiquent des clients peu ou jamais recontactés, ce qui peut influencer leur probabilité de souscription.

### Previous
![Distribbution de previous](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20previous.png?raw=true)

📌 Interprétation - previous :
La variable previous montre que la majorité des clients n’a eu que peu ou pas de contacts lors de campagnes précédentes. Cela suggère que la base clients est régulièrement renouvelée et que les campagnes antérieures ont un impact limité pour une grande partie de la population.

---

# 5.3 BOXPLOTS 

### Age
![Boxplot de age](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20age.png?raw=true)

📌 Interprétation - age :
Le boxplot de l’âge révèle une dispersion modérée autour de la médiane, avec quelques valeurs aberrantes correspondant à des clients très âgés. Ces outliers restent cohérents dans un contexte bancaire et ne nécessitent pas forcément d’exclusion, mais ils doivent être pris en compte dans l’interprétation des résultats.

### Balance
![Boxplot de Balance](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20Balance.png?raw=true)

📌 Interprétation - balance :
Le boxplot du solde bancaire met en évidence de nombreux outliers positifs, confirmant une forte hétérogénéité financière entre les clients. Cette variabilité est typique des données bancaires et justifie l’utilisation de modèles robustes aux valeurs extrêmes, comme les méthodes ensemblistes.

### Day of week
![Boxplot de day_of_week](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20day%20of%20week.png?raw=true)

📌 Interprétation - day_of_week :
Le boxplot du jour de la semaine montre une faible dispersion autour de la médiane et peu de valeurs extrêmes. Cela indique une répartition relativement homogène des contacts sur les différents jours ouvrables, sans concentration excessive susceptible d’introduire un biais temporel dans les campagnes marketing.

### Duration
![Boxplot de duration](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20duration.png?raw=true)

📌 Interprétation - duration :
La présence de nombreux points extrêmes dans la durée des appels confirme une forte asymétrie de la variable. Ces valeurs élevées correspondent à des interactions longues, potentiellement associées à une forte probabilité de souscription, mais elles peuvent aussi biaiser certains algorithmes si elles ne sont pas correctement traitées.

### Campaign
![Boxplot de campaign](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20campaign.png?raw=true)

📌 Interprétation - campaign :
Le boxplot de la variable campaign met en évidence une distribution asymétrique avec plusieurs valeurs aberrantes. La majorité des clients est contactée un nombre limité de fois, tandis qu’un petit groupe fait l’objet de relances répétées, traduisant une stratégie commerciale ciblée mais potentiellement intrusive.

### Pdays
![Boxplot de pdays](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20pdays.png?raw=true)

📌 Interprétation - pdays :
Le boxplot de pdays révèle une distribution très asymétrique, marquée par des valeurs extrêmes élevées. Ces observations correspondent à des clients n’ayant pas été recontactés depuis longtemps, voire jamais, ce qui peut influencer significativement leur probabilité de souscription.

### Previous
![Boxplot de previous](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20previous.png?raw=true)

📌 Interprétation - previous :
La variable previous présente une dispersion faible pour la majorité des observations, avec quelques valeurs extrêmes. Cela suggère que la plupart des clients ont eu peu de contacts lors des campagnes précédentes, alors qu’une minorité a été sollicitée de manière plus intensive.

# 5.4 HEATMAP – Corrélations
![Matrice de corrélation](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Matrice%20de%20corr%C3%A9lation.png?raw=true)

📌 Interprétation globale des corrélations :
La matrice de corrélation montre globalement de faibles corrélations entre les variables numériques, indiquant une faible redondance de l’information. La variable duration se distingue par une corrélation plus élevée avec la variable cible, ce qui confirme son pouvoir explicatif. L’absence de multi-colinéarité forte est favorable à l’utilisation de modèles comme la régression logistique.

--- FEATURE ENGINEERING ---

✅ Nouvelles variables créées :

![feature engineering](https://github.com/elamraouiabir6963encg-rgb/24010353_ELAMRAOUI-ABIR/blob/main/Image/feature%20engineering.png?raw=true)

Le feature engineering vise à enrichir le jeu de données initial en créant de nouvelles variables plus interprétables et plus pertinentes pour la modélisation prédictive. Dans cette analyse, trois nouvelles variables ont été construites:

La variable **age_group** permet de segmenter les clients en classes d’âge homogènes, facilitant l’interprétation métier et l’identification des segments les plus susceptibles de souscrire à un dépôt à terme. Cette transformation réduit également la complexité liée à la variabilité individuelle de l’âge.

La variable **duration_min**, obtenue par conversion de la durée des appels en minutes, améliore la lisibilité et l’interprétation de l’intensité des interactions commerciales, tout en conservant l’information essentielle contenue dans la variable initiale.

Enfin, la variable **total_contacts**, qui agrège le nombre de contacts de la campagne en cours et des campagnes précédentes, fournit une mesure globale de la pression commerciale exercée sur le client. Cette information est particulièrement pertinente pour analyser l’impact de la fréquence des sollicitations sur la probabilité de souscription.

### Répartition par âge
![Répartition par groupe d'âge](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/R%C3%A9partition%20par%20groupe%20d'%C3%A2ge.png?raw=true)

📌 Interprétation : 
La segmentation en groupes d’âge montre une dominance claire des adultes, suivis des seniors. Cette représentation facilite l’interprétation métier et permet d’adapter les stratégies commerciales selon les segments démographiques les plus rentables.

### Durée (minutes)
![Distribution de la durée de contact](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20la%20dur%C3%A9e%20de%20contact.png?raw=true)

📌 Interprétation : 
La transformation de la durée en minutes améliore la lisibilité du phénomène observé. La dominance des appels courts confirme que l’efficacité commerciale repose sur un faible nombre d’interactions longues et qualitatives.

### Total contacts
![Distribution du nobre total de contacts](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20du%20nombre%20total%20de%20contacts.png?raw=true)

📌 Interprétation : 
La majorité des clients est contactée peu de fois, ce qui traduit une stratégie de ciblage relativement sélective. Les clients très sollicités représentent une minorité, ce qui limite les coûts opérationnels tout en concentrant les efforts sur les profils jugés prometteurs.

# 6. Résultats de la régression logisqtique et random forest

# 6.1 Régression logistique
![Réression logisitique](https://github.com/elamraouiabir6963encg-rgb/24010353_ELAMRAOUI-ABIR/blob/main/Image/R%C3%A9gression%20logistique.png?raw=true)

📌 Interprétation :  
La régression logistique est utilisée comme modèle de référence en raison de sa simplicité et de sa forte interprétabilité. Elle permet d’estimer la probabilité de souscription à un dépôt à terme en fonction des caractéristiques du client, en supposant une relation linéaire entre les variables explicatives et le logarithme des chances de souscription.

Les résultats montrent que ce modèle fournit des performances correctes, mais reste limité dans sa capacité à capturer des relations complexes et non linéaires entre les variables. En particulier, la détection des clients réellement susceptibles de souscrire peut être insuffisante, ce qui est problématique dans un contexte bancaire où le coût d’un faux négatif est élevé. Ainsi, la régression logistique constitue une base de comparaison pertinente, mais non optimale pour ce problème.


# 6.2 Random forest
![Random forest](https://github.com/elamraouiabir6963encg-rgb/24010353_ELAMRAOUI-ABIR/blob/main/Image/Random%20forest.png?raw=true)

📌 Interprétation :
Le Random Forest est un modèle ensembliste basé sur l’agrégation de plusieurs arbres de décision, ce qui lui permet de capturer efficacement les interactions complexes et les relations non linéaires présentes dans les données. Il est également plus robuste face aux valeurs extrêmes et à l’hétérogénéité des profils clients.

Les performances obtenues avec le Random Forest sont globalement supérieures à celles de la régression logistique, notamment en termes de rappel et de précision sur la classe des clients souscripteurs. Cette amélioration est particulièrement importante dans le cadre du ciblage marketing, car elle permet de réduire les faux négatifs et d’optimiser l’efficacité des campagnes. Le Random Forest apparaît ainsi comme le modèle le plus adapté pour ce cas d’usage.

# 6.3 Analyse comparative
En comparaison, la régression logistique offre une bonne lisibilité des résultats mais montre des limites face à la complexité des données, tandis que le Random Forest privilégie la performance prédictive au détriment de l’interprétabilité. Le choix final du modèle dépend donc d’un compromis entre compréhension métier et efficacité opérationnelle.

---

# 7. Conclusion

Cette étude montre que l’exploitation des données clients permet d’améliorer significativement l’efficacité des campagnes de télé-marketing bancaire. L’analyse met en évidence que tous les clients ne présentent pas le même potentiel de souscription et que certaines caractéristiques, notamment liées aux interactions commerciales, jouent un rôle déterminant dans la décision finale.

La comparaison des modèles révèle que l’utilisation d’outils d’intelligence artificielle avancés, comme le Random Forest, permet de mieux identifier les clients à fort potentiel par rapport à des approches classiques. Cela se traduit concrètement par une réduction des appels inutiles, une meilleure allocation des ressources commerciales et une augmentation du taux de conversion des campagnes.

D’un point de vue décisionnel, ces résultats confirment l’intérêt d’intégrer des modèles prédictifs dans le processus de ciblage afin de soutenir les équipes commerciales et d’orienter les actions vers les clients les plus susceptibles de générer de la valeur.

---

## Pistes d’amélioration

* Ciblage plus fin des clients : Prioriser les clients avec une forte probabilité de souscription afin d’optimiser le temps des centres d’appels.
* Réduction des coûts opérationnels : Limiter les contacts auprès des clients à faible potentiel pour diminuer les dépenses liées aux campagnes.
* Intégration des coûts d’erreur : Ajuster les décisions en tenant compte du coût plus élevé des clients intéressés non contactés.
* Pilotage en temps réel : Utiliser les prédictions pour ajuster les campagnes au fil de leur déroulement.
* Aide à la décision commerciale : Fournir aux équipes des scores de probabilité simples à interpréter plutôt que des résultats techniques.
* Amélioration continue : Mettre à jour régulièrement les modèles afin de s’adapter à l’évolution du comportement des clients.

En conclusion, l’adoption d’un système de prédiction basé sur le machine learning constitue un levier stratégique de performance, permettant à la banque de concilier efficacité commerciale, maîtrise des coûts et amélioration de la relation client.


---

✅ **Analyse finalisée avec succès.**

