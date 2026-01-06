## Code utilisé

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

print("\n=== Aperçu des données ===")
print(df.head())
print("\n=== Informations générales ===")
print(df.info())
print("\n=== Statistiques descriptives ===")
print(df.describe())

# ============================================================
# 2) ANALYSE DES VARIABLES NUMERIQUES
# ============================================================
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# HISTOGRAMMES
for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution de {col}")
    plt.show()
    print(f"\n📌 Commentaire Histogramme ({col}):")
    print(f" - La variable '{col}' présente sa distribution ci-dessus.")
    print(" - L’histogramme permet d’observer sa forme (normale, asymétrique, aplatie...).")
    print(" - La courbe KDE aide à visualiser les densités et éventuels pics.\n")

# BOXPLOTS
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()
    print(f"\n📌 Commentaire Boxplot ({col}):")
    print(f" - Le boxplot indique la distribution et les valeurs extrêmes de '{col}'.")
    print(" - Les points situés à l’extérieur indiquent des potentiels outliers.\n")

# HEATMAP
plt.figure(figsize=(10,6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()

print("\n📌 Commentaire Heatmap :")
print(" - La matrice montre les corrélations entre toutes les variables numériques.")
print(" - Les corrélations fortes (positives ou négatives) peuvent indiquer des relations utiles pour la modélisation.\n")

# ============================================================
# 3) FEATURE ENGINEERING
# ============================================================
df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100],
                          labels=['Jeune', 'Adulte', 'Senior', 'Très senior'])
df['duration_min'] = df['duration'] / 60
df['total_contacts'] = df['campaign'] + df['previous']

print("\n📌 Feature Engineering effectué :")
print(" - Création de 'age_group' pour regrouper les âges.")
print(" - Conversion de 'duration' en minutes.")
print(" - Création de 'total_contacts' : somme de campaign + previous.\n")

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

print("\n======================")
print("📌 RÉSULTATS : RÉGRESSION LOGISTIQUE")
print("======================")
print(classification_report(y_test, y_pred_log))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_log))

print("\n📌 Commentaire sur la Régression Logistique :")
print(" - La régression logistique fournit une base simple pour prédire l'abonnement.")
print(" - Si le recall sur la classe 1 (yes) est faible, le modèle a du mal à détecter les clients positifs.")
print(" - Une bonne précision signifie peu de faux positifs.")
print(" - Ce modèle est linéaire : il peut sous-performer si les relations ne sont pas linéaires.\n")

# ============================================================
# MODELE 2 : RANDOM FOREST
# ============================================================
rf_model = Pipeline(steps=[
    ('prep', preprocess),
    ('model', RandomForestClassifier(n_estimators=300, random_state=42))
])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n======================")
print("🌲 RÉSULTATS : RANDOM FOREST")
print("======================")
print(classification_report(y_test, y_pred_rf))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_rf))

print("\n📌 Commentaire sur le Random Forest :")
print(" - Ce modèle est souvent plus performant car il capture les non-linéarités.")
print(" - Il réduit le surapprentissage grâce à l’agrégation de nombreux arbres.")
print(" - Si le recall et la précision sont supérieurs à ceux de la régression logistique, le modèle est meilleur.")
print(" - Random Forest est généralement le meilleur choix sur Bank Marketing.\n")

# ============================================================
# 📌 COMMENTAIRE GLOBAL FINAL
# ============================================================
print("\n🎯 CONCLUSION GLOBALE :")
print(" - L’EDA révèle la distribution, la présence potentielle d’outliers et les relations entre variables.")
print(" - Le feature engineering enrichit les données pour améliorer les modèles.")
print(" - La régression logistique sert de modèle de base linéaire.")
print(" - Le Random Forest capture des relations plus complexes et donne souvent de meilleurs résultats.")
print(" - La comparaison des métriques permet de choisir le meilleur modèle pour la prédiction de l'abonnement.\n")
