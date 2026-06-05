import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement des données
df = pd.read_csv("data/raw_data/raw.csv")

# Extraction de l'heure, du mois, du jour de la semaine à partir de la date
df["date"] = pd.to_datetime(df["date"])
df["hour"] = df["date"].dt.hour
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df = df.drop(columns=["date"])

# Séparation features / cible
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarde
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)

print("Split terminé : X_train, X_test, y_train, y_test sauvegardés.")