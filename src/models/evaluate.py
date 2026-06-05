import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json

# Chargement
X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze()

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Prédictions
y_pred = model.predict(X_test_scaled)

# Métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

scores = {"mse": mse, "r2": r2, "mae": mae}
print(f"Métriques : {scores}")

# Sauvegarde des métriques
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)