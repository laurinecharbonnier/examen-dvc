import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pickle

# Chargement
X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze()

# Définition de la grille de paramètres
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
}

# GridSearch
xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Sauvegarde des meilleurs paramètres
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)

print(f"Meilleurs paramètres : {grid_search.best_params_}")