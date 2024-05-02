# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pickle

# Cargar datos
data = pd.read_csv("hprice.csv")

# Separar variables dependientes e independientes
X = data.drop("dependent_variable_column_name", axis=1)
y = data["dependent_variable_column_name"]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir los modelos
models = {
    "RandomForest": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
    "XGBoost": XGBRegressor(),
    "LGBM": LGBMRegressor()
}

# Definir grillas de hiperparámetros
param_grids = {
    "RandomForest": {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
    "DecisionTree": {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
    "XGBoost": {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]},
    "LGBM": {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
}

# Entrenamiento y optimización de hiperparámetros
best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], scoring='neg_root_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best {name} Model: {grid_search.best_params_}, RMSE: {-grid_search.best_score_}")

# Evaluación y selección del mejor modelo
best_model_name = min(best_models, key=lambda x: mean_squared_error(y_test, best_models[x].predict(X_test)))

print(f"Best Model: {best_model_name}, RMSE: {mean_squared_error(y_test, best_models[best_model_name].predict(X_test), squared=False)}")

# Guardar el mejor modelo
with open("model.pickle", "wb") as f:
    pickle.dump(best_models[best_model_name], f)
