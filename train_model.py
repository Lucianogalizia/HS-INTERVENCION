import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ===============================
# 1. CARGA Y PROCESAMIENTO DE DATOS
# ===============================
file_path = r"ruta/al/archivo/DATOS PU(3).xlsx"  # Ajusta la ruta
sheet_name = "2017-2024 INT"
df = pd.read_excel(file_path, sheet_name=sheet_name)
df = df.dropna()

# ===============================
# 2. FILTRAR Y CODIFICAR VARIABLES CATEGÓRICAS
# ===============================
columns = ["Yacimiento", "Activo", "PU-FB", "MET PRODUCCION", "BATERIA", "Objetivo", "Obejtivo2", "HS pulling", "COSTO"]
df = df[columns]
categorical_columns = ["Yacimiento", "Activo", "PU-FB", "MET PRODUCCION", "BATERIA", "Objetivo", "Obejtivo2"]
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ===============================
# 3. SEPARACIÓN DE DATOS EN ENTRENAMIENTO Y PRUEBA
# ===============================
X = df[["Yacimiento", "Activo", "PU-FB", "MET PRODUCCION", "BATERIA", "Objetivo", "Obejtivo2"]]
y_hs = df["HS pulling"]
y_costo = df["COSTO"]

X_train, X_test, y_hs_train, y_hs_test = train_test_split(X, y_hs, test_size=0.2, random_state=42)
_, _, y_costo_train, y_costo_test = train_test_split(X, y_costo, test_size=0.2, random_state=42)

# ===============================
# 4. ESTANDARIZACIÓN DE LOS DATOS
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 5. ENTRENAMIENTO Y OPTIMIZACIÓN DE RANDOM FOREST PARA HS Y COSTO
# ===============================
rf_model = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Optimización para HS pulling
grid_hs = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_hs.fit(X_train, y_hs_train)
best_rf_model_hs = grid_hs.best_estimator_

# Optimización para COSTO
grid_costo = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_costo.fit(X_train, y_costo_train)
best_rf_model_costo = grid_costo.best_estimator_

# ===============================
# 6. EVALUACIÓN FINAL
# ===============================
y_hs_pred = best_rf_model_hs.predict(X_test)
mae_hs = mean_absolute_error(y_hs_test, y_hs_pred)
r2_hs = r2_score(y_hs_test, y_hs_pred)

y_costo_pred = best_rf_model_costo.predict(X_test)
mae_costo = mean_absolute_error(y_costo_test, y_costo_pred)
r2_costo = r2_score(y_costo_test, y_costo_pred)

print("Evaluación HS pulling - MAE: {:.2f}, R²: {:.2f}".format(mae_hs, r2_hs))
print("Evaluación COSTO - MAE: {:.2f}, R²: {:.2f}".format(mae_costo, r2_costo))

# ===============================
# 7. GUARDAR LOS MODELOS Y PREPROCESADORES
# ===============================
joblib.dump(best_rf_model_hs, "best_rf_model_hs.pkl")
joblib.dump(best_rf_model_costo, "best_rf_model_costo.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Modelos guardados exitosamente.")
