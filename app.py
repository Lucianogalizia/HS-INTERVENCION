from flask import Flask, request, render_template
import pandas as pd
import joblib
import json

app = Flask(__name__)

# ==========================================
# 1. CARGAR MODELOS Y PREPROCESADORES
# =========================================
model_hs = joblib.load("best_rf_model_hs.pkl")
model_costo = joblib.load("best_rf_model_costo.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Lista de columnas categóricas
categorical_columns = ["Yacimiento", "Activo", "PU-FB", "MET PRODUCCION", "BATERIA", "Objetivo", "Obejtivo2"]

# ==========================================
# 2. FUNCIÓN PARA CONSTRUIR EL DICCIONARIO
# ==========================================
def build_activo_dict(df):
    """
    Construye un diccionario con la estructura:
    {
      "Activo1": {
         "Yacimiento1": [BATERIA1, BATERIA2, ...],
         "Yacimiento2": [...],
         ...
      },
      "Activo2": { ... },
      ...
    }
    """
    data_dict = {}
    for _, row in df.iterrows():
        activo = row["Activo"]
        yacimiento = row["Yacimiento"]
        bateria = row["BATERIA"]
        if activo not in data_dict:
            data_dict[activo] = {}
        if yacimiento not in data_dict[activo]:
            data_dict[activo][yacimiento] = set()
        data_dict[activo][yacimiento].add(bateria)
    # Convertir sets en listas para serializar en JSON
    for act in data_dict:
        for yac in data_dict[act]:
            data_dict[act][yac] = list(data_dict[act][yac])
    return data_dict

# ==========================================
# 3. CARGAR DATOS PARA FILTRADO DINÁMICO
# ==========================================
try:
    # Asegúrate de actualizar la ruta al archivo Excel
    df = pd.read_excel("C:\Users\ry16123\OneDrive - YPF\Escritorio\DATOS ML\DATOS PU(3).xlsx")
    df = df.dropna()
    data_dict = build_activo_dict(df)
except Exception as e:
    print("Error cargando el archivo Excel:", e)
    # Si el archivo no está disponible en producción, puedes usar un diccionario por defecto
    data_dict = {}

# Convertir el diccionario a JSON para enviarlo a la plantilla
data_json = json.dumps(data_dict)

# ==========================================
# 4. RUTA PRINCIPAL
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Recoger datos del formulario
        input_data = {}
        for col in categorical_columns:
            input_data[col] = request.form.get(col)
        
        # Convertir valores a formato codificado
        encoded_values = []
        for col in categorical_columns:
            encoded_val = label_encoders[col].transform([input_data[col]])[0]
            encoded_values.append(encoded_val)
        
        # Crear DataFrame y estandarizar
        df_input = pd.DataFrame([encoded_values], columns=categorical_columns)
        df_input_scaled = scaler.transform(df_input)
        
        # Realizar predicciones
        pred_hs = model_hs.predict(df_input_scaled)[0]
        pred_costo = model_costo.predict(df_input_scaled)[0]
        
        return render_template(
            'index.html',
            prediction_hs=pred_hs,
            prediction_costo=pred_costo,
            input_data=input_data,
            label_encoders=label_encoders,
            data_json=data_json
        )
    
    # GET: pasar data_json y label_encoders
    return render_template('index.html', label_encoders=label_encoders, data_json=data_json)

# ==========================================
# 5. EJECUCIÓN LOCAL
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)


# ==========================================
# 4. EJECUCIÓN LOCAL
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)

