from flask import Flask, request, render_template
import pandas as pd
import joblib
import json

app = Flask(__name__)

# ==========================================
# 1. CARGAR MODELOS Y PREPROCESADORES
# ==========================================
model_hs = joblib.load("best_rf_model_hs.pkl")
model_costo = joblib.load("best_rf_model_costo.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Lista de columnas categóricas para la predicción
categorical_columns = ["Yacimiento", "Activo", "PU-FB", "MET PRODUCCION", "BATERIA", "Objetivo", "Obejtivo2"]

# ==========================================
# 2. FUNCIÓN PARA CONSTRUIR EL DICCIONARIO
#    (Activo -> {Yacimiento -> [BATERIA]})
# ==========================================
def build_activo_dict(df):
    """
    Construye un diccionario con la forma:
    {
      "ACTIVO_1": {
         "YACIMIENTO_1": ["BATERIA_1", "BATERIA_2", ...],
         "YACIMIENTO_2": [...],
         ...
      },
      "ACTIVO_2": { ... },
      ...
    }
    """
    data_dict = {}
    for _, row in df.iterrows():
        activo = row["Activo"]
        yacimiento = row["Yacimiento"]
        bateria = row["BATERIA"]
        
        # Inicializa estructuras si no existen
        if activo not in data_dict:
            data_dict[activo] = {}
        if yacimiento not in data_dict[activo]:
            data_dict[activo][yacimiento] = set()
        
        data_dict[activo][yacimiento].add(bateria)
    
    # Convertir sets a listas para poder serializar en JSON
    for act in data_dict:
        for yac in data_dict[act]:
            data_dict[act][yac] = list(data_dict[act][yac])
    
    return data_dict

# ==========================================
# 3. CARGAR EL EXCEL Y CREAR EL DICCIONARIO
# ==========================================
try:
    df = pd.read_excel("data/Bateria-Yacimiento-activo.xlsx")
    df = df.dropna()
    data_dict = build_activo_dict(df)
except Exception as e:
    print("Error cargando el archivo Excel:", e)
    data_dict = {}

# Convertir a JSON para enviarlo al template
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
        
        # Renderizar la respuesta:
        #  - prediction_hs y prediction_costo para mostrar resultados
        #  - input_data para Jinja (campos fijos)
        #  - input_data_json (convertido a JSON) para JS (campos dinámicos)
        return render_template(
            'index.html',
            prediction_hs=pred_hs,
            prediction_costo=pred_costo,
            input_data=input_data,                # Para los selects fijos (Jinja)
            input_data_json=json.dumps(input_data), # Para re-seleccionar en JS
            label_encoders=label_encoders,
            data_json=data_json
        )
    
    # GET: primera carga, sin selección previa
    return render_template(
        'index.html',
        label_encoders=label_encoders,
        data_json=data_json,
        input_data={},             # Campos fijos vacíos
        input_data_json="{}"       # Campos dinámicos vacíos
    )

# ==========================================
# 5. EJECUCIÓN LOCAL (OPCIONAL)
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)


