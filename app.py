from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar modelos y preprocesadores guardados
model_hs = joblib.load("best_rf_model_hs.pkl")
model_costo = joblib.load("best_rf_model_costo.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Lista de columnas categóricas (debe coincidir con el entrenamiento)
categorical_columns = ["Yacimiento", "Activo", "PU-FB", "MET PRODUCCION", "BATERIA", "Objetivo", "Obejtivo2"]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Recoger datos del formulario
        input_data = {}
        for col in categorical_columns:
            input_data[col] = request.form.get(col)
        
        # Convertir valores a formato adecuado (codificar)
        encoded_values = []
        for col in categorical_columns:
            # El encoder espera una lista, devuelve un array con el valor codificado
            encoded_val = label_encoders[col].transform([input_data[col]])[0]
            encoded_values.append(encoded_val)
        
        # Crear DataFrame y estandarizar
        df_input = pd.DataFrame([encoded_values], columns=categorical_columns)
        df_input_scaled = scaler.transform(df_input)
        
        # Realizar predicciones
        pred_hs = model_hs.predict(df_input_scaled)[0]
        pred_costo = model_costo.predict(df_input_scaled)[0]
        
        # Renderizar la respuesta
        return render_template('index.html', prediction_hs=pred_hs, prediction_costo=pred_costo, input_data=input_data)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
