import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# --- Configuración y Rutas de Archivos ---
NUM_IMPUTER_PATH = 'num_imputer.pkl'
NUM_SCALER_PATH = 'num_scaler.pkl'
CAT_IMPUTER_PATH = 'cat_imputer.pkl'
CAT_ONEHOT_PATH = 'cat_onehot.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
KERAS_MODEL_H5_PATH = 'keras_model.h5'

# Inicialización de los transformadores y el modelo
num_imputer_loaded = None
num_scaler_loaded = None
cat_imputer_loaded = None
cat_onehot_loaded = None
label_encoder = None
keras_model_loaded = None
model_tf = None # model_tf ahora será el modelo keras directamente

# Definición de columnas (DEBE COINCIDIR CON EL NOTEBOOK DESPUÉS DE ELIMINAR 'Marca')
numerical_cols = ['Medida_Pecho_cm', 'Medida_Cintura_cm', 'Medida_Cadera_cm', 'Altura_cm', 'Peso_kg', 'Edad']
categorical_cols = ['Tipo_Prenda', 'Talla_Ropa_General', 'Talla_Zapatillas', 'Talla_Pantalones']


# --- Carga de Componentes del Modelo ---
# Bandera para rastrear si todos los componentes se cargaron correctamente
all_components_loaded_successfully = True

try:
    if os.path.exists(NUM_IMPUTER_PATH):
        num_imputer_loaded = joblib.load(NUM_IMPUTER_PATH)
        print("Imputer Numérico cargado exitosamente.")
    else:
        print(f"ADVERTENCIA: Archivo de Imputer Numérico no encontrado en '{NUM_IMPUTER_PATH}'.")
        all_components_loaded_successfully = False

    if os.path.exists(NUM_SCALER_PATH):
        num_scaler_loaded = joblib.load(NUM_SCALER_PATH)
        print("Scaler Numérico cargado exitosamente.")
    else:
        print(f"ADVERTENCIA: Archivo de Scaler Numérico no encontrado en '{NUM_SCALER_PATH}'.")
        all_components_loaded_successfully = False

    if os.path.exists(CAT_IMPUTER_PATH):
        cat_imputer_loaded = joblib.load(CAT_IMPUTER_PATH)
        print("Imputer Categórico cargado exitosamente.")
    else:
        print(f"ADVERTENCIA: Archivo de Imputer Categórico no encontrado en '{CAT_IMPUTER_PATH}'.")
        all_components_loaded_successfully = False

    if os.path.exists(CAT_ONEHOT_PATH):
        cat_onehot_loaded = joblib.load(CAT_ONEHOT_PATH)
        print("OneHotEncoder Categórico cargado exitosamente.")
    else:
        print(f"ADVERTENCIA: Archivo de OneHotEncoder Categórico no encontrado en '{CAT_ONEHOT_PATH}'.")
        all_components_loaded_successfully = False

    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print("LabelEncoder cargado exitosamente.")
    else:
        print(f"ADVERTENCIA: Archivo de LabelEncoder no encontrado en '{LABEL_ENCODER_PATH}'.")
        all_components_loaded_successfully = False

    if os.path.exists(KERAS_MODEL_H5_PATH):
        keras_model_loaded = tf.keras.models.load_model(KERAS_MODEL_H5_PATH)
        print("Modelo Keras (.h5) cargado exitosamente.")
    else:
        print(f"ADVERTENCIA: Archivo de modelo Keras (.h5) no encontrado en '{KERAS_MODEL_H5_PATH}'.")
        all_components_loaded_successfully = False

    # Asignar model_tf solo si todos los componentes se cargaron con éxito
    if all_components_loaded_successfully:
        model_tf = keras_model_loaded
        print("Todos los componentes del modelo han sido cargados y el modelo Keras está listo para usar.")
    else:
        # Si no todos los componentes se cargaron, asegura que model_tf esté en None
        # para que el endpoint de predicción maneje el error.
        model_tf = None 
        # Ya se imprimieron advertencias individuales, no se necesita un mensaje general aquí.

except Exception as e:
    print(f"Error general al cargar o inicializar los componentes del modelo: {e}")
    all_components_loaded_successfully = False
    model_tf = None
    label_encoder = None


# --- NUEVA RUTA PARA SERVIR LA PÁGINA WEB ---
@app.route('/')
def index():
    """
    Ruta principal que renderiza el archivo index.html.
    Flask buscará automáticamente 'index.html' en la carpeta 'templates'.
    """
    return render_template('index.html')

# --- Endpoint de Predicción ---
@app.route('/predict', methods=['POST'])
def predict():
    # La verificación al inicio de predict() ya es suficiente para el manejo de errores
    if model_tf is None or label_encoder is None or num_imputer_loaded is None or num_scaler_loaded is None or cat_imputer_loaded is None or cat_onehot_loaded is None:
        return jsonify({'error': 'Modelo o codificadores no cargados. Por favor, asegúrate de que todos los archivos .pkl y .h5 existen y son válidos.'}), 500

    try:
        data = request.get_json(force=True)
        
        # Columnas esperadas (10 parámetros, sin 'Marca')
        expected_columns = [
            'Tipo_Prenda', 'Medida_Pecho_cm', 'Medida_Cintura_cm',
            'Medida_Cadera_cm', 'Altura_cm', 'Peso_kg', 'Edad',
            'Talla_Ropa_General', 'Talla_Zapatillas', 'Talla_Pantalones'
        ]
        
        processed_data = {}
        for col in expected_columns:
            value = data.get(col)
            if value == '': # Tratar cadenas vacías como None para imputación
                processed_data[col] = None
            elif col in numerical_cols:
                processed_data[col] = float(value) if value is not None else None
            else:
                processed_data[col] = value
        
        input_df = pd.DataFrame([processed_data])

        # --- Preprocesamiento manual de los datos de entrada para la predicción ---
        input_num = input_df[numerical_cols]
        input_num_imputed = num_imputer_loaded.transform(input_num)
        input_num_scaled = num_scaler_loaded.transform(input_num_imputed)

        input_cat = input_df[categorical_cols]
        # Asegúrate de que las columnas categóricas se traten como tipo 'object' o 'category' antes de imputar si es necesario
        for col in categorical_cols:
            if col in input_cat.columns:
                input_cat[col] = input_cat[col].astype('object') # O 'category' si SimpleImputer lo maneja bien

        input_cat_imputed = cat_imputer_loaded.transform(input_cat)
        input_cat_encoded = cat_onehot_loaded.transform(input_cat_imputed)

        transformed_input_data = np.hstack((input_num_scaled, input_cat_encoded.toarray()))

        # Realizar la predicción con el modelo Keras directamente
        # predict_proba() devuelve probabilidades para cada clase
        predictions_proba = model_tf.predict(transformed_input_data)
        # np.argmax convierte las probabilidades en el índice de la clase predicha
        prediction_encoded = np.argmax(predictions_proba, axis=1) 
        predicted_label = label_encoder.inverse_transform(prediction_encoded)[0]

        return jsonify({'feedback_ajuste': predicted_label})

    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({'error': str(e), 'message': 'Ocurrió un error durante la predicción. Asegúrate de que los datos de entrada son correctos y completos.'}), 400

# --- Ejecutar la Aplicación ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
