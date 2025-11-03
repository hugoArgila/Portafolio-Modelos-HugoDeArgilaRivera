# Modelo DNN para predecir si una talla de ropa se ajusta pequeña, grande o perfecta con API Flask

Esta carpeta contiene los archivos necesarios para entender, utilizar y probar un modelo de Red Neuronal Profunda (DNN) diseñado para predecir si una talla de ropa se ajusta pequeña, grande o perfecta. Además, incluye una API construida con Flask para interactuar con el modelo a través de una interfaz web sencilla.

## Contenido

* **`datos/`**:
    * **`tallas.csv`**: Dataset con registros de tallas de ropa, incluyendo características como medidas corporales, marcas, y etiquetas de ajuste (pequeña, grande, perfecta) utilizadas para entrenar y evaluar el modelo.

* **`Entrenamiento/`**:
    * **`EntrenamientoDNN.ipynb`**: Notebook de Jupyter que contiene una explicación detallada del proceso de entrenamiento del modelo DNN. Incluye:
        * **Preprocesamiento de datos:** Limpieza, imputación y transformación de variables categóricas y numéricas.
        * **Definición y entrenamiento del modelo:** Arquitectura de la red neuronal profunda, configuración de capas, compilación y ajuste.
        * **Evaluación:** Métricas para medir el desempeño del modelo en conjunto de validación o prueba.
        * **Visualizaciones:** Gráficos y análisis de resultados.
    * **`EntrenarRandomForest_LogisticRegression.ipynb`**: Notebook que muestra un enfoque alternativo para el mismo problema usando modelos de machine learning clásicos, como Random Forest y Regresión Logística, sin utilizar TensorFlow. Explica el preprocesamiento, entrenamiento, evaluación y comparación de estos modelos con el DNN.

* **`templates/`**:
    * **`index.html`**: Archivo HTML que define la estructura y los elementos de la interfaz web para interactuar con la API del modelo, permitiendo ingresar datos y obtener predicciones sobre el ajuste de la talla.

* **Archivos en la carpeta principal:**
    * **`app.py`**: Archivo Python que contiene la aplicación Flask para la API. Define rutas para recibir datos, preprocesarlos utilizando los transformadores guardados, realizar predicciones con el modelo DNN y mostrar resultados en la interfaz web.
    * **`keras_model.h5`**: Archivo con los pesos guardados del modelo DNN entrenado, en formato compatible con Keras.
    * **`cat_imputer.pkl`**: Objeto para imputación de valores faltantes en variables categóricas.
    * **`cat_onehot.pkl`**: Codificador One-Hot para transformar variables categóricas en formato numérico.
    * **`label_encoder.pkl`**: Codificador para las etiquetas de salida (pequeña, grande, perfecta).
    * **`num_imputer.pkl`**: Objeto para imputación de valores faltantes en variables numéricas.
    * **`num_scaler.pkl`**: Escalador para normalizar variables numéricas.

## Cómo utilizar

1.  **Explorar los Notebooks:**  
    Abre los archivos dentro de `Entrenamiento/` para entender en detalle los diferentes enfoques para el problema:
    * `EntrenamientoDNN.ipynb`: Entrenamiento y evaluación del modelo DNN con TensorFlow.  
    * `EntrenarRandomForest_LogisticRegression.ipynb`: Entrenamiento y evaluación de modelos clásicos de machine learning (Random Forest, Regresión Logística) sin TensorFlow.

2.  **Ejecutar la API de Flask:**
    * Asegúrate de tener instaladas todas las dependencias listadas en `requirements.txt` (no incluida aquí, pero necesaria). Puedes instalar las bibliotecas necesarias con:
        ```bash
        pip install -r requirements.txt
        ```
    * Navega hasta el directorio que contiene el archivo `app.py` en tu terminal.
    * Ejecuta la aplicación Flask con:
        ```bash
        python app.py
        ```
    * Una vez que la aplicación esté corriendo, abre tu navegador y accede a la dirección indicada (normalmente `http://127.0.0.1:5000/`).

3.  **Interactuar con la Interfaz Web:**
    * En la página web, podrás ingresar los datos necesarios para que el modelo prediga si la talla de ropa se ajusta pequeña, grande o perfecta.
    * Envía el formulario y observa la predicción generada por el modelo en la misma página.

## Requisitos

Para ejecutar la aplicación Flask y utilizar el modelo DNN, necesitarás tener instaladas las siguientes bibliotecas de Python (sugeridas):

* **Python** (versión recomendada: 3.x)
* **Flask**
* **TensorFlow** (con Keras)
* **scikit-learn**
* **NumPy**
* **pandas**
* Otras librerías necesarias que pueden listarse en un archivo `requirements.txt`.

## Notas

* El modelo DNN fue entrenado usando un dataset propio con medidas y etiquetas de ajuste de tallas.
* Los notebooks en `Entrenamiento/` contienen la arquitectura y detalles del proceso de entrenamiento tanto para DNN como para modelos clásicos de machine learning.
* La API Flask (`app.py`) utiliza los objetos de preprocesamiento guardados para transformar datos de entrada y realiza predicciones con el modelo DNN cargado desde `keras_model.h5`.
* La interfaz web está definida en `templates/index.html` y permite una experiencia simple para probar el modelo sin necesidad de código.
