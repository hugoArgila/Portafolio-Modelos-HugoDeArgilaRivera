# Modelo CNN para predecir numeros escritos a mano en códigos postales con API Flask

Esta carpeta contiene los archivos necesarios para entender, utilizar y probar un modelo de Red Neuronal Recurrente (RNN) diseñado para predecir numeros escritos digitalmente, principalmente para códigos postales. Además, incluye una API construida con Flask para interactuar con el modelo a través de una interfaz web sencilla.

## Contenido

* **`static/`**:
    * **`uploads/`**: Carpeta donde se guardan imágenes para probar nuestro modelo, siempre se podría probar con nuevas imágenes hechas por ti, por ejemplo en paint.
    * **`style.css`**: Archivo de hojas de estilo en cascada para personalizar la apariencia de la interfaz web.

* **`templates/`**:
    * **`index.html`**: Archivo HTML que define la estructura y los elementos de la interfaz web para interactuar con la API del modelo.

* **`Entrenamiento_modelo_CNN_codigos_postales.ipynb`**: Este notebook de Jupyter contiene una explicación detallada del proceso de entrenamiento del modelo RNN. Incluye:
    * **Preprocesamiento de datos:** Pasos realizados para preparar los datos para el entrenamiento.
    * **Arquitectura del modelo:** Definición de las capas RNN y otras capas utilizadas.
    * **Entrenamiento:** Código para entrenar el modelo utilizando los datos preparados.
    * **Evaluación:** Métricas utilizadas para evaluar el rendimiento del modelo en datos de prueba.
    * **Visualizaciones:** Gráficos para ilustrar el proceso de entrenamiento y los resultados.

* **`app.py`**: Este archivo de Python contiene el código de la aplicación Flask que define la API para interactuar con el modelo RNN. Incluye rutas para cargar datos (posiblemente imágenes en la carpeta `uploads`), procesarlos con el modelo y mostrar los resultados a través de la interfaz web (`index.html`).

* **`requirements.txt`**: Este archivo lista las dependencias de Python necesarias para ejecutar la aplicación Flask y utilizar el modelo RNN. Se utiliza para instalar las bibliotecas requeridas con `pip`.

* **`modelRNNnum.h5`**: Este archivo contiene los pesos entrenados del modelo RNN guardado en formato HDF5. Este formato es comúnmente utilizado por bibliotecas como Keras para almacenar modelos de redes neuronales.

## Cómo utilizar

1.  **Explorar el Notebook:** Abre el archivo `RNN.ipynb` para entender en detalle la implementación del modelo RNN, el proceso de entrenamiento y los resultados obtenidos. Puedes ejecutar las celdas del notebook en un entorno de Jupyter para revisar el código y las salidas.

2.  **Ejecutar la API de Flask:**
    * Asegúrate de tener instaladas todas las dependencias listadas en `requirements.txt`. Puedes instalarlas ejecutando el siguiente comando en tu terminal:
        ```bash
        pip install -r requirements.txt
        ```
    * Navega hasta el directorio que contiene el archivo `app.py` en tu terminal.
    * Ejecuta la aplicación Flask utilizando el siguiente comando:
        ```bash
        python app.py
        ```
    * Una vez que la aplicación se esté ejecutando, podrás acceder a la interfaz web abriendo tu navegador y navegando a la dirección que se mostrará en la terminal (normalmente `http://127.0.0.1:5000/`).

3.  **Interactuar con la Interfaz Web:**
    * Abre la URL de la aplicación Flask en tu navegador.
    * Deberías ver la interfaz definida en `templates/index.html`.
    * Sigue las instrucciones en la página web para cargar datos (por ejemplo, imágenes en la sección correspondiente si tu modelo procesa imágenes) y obtener predicciones del modelo RNN. Los resultados se mostrarán en la misma página o en una página de resultados.

## Requisitos

Para ejecutar la aplicación Flask y utilizar el modelo RNN, necesitarás tener instaladas las siguientes bibliotecas de Python (listadas en `requirements.txt`):

* **Python** (versión recomendada: 3.x)
* **Flask**
* **TensorFlow** (con Keras)
* **NumPy**
* **Pillow** 
* Otras bibliotecas que puedan estar listadas en `requirements.txt`.

## Notas

* Este modelo RNN fue entrenado usando el dataset de MINST de números, muy utilizado para investigación.
* La arquitectura del modelo y los hiperparámetros están definidos en el notebook `Entrenamiento_modelo_CNN_codigos_postales.ipynb`.
* La API de Flask (`app.py`) se encarga de cargar el modelo entrenado (`modelRNNnum.h5`), recibir los datos a través de la interfaz web y realizar las predicciones utilizando el modelo.
* El archivo `static/style.css` contiene estilos para mejorar la presentación de la interfaz web definida en `templates/index.html`.
