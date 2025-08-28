# 🚢 Modelo de Predicción de Supervivencia en el Titanic

Este proyecto se centra en la aplicación de un modelo de **Random Forest** para predecir la supervivencia de los pasajeros del famoso desastre del Titanic. Los datos y la competición utilizados son de la clásica competencia de **Kaggle** "Titanic: Machine Learning from Disaster".

---

## 📂 Archivos del Proyecto

- **`Titanic_Entrenamiento.ipynb`**: Este notebook de Jupyter contiene el proceso completo de desarrollo del modelo. Aquí encontrarás los pasos de:
    - **Análisis exploratorio de datos (EDA)** y preprocesamiento.
    - **Entrenamiento** del modelo Random Forest.
    - **Afinación** de hiperparámetros (**fine-tuning**) para mejorar el rendimiento.
- **`gender_submission.csv`**: Este archivo contiene las predicciones finales generadas por el modelo, siguiendo el formato requerido por la competición de Kaggle. Con este modelo logré un **74% de precisión** al compararlo con los datos reales en la plataforma.
- **`train.csv`**: El conjunto de datos de entrenamiento, utilizado para construir el modelo.
- **`test.csv`**: El conjunto de datos de prueba, usado para generar las predicciones.
- **`requirements.txt`**: Un listado de todas las librerías necesarias para ejecutar el notebook y replicar el proyecto.

---

## 📈 Desempeño y Contexto

El modelo alcanzó un 74% de precisión en la competición de Kaggle, lo que demuestra su capacidad para capturar patrones significativos en los datos. La competición completa se puede encontrar en el siguiente enlace:

[**Titanic: Machine Learning from Disaster**](https://www.kaggle.com/competitions/titanic/overview)

---

## 🛠️ Requisitos e Instalación

Para ejecutar el notebook, necesitarás las librerías listadas en `requirements.txt`. Puedes instalarlas fácilmente desde tu terminal:

```bash
pip install -r requirements.txt
