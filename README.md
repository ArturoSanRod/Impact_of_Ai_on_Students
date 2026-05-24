# Proyecto de Machine Learning
# Clasificación de Riesgo de Burnout Académico

## Descripción del proyecto

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning capaz de clasificar el nivel de riesgo de burnout académico en estudiantes a partir de diferentes variables relacionadas con el uso de herramientas de inteligencia artificial, hábitos de estudio, ansiedad y desempeño académico.

El proyecto se desarrollará de manera progresiva durante 5 semanas, siguiendo las etapas fundamentales de un flujo de trabajo de Machine Learning.

---

# Dataset utilizado

Se utilizó el dataset:

```text
Impact of Ai on Students
```

El dataset contiene información de estudiantes como:

- Horas de uso de herramientas de IA
- Horas de estudio tradicional
- Nivel de ansiedad durante exámenes
- GPA antes y después del semestre
- Dependencia percibida de IA
- Nivel de habilidades de prompt engineering
- Riesgo de burnout académico

La variable objetivo del proyecto es:

```text
Burnout_Risk_Level
```

Las clases a predecir son:

- Low
- Medium
- High

---

# Objetivo del proyecto

Desarrollar un modelo de clasificación capaz de predecir el nivel de burnout académico utilizando técnicas de Machine Learning y preprocesado de datos.

---

# Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

# Desarrollo por semanas

# Semana 2 — Preparación y preprocesado de datos

## Objetivos

- Obtener y seleccionar un dataset
- Explorar el dataset
- Realizar limpieza de datos
- Separar datos de entrenamiento y prueba
- Aplicar técnicas de escalamiento
- Preparar los datos para Machine Learning

## Actividades realizadas

### 1. Carga del dataset

Se utilizó pandas para cargar el dataset CSV:

```python
pd.read_csv()
```

---

### 2. Exploración de datos

Se revisaron:

- Primeras filas del dataset
- Tipos de datos
- Cantidad de registros
- Valores nulos

Funciones utilizadas:

```python
df.head()
df.info()
df.isnull().sum()
```

---

### 3. Preprocesado de datos

Se realizaron las siguientes tareas:

- Eliminación de valores nulos
- Conversión de variables categóricas a numéricas usando:

```python
pd.get_dummies()
```

---

### 4. Separación Train/Test

Los datos fueron divididos en:

- 80% entrenamiento
- 20% prueba

Usando:

```python
train_test_split()
```

---

### 5. Escalamiento de datos

Se aplicó escalamiento utilizando:

```python
StandardScaler()
```

Esto permitió normalizar las variables numéricas para mejorar el rendimiento futuro del modelo.

---

### 6. Exportación de datasets procesados

Se generaron los siguientes archivos:

- X_train_scaled.csv
- X_test_scaled.csv
- y_train.csv
- y_test.csv

Estos archivos serán utilizados en las siguientes etapas del proyecto.

---

# Semana 3 — Implementación del modelo

## Objetivos

- Seleccionar un modelo de Machine Learning
- Implementar el modelo usando Scikit-learn
- Entrenar el modelo con los datos procesados

## Actividades planeadas

- Selección del modelo
- Entrenamiento del modelo
- Generación de predicciones
- Evaluación inicial

---

# Semana 3 — Evaluación inicial del modelo

## Objetivos

- Evaluar el desempeño del modelo
- Aplicar métricas de clasificación
- Interpretar resultados

## Métricas planeadas

- Accuracy
- Precision
- Recall
- F1-Score

---

# Semana 4 — Refinamiento del modelo

## Objetivos

- Mejorar el desempeño del modelo
- Ajustar hiperparámetros
- Comparar resultados

## Actividades planeadas

- Optimización del modelo
- Ajuste de parámetros
- Comparación de métricas
- Análisis de mejoras

---

# Semana 5 — Entrega final

## Objetivos

- Documentar correcciones
- Organizar resultados finales
- Entregar versión final del proyecto

## Actividades planeadas

- Corrección de errores
- Limpieza del código
- Actualización de documentación
- Entrega final del repositorio

---

# Estructura del proyecto

```text
Proyecto/
│
├── sem2.ipynb
├── ai_student_impact_dataset.csv
├── X_train_scaled.csv
├── X_test_scaled.csv
├── y_train.csv
├── y_test.csv
├── README.md
└── requirements.txt
```

---

# Autor

Proyecto desarrollado como parte del módulo de Machine Learning.
