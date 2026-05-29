# Instituto Tecnológico y de Estudios Superiores de Monterrey Campus Querétaro
# Impact of AI on Students
### Arturo Sánchez Rodríguez | A01275427

## Introducción
En este proyecto se va a trabajar con el Dataset de estudiantes que tienen probabilidad de Burnout (High, Medium, Risk) y el posible impacto que tienen las herramientas de Inteligencia Artificial.
Buscamos solucionar un problema de clasificación ya que la salida va a ser uno de las tres probabilidades de Burnout la cual es una categoria.

## Dataset
El Dataset tiene 50,000 registros uno por cada estudiante.
- Student_ID
- Major_Category
- Year_of_Study
- Pre_Semester_GPA
- Weekly_GenAI_Hours
- Primary_Use_Case
- Prompt_Engineering_Skill
- Tool_Diversity
- Paid_Subscription
- Traditional_Study_Hours
- Perceived_AI_Dependency
- Institutional_Policy
- Anxiety_Level_During_Exams
- Post_Semester_GPA
- Skill_Retention_Score
- **Burnout_Risk_Level** (El importante que vamos a utilizar)<br>
[Descargar CSV del DataSet](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students)

## Dependencias/Librerías
- pandas  
- Manipulación y análisis de datos 
- numpy
- Operaciones matemáticas y arreglos numéricos 
- scikit-learn
- Preprocesado de datos y herramientas de Machine Learning 
- matplotlib 
- Visualización de datos

## Funcionalidad Primera Parte
1. Generación o selección del set de datos  20% del módulo.
2. Obtener, generar o aumentar un set de datos.
3. Hacer la separación de los sets de prueba y entrenamiento.
4. Preprocesado de los datos 20% del módulo
5. Aplicar las técnicas de escalamiento.
6. Hacer el preprocesado pertinente de los datos

# Flujo de funcionamiento

## 1. Importamos las librerías necesarias
```
import pandas as pd #Para trabajar con tablas y datos.
import numpy as np #Para operaciones matemáticas y arreglos numéricos.

from sklearn.model_selection import train_test_split #Para dividir los datos de test y train.
from sklearn.preprocessing import StandardScaler #Para el escalamiento de los datos (los datos/tamaños distintos en el DataSet).
```

Aqui se utilizó sklearn debido a que estamos trabajando con sklearn debido a que estamos trabajando con un dataset tabular que es un archivo CSV por lo que fue lo más apropiado.

## 2. Cargamos el Dataset
```
df = pd.read_csv("ai_student_impact_dataset.csv")

#Ver primeras filas
print("Primeras filas del dataset:")
df.head()

#Ver información general
print("\nInformación del dataset:")
df.info()

#Revisar valores nulos
print("\nValores nulos por columna:")
df.isnull().sum()

#Eliminar filas que tengan valores nulos si es que lo hay
df = df.dropna()
```
El data set fue extraído desde Kaggle y se cargo manualmente utilizando pandas. Con "pd.read_csv()" ya que son datos tabulares.
Utilizando "head(), info(), isnull()" se revisa la estructura que tiene el Dataset
- head() = para ver ejemplos de los registros
- info() = para conocer el tipo de datos y la cantidad de columnas que tiene el dataset
- isnull() = para validar que no existan valores faltantes en el dataset


## 3. Separamos las variables de entrada X y variables objetivo Y. (Entrada y salida)
```
X = df.drop("Burnout_Risk_Level", axis=1) #Con el ".drop()" quitamos la columna de burnout para que no vea la respuesta el modelo
y = df["Burnout_Risk_Level"] #Aqui tomamos solo la columna de burnout para que sea la respuesta del modelo
```

Aqui es donde separamos los datos que el modelo va a utilizar para aprender separamos X que es la información que el modelo va a utilizar para aprender, Y siendo la respuesta correcta que queremos que prediga.
Eliminamos "Burnout_Risk_Level" de X para que el modelo NO vea la respuesta antes de intentar predecirla.

## 4. Convertimos las colúmnas categóricas de X en números
```
X = pd.get_dummies(X, drop_first=True) #Convertimos las variables categóricas (el texto) a números para que sea algo que el modelo pueda entender y eliminamos la primera categoría automáticamente con "drop_first=True" para evitar redundancia en el modelo.

#Separar los datos Train y Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #Separamos Test y Train, "0.2" es el 20% para test y 80% para train, "random_state=42" es para que siempre se separen de manera aleatoria, "stratify=y" mantiene el balance de las clases train y test con porciones similares.
```

Como las entradas no son valores numéricos tnemos que convertirlos de texto a números ya que los modelos normalmente trabajan con datos numéricos por lo que utilizamos "pd.get_dummies()" para transformarlos.
También lo separamos en Train y Test, se dividió en 20% para prueba y 80% para entrenamiento, lo dividimos en 4 archivos datos de entrenamiento, datos de prueba, entradas de entrenamiento y entradas de respuestas.
- X_train = Información para entrenar el modelo
- X_test = Información para probar el modelo
- Y_train = Respuestas correctas del entrenamiento
- Y_test = Respuestas correctas de la prueba

Tambien utilizamos "stratify=y" para balancear las categorías en Entrenamiento y Prueba para que cuando se separen mantenga balance en el test y en el train para que el modelo aprenda de manera justa.

## 5. Aplicamos Escalamiento
```
scaler = StandardScaler() #Balanceamos las esclas para que el modelo no se vaya por números grandes o pequeños.

X_train_scaled = scaler.fit_transform(X_train) #Con "fit" lo que hace es que aprende cómo son los datos y con "transform" convertimos los números centrandolos alrededor de 0 y con eso tenemos el entrenamiento ya escalado.
X_test_scaled = scaler.transform(X_test) #Con el test no aplicamos "fit" porque no queremos que se aprenda los datos, queremos que el test simule datos nuevos, por eso solo aplicamos "transform" para que se escale con la misma escala que el entrenamiento.

#Convertir los datos escalados a DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns) #Convertimos de nuevo los datos a tabla pandas para que sea más fácil de manejar y con "columns=X.xolumns" le ponemos los nombres originales de columnas (Restaura los nombre). 
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
```

Como no estamos trabajando con imágenes donde los valores de pixeles se dividen entre 255 para que tuvieran una escala similar, aqui estamos trabajando con datos tabulares por eso utilizamos "StandardScaler()" para centrar los datos alrededor de 0 para que ninguna variable domine al modelo por la diferencia de escala.

- StamdardScaler() = Balancea para que el modelo no le de más importancia a números grandes o pequeños solo por su valor
- fit() = Con esto hacemos que el scaler aprenda los datos de entrenamiento, despues con "transform()" convertimos los números a una escala centrada alrededor de 0 para obtener el conjunto de entrenamiento ya escalado. En el conjunto de prueba no lo utilizamos para que el modelo no aprenda la información de esos datos.

Por último convertimos los datos escalados de nuevo a DataFrame para seguir trabajando con ellos.

## 6. Resultados
```
#Mostrar resultados finales
print("\nTamaño de X_train:", X_train.shape) #Aqui (x) con ".shape" vemos las dimensiones de la tabla pero si nuestro dataset original tiene 16 columnas porque hay 16? Es por el "get_dummies" que convierte de texto a números y para eso necesita crear columnas nuevas de las categorías.
print("Tamaño de X_test:", X_test.shape)
print("Tamaño de y_train:", y_train.shape) #Aqui (y) solo muestra una columna que es la respuesta que buscamos predecir, por eso solo muestra una dimensión.
print("Tamaño de y_test:", y_test.shape)
print("\nClases de la variable objetivo:")
print(y.value_counts())
print("\nPrimeras filas de X_train escalado:")
print(X_train_scaled.head())
```

Despues del preprocesado el dataset quedo dividido en:
- 40,000 registros de entrenamiento
- 10,000 registros de pruebas

Tambien podemos notar que el número de columnas aumentó de 16 a 26 porque utilizamos "get_dummies()" que convirtió algunas variables de texto en columnas numéricas para que el modelo lo pueda entender.
- .shape() = vemos las dimensiones de las tablas
- En Y solo existe una columna de respuesta porque queremos predecir "Burnout_Risk_Level"
- .head() = mostramos las primeras filas de los datos para comprobar que el preprocesado se realizó correctamente

## 7. Guardado de los archivos procesados
```
X_train_scaled.to_csv("X_train_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_scaled.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("\nArchivos generados correctamente.")
```

Después de terminar el preprocesado se guardaron los conjuntos de datos ya preparados en diferentes archivos CSV. Esto se hizo para poder continuar en las siguientes etapas del proyecto donde se entrenará el modelo de clasificación.

## Conclusión 

Durante esta primera etapa del proyecto se logró preparar correctamente el dataset, se realizó la exploración de los datos, se identificaron las variables, se limpiaron los datos, se transformaron las variables categóricas a números, se separaron en entrenamiento y prueba y se aplico escalamiento. 
Como se trabajaron con datos tabulares almacenados en un CSV fue necesario utilizar herramientas diferentes como pandas y scikit-learn para el preprocesado en lugar de tensorflow.
Podemos decir que el dataset quedo preparado para las siguientes etapas del proyecto.
























