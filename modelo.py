import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
data ={ #Datasets fictiops
    'Horas_Estudio': [2, 5, 1, 8, 6, 7, 3, 4, 9, 10],
    'Conocimiento_Previo': [3, 7, 2, 9, 6, 8, 4, 5, 9, 10],
    'Asistencia': [60, 90, 50, 100, 85, 95, 65, 70, 100, 100],

    'Promedio_Tareas': [60, 80, 50, 95, 85, 90, 70, 75, 100, 98],
    'Tipo_Estudiante': [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    'Resultado': [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]  #1 = Aprobado, 0 = No Aprobado
}

#Se convierte a DataFrame
df =pd.DataFrame(data)
X = df.drop('Resultado', axis=1)
y = df['Resultado']

#Escalacion de caracteristicas
escala = StandardScaler()
X_escalado = escala.fit_transform(X)
#División en conjuntos de entrenamiento y prueba
X_entrenado, X_prueba, y_entrenado, y_prueba = train_test_split(X_escalado, y, test_size=0.3, random_state=42)

#modelo arbol de decisión
model = DecisionTreeClassifier(random_state=42)
model.fit(X_entrenado, y_entrenado)

#predicciones
y_prediccion = model.predict(X_prueba)

#Ejecutar el modelo
print("Matriz de comparaciones de las predicciones:")
print(confusion_matrix(y_prueba, y_prediccion))
print("\nReporte de Clasificación:")
print(classification_report(y_prueba, y_prediccion))
print("Precisión:", accuracy_score(y_prueba, y_prediccion))
