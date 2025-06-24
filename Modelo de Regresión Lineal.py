import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Datos teóricos
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)  # Gasto en publicidad
Y = np.array([2.0, 2.5, 3.5, 4.0, 4.5, 5.0])  # Ventas

# División en conjunto de entrenamiento (70%) y prueba (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Crear el modelo y entrenarlo
model = LinearRegression()
model.fit(X_train, Y_train)

# Predicción sobre el conjunto de prueba
Y_test_pred = model.predict(X_test)

# Evaluación del modelo: R²
r2_score_test = model.score(X_test, Y_test)
print(f"Coeficiente R² en el conjunto de prueba: {r2_score_test:.2f}")

# Visualización
plt.scatter(X_train, Y_train, color='blue', label='Entrenamiento')
plt.scatter(X_test, Y_test, color='orange', label='Prueba real')
plt.plot(X, model.predict(X), color='red', label='Línea de regresión')
plt.scatter(X_test, Y_test_pred, color='green', label='Predicción prueba')
plt.title('Regresión Lineal: División Entrenamiento / Prueba')
plt.xlabel('Gasto en Publicidad (miles $)')
plt.ylabel('Ventas (miles $)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()