import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def carga_datos():
    """Carga datos simulados de publicidad y ventas."""
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)
    Y = np.array([2.0, 2.5, 3.5, 4.0, 4.5, 5.0])
    return X, Y

def divicion_datos(X, Y, test_size=0.3):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    return train_test_split(X, Y, test_size=test_size, random_state=42)

def planteamiento_modelo():
    """Devuelve el modelo de Regresión Lineal sin entrenar."""
    return LinearRegression()

def entramiento_modelo(modelo, X_train, Y_train):
    """Entrena el modelo con los datos de entrenamiento."""
    modelo.fit(X_train, Y_train)
    return modelo

def evaluacion_modelo(modelo, X_test, Y_test):
    """Evalúa el modelo con R² y retorna las predicciones."""
    Y_pred = modelo.predict(X_test)
    r2 = modelo.score(X_test, Y_test)
    print(f"Intersección (β₀): {modelo.intercept_:.2f} mil $MXN")
    print(f"Pendiente (β₁): {modelo.coef_[0]:.2f}")
    print(f"R² en prueba: {r2:.2f}")
    return Y_pred

def grafica(X_train, Y_train, X_test, Y_test, modelo, Y_pred):
    """Grafica entrenamiento, prueba y regresión."""
    X_all = np.vstack((X_train, X_test))
    plt.scatter(X_train, Y_train, color='blue', label='Entrenamiento')
    plt.scatter(X_test, Y_test, color='orange', label='Prueba real')
    plt.plot(X_all, modelo.predict(X_all), color='red', label='Regresión')
    plt.scatter(X_test, Y_pred, color='green', label='Predicción prueba')
    plt.title('Publicidad vs. Ventas (miles de $MXN)')
    plt.xlabel('Publicidad (miles de $MXN)')
    plt.ylabel('Ventas (miles de $MXN)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def proyeccion(modelo, nuevos_X):
    """Predice ventas para nuevos valores de publicidad."""
    nuevos_X = np.array(nuevos_X).reshape(-1, 1)
    predicciones = modelo.predict(nuevos_X)
    print("\n📌 Nuevas proyecciones:")
    for i, val in enumerate(nuevos_X):
        print(f"Gasto: {val[0]:.1f} mil $MXN → Ventas estimadas: {predicciones[i]:.2f} mil $MXN")
    return predicciones

def main():
    """Ejecuta el flujo completo del modelo."""
    X, Y = carga_datos()
    X_train, X_test, Y_train, Y_test = divicion_datos(X, Y)
    modelo = planteamiento_modelo()
    modelo = entramiento_modelo(modelo, X_train, Y_train)
    Y_pred = evaluacion_modelo(modelo, X_test, Y_test)
    grafica(X_train, Y_train, X_test, Y_test, modelo, Y_pred)
    proyeccion(modelo, [7.0, 8.5])

if __name__ == '__main__':
    main()