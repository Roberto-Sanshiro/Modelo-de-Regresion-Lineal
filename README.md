# 📊 Regresión Lineal Simple en Python con Scikit-learn

Este repositorio muestra cómo implementar un modelo de **Regresión Lineal Simple** utilizando Python y la biblioteca `scikit-learn`. El propósito principal es **predecir las ventas mensuales (en miles de pesos mexicanos, $MXN)** a partir del **gasto mensual en publicidad**.

El proyecto está diseñado con un enfoque educativo, destacando buenas prácticas como la separación en funciones, claridad en el flujo de trabajo y visualización de resultados. Ideal para estudiantes, docentes y personas interesadas en aprender los fundamentos del aprendizaje supervisado.

---

## 🧠 ¿Qué es la regresión lineal simple?

La regresión lineal simple es una técnica estadística utilizada para modelar la relación entre dos variables numéricas: una independiente (X) y una dependiente (Y). Su objetivo es encontrar la mejor línea recta que ajuste los datos.

### Fórmula general:

Y = β₀ + β₁X + ε


Donde:
- **Y** es la variable a predecir (ventas)
- **X** es la variable explicativa (publicidad)
- **β₀** es la intersección (valor base cuando X = 0)
- **β₁** es la pendiente (cambio en Y por unidad de X)
- **ε** es el término de error (residuo)

---

## 📂 Estructura del proyecto

El archivo principal `regresion_lineal.py` contiene todas las funciones necesarias para llevar a cabo el flujo completo de un modelo de regresión supervisada:

| Función                    | Descripción                                                                 |
|---------------------------|------------------------------------------------------------------------------|
| `cargar_datos()`          | Carga un conjunto simulado de datos en miles de $MXN                         |
| `planteamiento_modelo()`  | Crea e inicializa el modelo de regresión lineal                              |
| `entrenamiento_modelo()`  | Divide los datos y entrena el modelo con el conjunto de entrenamiento        |
| `evaluar_modelo()`        | Evalúa el modelo sobre los datos de prueba y muestra los coeficientes        |
| `graficar_resultados()`   | Muestra los datos, la línea de regresión y las predicciones                  |
| `predecir_nuevos_valores()` | Realiza predicciones para nuevos valores no vistos durante el entrenamiento |
| `main()`                  | Función principal que orquesta todo el flujo                                 |

---

## 🚀 Cómo ejecutar el proyecto

1. Clona el repositorio o descarga el archivo `regresion_lineal.py`.
2. Asegúrate de tener Python instalado (versión 3.7 o superior).
3. Instala las dependencias necesarias con:

```bash
pip install numpy matplotlib scikit-learn
