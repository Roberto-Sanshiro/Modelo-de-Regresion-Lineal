## 📊 Regresión Lineal Simple con Scikit-learn

Este repositorio muestra cómo implementar un modelo de **Regresión Lineal Simple** en Python utilizando `scikit-learn`, una de las bibliotecas más populares para **Machine Learning**. El objetivo principal es predecir las ventas mensuales (en miles de dólares) en función del gasto mensual en publicidad (también en miles de dólares).

Este ejemplo es **didáctico** y está diseñado para ayudar a comprender los fundamentos de los modelos lineales supervisados, su entrenamiento, evaluación y visualización.

---

### 🧱 Fundamento teórico

La regresión lineal simple intenta modelar la relación entre una variable independiente \( X \) (por ejemplo, el gasto en publicidad) y una variable dependiente \( Y \) (por ejemplo, las ventas), mediante la ecuación:

Y = β₀ + β₁X + ε

Donde:

- \( \beta_0 \) es el intercepto (ordenada al origen)
- \( \beta_1 \) es la pendiente de la línea de regresión
- \( \varepsilon \) representa el error residual

El modelo busca encontrar los coeficientes \( \beta_0 \) y \( \beta_1 \) que minimicen el error cuadrático medio entre los valores reales y los predichos.

---

### 📁 Contenido del repositorio

- `regresion_lineal.py`: Script principal que incluye:
  - Simulación de datos
  - División en conjunto de entrenamiento y prueba
  - Entrenamiento del modelo
  - Predicción y evaluación del rendimiento con R²
  - Visualización con `matplotlib`

---

### 🛠️ Requisitos del entorno

Asegúrate de tener instalado Python 3.7 o superior. Luego instala las dependencias necesarias:

```bash
pip install numpy matplotlib scikit-learn
