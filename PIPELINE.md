# Pipeline propuesto (baseline basado en los papers)

> Basado en los resúmenes de los papers de los integrantes 1, 2 y 3 (el del integrante 4 aún no está en el repo).
> Solo se usan modelos vistos en clase: regresión lineal/logística, KNN, Random Forest, SVM/SVR, MLP, LSTM y Transformers.

## 1. Features del dataset final

El dataset se genera con `src/preprocessing.py`, que crea `data/processed/panel_mensual_distrital.csv`. Las cifras vienen de las salidas de `notebooks/01_eda.ipynb`.

**Unidad:** distrito × mes. Son 435 distritos con al menos 50 casos acumulados, de 2000-01 a 2024-12. Hay 130 500 filas, que bajan a 125 280 al quitar el año 2000 (no tiene `lag12`). Entrenamiento hasta 2018 y prueba de 2019 a 2024. Los meses en alerta son el 4,69 % en entrenamiento y el 3,43 % en prueba.

| Grupo | Columnas |
|---|---|
| Identificadores | `ubigeo`, `departamento`, `provincia`, `distrito`, `fecha_mes`, `es_train` |
| **Features (7)** | `lag1`, `lag2`, `lag3`, `lag12`, `media3`, `mes_sin`, `mes_cos` |
| Objetivos | `casos` (regresión), `alerta` (clasificación). `umbral` es auxiliar y no entra al modelo |

### Clima (`data/clima/`, todavía sin integrar)

Son 8 estaciones SENAMHI de Cusco con datos diarios de año, mes, día, precipitación, temperatura máxima y mínima. El valor `-99.9` marca un faltante. Hay dos problemas:

- **Las series terminan en 2014** (Granja Kcayra llega a octubre de 2015). Solo se solapan con los casos entre 2000 y 2014, y no cubren nada del periodo de prueba 2019–2024.
- **Las estaciones están en distritos andinos que casi no tienen casos:** Pisac, Acomayo, Pomacanchi, Colquepata, Caicay y Ccatca tienen 0 y Paruro tiene 1. Los 18 149 casos de Cusco están en la selva alta: Echarate (5 925), Quellouno (2 224), Yanatile (1 950), Ocobamba (1 396), Santa Ana (1 260) y Kosñipata (1 011).

Por eso estas estaciones solo sirven como aproximación regional. Los papers 1 y 3 usan o recomiendan ERA5 para cubrir estos vacíos.

## 2. Pasos del pipeline

**Paso 1. Alcance.** Se trabaja con la serie mensual distrito × mes de distritos con al menos 50 casos. Si el grupo se enfoca en Cusco, se filtra ese departamento.

**Paso 2. Limpieza y agregación.** Se usa lo que ya está en `preprocessing.py`: ubigeo a 6 dígitos, edad en años y agregación a mes. No se recortan los picos porque son los brotes que queremos predecir (el paper 3 hace lo mismo con las semanas).

**Paso 3. Features de casos.** Rezagos 1, 2, 3 y 12 más `media3`, porque el paper 1 usa rezagos 1–12, el paper 2 usa 1–3 y el paper 3 usa una media móvil. A esto se suma la codificación seno/coseno del mes (paper 3). Ya están implementados.

**Paso 4. Features de clima (opcional, para la segunda versión).**
- Datos diarios a mensuales: promedio para temperatura y suma para lluvia (paper 3).
- Rezagos climáticos de 1 a 6 meses (paper 1).
- Faltantes rellenados con el mismo mes del año anterior (paper 2).
- Opcionalmente, el ONI (fenómeno El Niño), que en el paper 1 fue el predictor dominante.
- Se compara la versión con clima contra la versión sin clima.

**Paso 5. Objetivos.**
- Regresión: casos del mes t+h. h = 1 es el caso principal y h = 3 replica el paper 1. Para h = 3 solo se pueden usar rezagos de 3 meses o más, para no filtrar información.
- Clasificación: `alerta`, con media + 2 desviaciones como definición actual. Como sensibilidad se prueba el percentil 75 del distrito (paper 3).

**Paso 6. Partición temporal, nunca aleatoria.** Entrenamiento hasta 2018 y prueba 2019–2024. Los hiperparámetros se ajustan con `TimeSeriesSplit` o con ventana deslizante de 24 meses de entrenamiento y 3 de validación (paper 1). Los escaladores y SMOTE se ajustan solo con entrenamiento.

**Paso 7. Modelos.**

| # | Modelo | Papel | Según los papers |
|---|---|---|---|
| B1 | Naïve: `lag1` | Línea base | Necesario para medir el MASE (paper 3) |
| B2 | Naïve estacional: `lag12` | Línea base | Paper 3; el EDA muestra autocorrelación de 0,79 en el rezago 12 |
| M1 | Regresión lineal / logística sobre los rezagos | Reemplaza al ARIMA | En el paper 3 el ARIMA fue el modelo más exacto |
| M2 | **Random Forest** | Modelo principal | Paper 3; reemplaza al XGBoost de los papers 1 y 3. Búsqueda: 100–500 árboles, profundidad 10–40, `min_samples_split` 5–30 |
| M3 | KNN | Comparación | Visto en el curso; ningún paper lo usa |
| M4 | SVR / SVM | Comparación | Paper 2: mejor RMSE en Fortaleza |
| M5 | MLP de 1 y 2 capas (SFNN / DFNN) | Red neuronal | Paper 2 |
| M6 | LSTM con ventana de 12 meses | Secuencial | Papers 2 y 3 |
| M7 | Transformer | Exploratorio | Ningún paper lo usa. Debe ser un modelo global con todos los distritos juntos, no uno por distrito |

**Paso 8. Métricas.**
- Regresión: RMSE, MAE y MASE frente al naïve estacional (papers 2 y 3).
- Alerta: priorizar el recall, más precisión, F1, AUC-ROC (papers 1 y 3) y PR-AUC por el desbalance de clases.
- Gráfico de predicho vs. observado: el paper 2 muestra que un RMSE bajo puede esconder un mal ajuste.
- Opcional: prueba de Diebold-Mariano entre modelos (paper 3).

**Paso 9. Interpretabilidad.** Importancia por permutación o SHAP sobre el Random Forest (papers 1 y 3).

**Paso 10. Salida operativa (opcional).** Convertir la probabilidad de alerta en un semáforo de 4 niveles: menos de 25 %, 25–50 %, 50–75 % y más de 75 % (paper 3).

## 3. Qué esperar según los papers

Los modelos de árboles y los lineales suelen ganar. En el paper 3 el LSTM fracasó con 313 puntos de serie, y nuestras series tienen unos 300 meses por distrito. El LSTM y el Transformer solo tienen alguna posibilidad si se entrenan con todos los distritos juntos, y deben presentarse como comparación, no como modelo principal.
