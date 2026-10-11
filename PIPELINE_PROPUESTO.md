## 1. Problema

Anticipar los casos mensuales de leishmaniosis cutánea por distrito del departamento de Cajamarca, Perú (regresión) y detectar los meses de alerta (clasificación), usando el historial de casos del CDC-MINSA 2000–2024 y, cuando haya cobertura, el clima de las estaciones de SENAMHI en Cajamarca.

**Alcance geográfico.** El análisis se limita a Cajamarca por disponibilidad de datos: es la región donde se pudo asociar el clima de SENAMHI a los distritos con casos. Los resultados aplican a Cajamarca y no deben extrapolarse al resto del país.

## 2. Evidencia de la literatura

| Paper | Problema | Modelos | Qué aporta a nuestro proyecto |
|---|---|---|---|
| **Integrante 1** — Adams et al. (2025), *Environ. Res.: Health* | Leishmaniosis visceral en 113 municipios de Brasil | XGBoost en ventana deslizante (entrena 24 meses, valida 3); regresión de conteos + clasificación binaria; SHAP | Ventana deslizante, lags 1–12 sin fuga, umbral operativo definido por la autoridad sanitaria, doble enfoque regresión/clasificación, ONI como predictor |
| **Integrante 2** — Donizette, Rocco & de Queiroz (2025), *Oper. Res. Health Care* | Leishmaniosis visceral y tegumentaria en 6 ciudades de Brasil | SFNN y DFNN (MLP), LSTM, SVR; RMSE como métrica; también probaron RF, GB, XGBoost y LightGBM (peores) | Comparación MLP / LSTM, lags 1–3 de casos como entradas, faltantes del clima con el mismo mes del año anterior, evaluar el ajuste gráfico además del RMSE |
| **Integrante 3** — Araque & Paipa (2026), tesis de maestría, Univ. Santo Tomás | Brotes de dengue en Bucaramanga (Colombia): 10 259 casos semanales, 17 comunas, 2015–2020, con clima del IDEAM | ARIMA (línea base estadística), Random Forest, XGBoost, LSTM y naïve estacional; brote = percentil 75; horizontes de 1 a 4 semanas; métricas MASE, F1 y AUC; SHAP | Mismo diseño de datos (casos + clima rezagado). ARIMA gana en exactitud y XGBoost se elige por explicabilidad; el LSTM falla con 313 puntos. Aporta naïve estacional como referencia, MASE, brote como percentil 75 y semáforo de riesgo |
| **Integrante 4** — El Alaoui et al. (2026), *Parasitol. Int.* | Leishmaniosis cutánea en 29 municipios de Errachidia (Marruecos), 2010–2025 | SARIMA (0,1,0)(1,1,0) con s = 12, elegido por AIC y validado con Ljung-Box; RMSE = 12,35 y MAE = 10,15 en prueba (80/20) | SARIMA como línea base simple e interpretable; el clima se relaciona con los casos (temperatura r = −0,49, humedad r = +0,47, lluvia sin relación) |

**Patrón común:** los modelos complejos no superan de forma clara a los simples. En el paper 3 el ARIMA gana y el LSTM falla; en el paper 2 el mejor modelo cambia por ciudad. Por eso importa comparar siempre contra baselines fuertes.

## 3. Datos y variables

**Unidad:** distrito × mes de Cajamarca, construida con `src/preprocessing.py` y filtrada a Cajamarca en `src/integracion_clima.py` (`data/processed/dataset_cajamarca.csv`). Dentro de Cajamarca se conservan los distritos con un mínimo de casos acumulados; el umbral actual es 50 (`UMBRAL_CASOS`) y la cantidad de distritos resultante depende de ese valor.

| Grupo | Columnas |
|---|---|
| Identificadores | `ubigeo`, `departamento`, `provincia`, `distrito`, `fecha_mes`, `es_train` |
| **Variables de casos (7)** | `lag1`, `lag2`, `lag3`, `lag12`, `media3`, `mes_sin`, `mes_cos` |
| Objetivos | `casos` (regresión), `alerta` (clasificación). `umbral` es auxiliar y no entra al modelo |

**Clima (`data/clima/`):** 26 estaciones SENAMHI de Cajamarca con datos diarios de precipitación y temperatura máxima/mínima (`-99.9` marca un faltante). La carpeta incluye además 8 estaciones de Cusco, que no se usan (ver abajo).

- `src/integracion_clima.py` asigna a cada distrito de Cajamarca la estación más cercana y genera `dataset_cajamarca.csv`. Solo parte de las filas queda con clima, por lo que hay que decidir cómo tratar el resto.
- Las estaciones de Cusco quedan fuera porque Cusco no forma parte del alcance. Además, sus series terminan en 2014 y están en distritos andinos con casi ningún caso (los casos están en la selva alta), así que no cubren el periodo de prueba.
- Alternativa para cubrir los vacíos de clima en Cajamarca: datos reanalizados como ERA5 (recomendado en los papers 1 y 3).

## 4. Pasos del pipeline

**Paso 1. Alcance geográfico.** Se modela solo el departamento de Cajamarca. Falta fijar el mínimo de casos acumulados por distrito dentro de Cajamarca.

**Paso 2. Limpieza y agregación.** Lo que ya hace `preprocessing.py`: `ubigeo` a 6 dígitos, edad en años y agregación a mes. No se recortan los picos, porque son los brotes que se quiere predecir.

**Paso 3. Variables de casos.** Rezagos 1, 2, 3 y 12, media de 3 meses y mes en seno/coseno (papers 1, 2 y 3). Ya implementado.

**Paso 4. Variables de clima (segunda versión).**
- Datos diarios a mensuales: promedio para temperatura y suma para lluvia (paper 3).
- Rezagos climáticos de 1 a 6 meses (paper 1).
- Faltantes rellenados con el mismo mes del año anterior (paper 2).
- Opcional: ONI (fenómeno El Niño), predictor dominante en el paper 1.
- Se compara la versión con clima contra la versión sin clima.

**Paso 5. Objetivos.**
- Regresión: casos del mes siguiente (h = 1) como caso principal. h = 3 es opcional y solo con rezagos de 3 meses o más, para no filtrar información.
- Clasificación: `alerta`, con media + 2 desviaciones como definición provisional. Como sensibilidad, el percentil 75 del distrito (paper 3).

**Paso 6. Partición temporal, nunca aleatoria.** Entrenamiento hasta 2018 y prueba 2019–2024. Los hiperparámetros se ajustan con `TimeSeriesSplit` o ventana deslizante (paper 1). Escaladores y SMOTE se ajustan solo con entrenamiento.

**Paso 7. Modelos.**

| # | Modelo | Tarea | Papel | Según los papers | Técnica del curso |
|---|---|---|---|---|---|
| B1 | Último valor observado (`lag1`) | Regresión y alerta | Línea base | Naïve del paper 3 | — |
| B2 | Mismo mes del año anterior (`lag12`) | Regresión y alerta | Línea base | Naïve estacional del paper 3; el EDA muestra autocorrelación de 0,79 en el rezago 12 | — |
| M1 | Regresión lineal / logística sobre los rezagos | Regresión / alerta | Modelo estadístico de referencia, en lugar de ARIMA/SARIMA | Papers 3 y 4 usan ARIMA/SARIMA como referencia | Regresión lineal, regresión logística |
| M2 | Árbol de decisión | Regresión / alerta | Reglas interpretables | Paper 3 (modelos de árboles) | Árboles de decisión |
| M3 | KNN | Regresión / alerta | Comparación no paramétrica | Ningún paper lo usa | KNN |
| M4 | MLP de 1 y 2 capas | Regresión / alerta | Red neuronal principal | Paper 2 (SFNN y DFNN) | MLP |
| M5 | LSTM con ventana de 12 meses | Regresión | Modelo secuencial | Papers 2 y 3 | RNN |
| M6 | Transformer | Regresión | Exploratorio; entrenado con todos los distritos de Cajamarca a la vez | Ningún paper lo usa | Transformers |

**Paso 8. Métricas.**
- Regresión: RMSE, MAE y MASE frente al naïve estacional (papers 2 y 3).
- Alerta: recall, precisión, F1 y PR-AUC por el desbalance (no accuracy); AUC-ROC como complemento (papers 1 y 3).
- Gráfico de predicho vs. observado, porque el paper 2 muestra que un RMSE bajo puede esconder un mal ajuste.

**Paso 9. Interpretabilidad.** Importancia por permutación sobre el mejor modelo y, para el árbol de decisión, lectura directa de las reglas.

**Paso 10. Sensibilidad.** Repetir con y sin la semana 1 (`--exclude-week1`), con y sin clima, y con la alerta alternativa del percentil 75.

**Paso 11 (opcional). Salida operativa.** Convertir la probabilidad de alerta en un semáforo de 4 niveles: menos de 25 %, 25–50 %, 50–75 % y más de 75 % (paper 3).

## 5. Fuera del alcance por ahora

Otras técnicas no vistas en clase.

| Modelo | Aparece en | Nota |
|---|---|---|
| Random Forest | Paper 3 (y 2, como alternativa) | Conjunto de árboles de decisión; podría considerarse una extensión de M2 |
| SVR / SVM | Paper 2 | Mejor RMSE en Fortaleza |
| XGBoost | Papers 1 y 3 | Elegido por el paper 3 por su explicabilidad; permitiría usar SHAP |
| ARIMA / SARIMA | Papers 3 y 4 | Línea base estadística estándar; M1 la sustituye por ahora |

## 6. Qué esperar

Según los papers, lo razonable es que los baselines estacionales y los modelos lineales sean difíciles de superar. El LSTM y el Transformer solo tienen opciones si se entrenan con varios distritos de Cajamarca a la vez; con una serie de unos 300 meses por distrito, el LSTM falló en el paper 3. Conviene presentarlos como comparación y no como modelo principal.

## 7. Decisiones abiertas

1. Mínimo de casos acumulados por distrito dentro de Cajamarca (el alcance geográfico ya está decidido).
2. Definición final de alerta: la actual (media + 2 desviaciones) es provisional; alternativas: percentil 75 del distrito (paper 3) o un umbral operativo de la autoridad sanitaria (paper 1).
3. Qué hacer con los meses sin clima: descartarlos, imputarlos o modelar también sin clima.
4. Confirmar con la profesora los modelos de la sección 5.
