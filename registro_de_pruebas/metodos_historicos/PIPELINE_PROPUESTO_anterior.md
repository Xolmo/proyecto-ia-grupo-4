# Pipeline: implementación actual y propuestas pendientes

Este documento reemplaza la descripción que mezclaba el diseño propuesto con el flujo ejecutado. La fuente de verdad son `src/preprocessing.py`, `notebooks/03_baseline.ipynb` y `notebooks/04_arbol_decision.ipynb`.

**Extensión separada:** `src/experimento_clima.py` implementa ahora tres variantes climáticas y cinco familias de modelos. Su guía es [docs/EXPERIMENTO_CLIMA.md](docs/EXPERIMENTO_CLIMA.md). Las secciones siguientes describen el flujo original de los notebooks; los pendientes de esa ruta no implican ausencia de la comparación exploratoria nueva.

## 1. Objetivo y alcance implementados

Estimar casos mensuales de leishmaniosis cutánea y clasificar meses de alerta por distrito de Cajamarca. Los modelos evaluados utilizan exclusivamente casos históricos y calendario, sin variables climáticas.

## 2. Flujo implementado

### Carga y limpieza

`clean_cases(load_raw(RAW))` normaliza ubigeos a seis dígitos, convierte edades a años, reemplaza edades mayores de 110 por la mediana, completa diresa por moda distrital y localidad con `SIN DATO`. Elimina columnas innecesarias y aproxima fechas desde año y semana. No elimina duplicados ni recorta brotes.

La imputación de edad se realiza antes de la partición, pero esa columna no entra en los modelos actuales.

### Panel de Cajamarca

Los notebooks filtran `departamento == "CAJAMARCA"` antes de llamar a `build_monthly_panel`. Se retienen distritos con al menos 50 casos acumulados y se construye un calendario distrito × mes con ceros donde no hay registros.

El filtro de distritos usa todo el historial, incluida la prueba. El comando independiente `python src/preprocessing.py` genera un panel nacional; no debe confundirse con el panel filtrado de los notebooks.

### Variables y objetivos

- Entradas base: `lag1`, `lag2`, `lag3`, `lag12`, `media3`, `mes_sin`, `mes_cos`.
- `media3` usa los tres meses anteriores, mediante desplazamiento previo a la media móvil.
- Se eliminan filas sin las siete entradas completas.
- Regresión: casos del mes de la fila, usando historia anterior.
- Clasificación: casos superiores a media más dos desviaciones del distrito, calculadas en entrenamiento hasta 2018.
- Variante relativa: añade `lag1_rel`, `lag2_rel`, `lag3_rel`, `lag12_rel` y `media3_rel`, dividiendo los conteos entre el umbral distrital. En esta variante el umbral sí participa indirectamente como entrada.

### Partición y validación

Entrenamiento hasta 2018 y prueba 2019–2024. El ajuste de árboles emplea cinco pliegues de ventana creciente: cada año de 2014 a 2018 se valida con los años anteriores como entrenamiento.

**Limitación:** el umbral que define las alertas y las variables relativas no se recalcula dentro de cada pliegue; usa todo el entrenamiento hasta 2018. Para una validación interna estricta debe recalcularse con el tramo de entrenamiento de cada pliegue.

### Líneas base

`03_baseline.ipynb` evalúa último valor (`lag1`) y mismo mes del año anterior (`lag12`), tanto para casos como para alerta. Añade referencias de media distrital de entrenamiento y predicción constante cero para regresión.

### Árboles de decisión

`04_arbol_decision.ipynb` compara profundidades 3, 4, 5, 6, 8 y 10, y mínimos de 20, 50, 100 y 200 observaciones por hoja.

- Regresión: selección por RMSE medio de validación. El resultado guardado corresponde a profundidad 5 y mínimo 20 por hoja.
- Clasificación: variantes base y relativa, con `class_weight="balanced"`; selección por promedio de `average_precision_score`, denominado PR-AUC en los resultados.
- El umbral de probabilidad se elige maximizando F1 sobre predicciones fuera de pliegue del entrenamiento.
- Los modelos seleccionados se ajustan con todo el entrenamiento y se evalúan en prueba.
- No se aplican escalamiento ni SMOTE. Existen funciones auxiliares para otros modelos, pero no forman parte del entrenamiento actual.

Las predicciones usan rezagos observados dentro de la prueba; representan un paso temporal, no una trayectoria recursiva de seis años generada desde 2018.

### Evaluación e interpretación

Regresión: RMSE, MAE y MASE; su denominador es el error absoluto medio estacional del entrenamiento. Clasificación: precisión, recall, F1, average precision y AUC-ROC.

Se generan curvas, matriz de confusión, importancia por permutación en prueba y reglas de un árbol ilustrativo de profundidad 3. Este árbol ilustrativo no es el modelo principal evaluado.

Resultados: `results/baseline_metricas.csv`, `results/arbol_metricas.csv`, `results/arbol_reglas.txt` y gráficos en `results/plots/`.

## 3. Rama climática auxiliar, fuera del modelado

`src/integracion_clima.py` implementa una unión de casos y clima. Depura registros diarios, exige 80 % de días válidos por variable, promedia temperaturas y estima precipitación mensual como promedio diario por días del mes.

Asocia localidades con coordenadas y estaciones cercanas, distingue estaciones para lluvia y temperatura y asigna al distrito la estación que concentra más casos de entrenamiento. Crea rezagos climáticos de 1, 2 y 3 meses y exporta `dataset_cajamarca.csv`, entre otros archivos.

**Los notebooks 03 y 04 no leen esa salida.** Reconstruyen el panel desde los registros epidemiológicos. Según sus notas metodológicas, la cobertura climática no alcanza la prueba 2019–2024; por eso el clima se excluyó de los modelos reportados.

No están implementados en esta rama los rezagos climáticos de 1–6 meses, la imputación por el mismo mes del año anterior ni ONI que figuraban en la propuesta inicial.

## 4. Resultados actuales

En prueba, el árbol de regresión obtiene RMSE 0,907 frente a 1,142 del último mes, pero MAE 0,576 frente a 0,537. La referencia constante cero obtiene MAE 0,413 y RMSE 1,054.

La clasificación relativa obtiene recall 0,301, F1 0,222 y average precision 0,155: detecta 55 de 183 alertas y genera 258 falsas alarmas. La variante base obtiene average precision 0,103. No se evaluó una mejora atribuible al clima.

## 5. Propuestas pendientes, no ejecutadas en el flujo actual

- Regresión lineal/logística, KNN, MLP de una y dos capas, LSTM y Transformer.
- Comparación con clima, condicionada a conseguir cobertura compatible con validación y prueba.
- Sensibilidad completa con y sin semana 1: existe `--exclude-week1`, pero no una comparación final de modelos documentada en el notebook 04.
- Sensibilidad con alerta por percentil 75 y definición operativa del umbral.
- Pronósticos de tres meses, que requieren definir entradas disponibles y evaluación específica.
- Semáforo de riesgo, todavía opcional y sin implementación actual.

## 6. Evidencia bibliográfica y su alcance

Los artículos y resúmenes están en `papers/`: Adams et al., Donizette et al., Araque y Paipa, y El Alaoui et al. Motivaron la comparación con referencias simples, los rezagos, la validación temporal y las propuestas de redes y clima.

ARIMA/SARIMA, Random Forest, SVR y XGBoost pertenecen a las referencias o extensiones discutidas; no deben presentarse como modelos entrenados en este repositorio.
