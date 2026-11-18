# Comparación climática aislada

## Alcance y reproducción

Este experimento añade `src/experimento_clima.py`; no cambia `preprocessing.py`, `integracion_clima.py`, los notebooks 03/04 ni sus métricas guardadas. Los nuevos resultados están en `results/experimento_clima/`.

```powershell
uv --cache-dir .cache/uv venv .venv
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe -r requirements-experimento.txt
.venv/Scripts/python.exe -u src/experimento_clima.py
.venv/Scripts/python.exe src/reporte_experimento_clima.py
```

Para regenerar solo datos: `--datasets-only`. Para repetir modelos con los mismos CSV: `--reuse-datasets`. Las dependencias efectivamente utilizadas se registran en `ejecucion.json`.

Pruebas de presupuesto, conservación de observaciones, causalidad, consistencia térmica y respaldo sin clima: `.venv/Scripts/python.exe -m unittest discover -s tests -v`.

## Tres versiones

- [Estacional 52 %](DATASET_ESTACIONAL_52.md).
- [Sin imputación](DATASET_SIN_IMPUTAR.md).
- [Huecos cortos y anomalías](DATASET_HUECOS_CORTOS.md), propuesta propia.

Se agrega una cuarta condición de control, `solo_casos`, sin variables climáticas ni imputación.

El 52 % se interpreta como fracción de **celdas originalmente faltantes**, no de todas las celdas ni de filas. No se presenta como un porcentaje científicamente seguro: Donizette et al. reportan 52,08 % de humedad faltante en una ciudad; no establecen un umbral universal ni prueban que reconstruir años enteros sea válido.

## Dos evaluaciones independientes

| Protocolo | Ajuste inicial | Validación | Prueba | Finalidad |
|---|---|---|---|---|
| Histórico | 2001–2009 | 2010–2011 | 2012–2014 | Medir el aporte de estaciones dentro de su cobertura |
| Oficial | 2001–2015 | 2016–2018 | 2019–2024 | Conservar las fechas de prueba del proyecto |

Los distritos tienen al menos 50 casos antes de terminar el ajuste inicial. Tanto la asignación de estaciones como su aptitud para temperatura se determinan con datos disponibles hasta ese corte. Las coordenadas son información geográfica estática.

Las tres variantes y el control se puntúan sobre las mismas filas de prueba dentro de cada protocolo. Los dos protocolos pueden tener diferente cantidad de distritos: no se comparan directamente sus errores como si fueran el mismo conjunto.

El calendario se agrega mensualmente como en el proyecto. Los rezagos de casos y clima usan meses observados anteriores al objetivo; se evalúa predicción de un paso, no una trayectoria recursiva desde el final del entrenamiento.

## Modelos y selección

Se prueban cinco familias en **regresión y clasificación**, con dos configuraciones por familia:

- Ridge / regresión logística: `alpha=1,10` o `C=0.1,1`.
- Árbol de decisión: profundidad 3 o 5 y mínimo 20 por hoja.
- Random Forest: 100 árboles, profundidad 8 y mínimo 5 o 20 por hoja.
- MLP: una capa de 32 o dos capas de 32/16, regularización 0,01 y 150 iteraciones, sin validación interna aleatoria.
- CNN temporal: convolución sobre los tres meses previos, 8 o 16 filtros, contexto de rezago anual/media trimestral/calendario, 50 épocas y lotes de 512.

La CNN usa canales de casos y, en su rama climática, lluvia/tmax/tmin, ordenados de pasado a presente. No se aplica una convolución arbitraria sobre columnas tabulares. Es una arquitectura pequeña exploratoria; no un ajuste exhaustivo.

Las entradas tabulares conservan las siete variables originales y añaden nueve rezagos climáticos: tres meses para cada variable primaria. Ridge, logística, MLP y CNN ajustan su escalamiento solo con entrenamiento. Los árboles no se escalan. La clasificación utiliza pesos balanceados; no se aplica SMOTE.

Los rezagos climáticos se recalculan dentro del panel publicado que comienza en 2001. Sus primeros tres meses pueden quedar incompletos y, por tanto, fuera del ajuste de la rama climática; esto se aplica a las tres versiones por igual. Los antecedentes de estaciones anteriores al panel se usan para imputación estacional, no para ampliar filas de entrenamiento.

El objetivo `alerta` se recalcula como casos superiores a media más dos desviaciones distritales, usando solo el tramo que entrena el modelo: hasta 2009/2015 durante selección y hasta 2011/2018 durante el ajuste final. El umbral probabilístico maximiza F1 de validación; permanece fijo al evaluar prueba. Cambiar el tramo de referencia también cambia la definición de alerta entre validación y prueba: ambas se informan según su historia disponible.

Para aislar el contraste climático se conserva el conjunto **base** de siete variables en todas las familias. Esta comparación no añade los cinco conteos relativos al umbral de la variante alternativa del notebook 04; los resultados de alerta no representan una optimización exhaustiva de aquella variante.

La configuración de regresión se selecciona por menor RMSE de validación y la de clasificación por mayor `average_precision_score` (rotulado PR-AUC). Después se reentrena con ajuste más validación y se evalúa una sola vez en prueba.

Los empates exactos conservan el orden de evaluación: primero el control de casos, después sin imputación, estacional 52 % y huecos cortos. No se atribuye una mejora climática a un empate que en realidad utiliza el mismo respaldo de casos.

## Filas sin clima y comparación justa

**No hay un `SimpleImputer` silencioso.** La rama climática de cada familia se entrena exclusivamente con filas cuyas nueve entradas climáticas están disponibles en esa versión. En una fila de prueba con alguna entrada ausente se utiliza un modelo de respaldo de la misma familia entrenado solo con casos.

Si no hay al menos 100 filas completas de entrenamiento, o no hay ambas clases para clasificación, se utiliza el respaldo. Se informa `climate_fraction`: fracción de predicciones efectivamente producidas por la rama climática. Esto permite comparar todas las filas sin descartar selectivamente meses difíciles. La comparación mide el sistema combinado y su cobertura, no únicamente un modelo climático.

En el periodo oficial no hay nuevas mediciones de estaciones; una cobertura positiva de `estacional_52` puede contener estimaciones propagadas, no evidencia de observaciones recientes.

## Salidas y limitaciones

Los datasets y auditorías se generan bajo `data/processed/experimento_clima/{historico,oficial}/`, excluidos de Git por su tamaño y porque son reproducibles. `resumen_imputacion.csv` informa el presupuesto de imputación y su cumplimiento.

En los CSV, `es_train` identifica el ajuste **inicial** de cada protocolo; los cortes de validación y prueba se reconstruyen desde `fecha_mes`. `casos` permanece como objetivo de regresión; `umbral` y `alerta` se omiten para evitar publicar una etiqueta calculada con un corte equivocado. El script los reconstruye con el entrenamiento que corresponda.

`metricas.csv` guarda resultados de prueba; `validacion.csv`, todos los candidatos; `seleccion_por_validacion.csv`, los ganadores sin elegir por prueba; `predicciones.csv`, predicciones por distrito/mes y uso del clima; `ejecucion.json`, versiones y hash del CSV epidemiológico.

El segundo comando genera [el informe de resultados](../results/experimento_clima/RESULTADOS.md) a partir de los CSV, sin reentrenar. `diagnostico_ajuste.csv` informa si los ajustes MLP alcanzaron su máximo de iteraciones.

MASE usa como escala el MAE estacional del entrenamiento final. Clasificación informa precisión, recall, F1, average precision, AUC-ROC y falsas alarmas. Las referencias de último mes, año anterior y cero se evalúan sobre las mismas filas de regresión.

Se utiliza una semilla fija y una búsqueda pequeña, sin intervalos de confianza ni validación externa: los resultados son exploratorios. MLP está limitada a 150 iteraciones y puede no converger; CNN a 50 épocas. El mejor resultado puntual no demuestra superioridad estadística ni validez clínica. El periodo histórico no valida desempeño climático posterior a 2015. Este experimento no descarga ERA5 ni incorpora datos climáticos externos.

