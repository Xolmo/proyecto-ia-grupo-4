# Predicción de leishmaniosis cutánea en Cajamarca

Proyecto del Grupo 4 para estimar casos mensuales por distrito y detectar meses de alerta con registros CDC-MINSA de 2000–2024.

**Flujo original implementado (notebooks 03/04):** preprocesamiento, exploración epidemiológica y climática, líneas base y árboles de decisión de regresión y clasificación. Estos notebooks usan exclusivamente variables derivadas de casos y calendario; **no utilizan clima**.

**Extensión independiente:** `src/experimento_clima.py` compara ahora tres variantes climáticas con regresión/logística, árbol, Random Forest, MLP y CNN temporal. Sus resultados y metodología se documentan en [la guía del experimento](docs/EXPERIMENTO_CLIMA.md), sin reemplazar el flujo original.

## Integrantes

- Dhanny Sebastian Ramirez Rivera
- Sergio Estéfano Huamán Noriega
- Leonardo Jaime Flores Vera
- Vilcapoma Madueño Franz Javier

Curso: Inteligencia Artificial Aplicada · Docente: Galareta Asian, Ana Paula.

## Datos y alcance

- CDC-MINSA: 156 819 registros individuales de leishmaniosis cutánea, 2000–2024, de 23 departamentos. El modelado filtra Cajamarca.
- SENAMHI: series diarias de 26 estaciones de Cajamarca y 8 de Cusco. No entran al modelado de los notebooks 03/04 porque no cubren la prueba 2019–2024; la extensión independiente evalúa explícitamente sus faltantes e imputaciones.
- INEI y coordenadas de estaciones: sirven para la integración climática auxiliar. Fuentes en `data/raw/external/FUENTES.md`.

La información epidemiológica es provisional. La ubicación corresponde al probable sitio de infección; los resultados no se extrapolan a otras regiones.

## Flujo ejecutado por los notebooks de modelado

1. Cargar el CSV epidemiológico y limpiar registros con `src/preprocessing.py`.
2. Filtrar Cajamarca y conservar distritos con al menos 50 casos acumulados.
3. Agregar por distrito y mes, completando meses sin registros con cero.
4. Crear `lag1`, `lag2`, `lag3`, `lag12`, `media3`, `mes_sin` y `mes_cos`; eliminar filas sin historia completa.
5. Definir `casos` como objetivo de regresión y `alerta` como casos superiores a media más dos desviaciones del distrito, calculadas con entrenamiento.
6. Separar entrenamiento hasta 2018 y prueba 2019–2024.
7. Evaluar líneas base y entrenar árboles con validación temporal creciente en 2014–2018.
8. Exportar métricas, gráficos, importancia por permutación y reglas.

`03_baseline.ipynb` y `04_arbol_decision.ipynb` reconstruyen el panel desde el CSV de casos. No leen `dataset_cajamarca.csv` ni llaman a `integracion_clima.py`.

## Archivos principales

| Archivo | Función real |
|---|---|
| `src/preprocessing.py` | Limpieza, panel, variables, alertas y funciones auxiliares de escalamiento y desbalance |
| `notebooks/01_eda.ipynb` | Exploración epidemiológica |
| `notebooks/02_eda_clima.ipynb` | Exploración del clima y su cobertura |
| `notebooks/03_baseline.ipynb` | Evaluación de último mes y mismo mes del año anterior |
| `notebooks/04_arbol_decision.ipynb` | Ajuste, evaluación e interpretación de árboles |
| `src/integracion_clima.py` | Integración climática; no alimenta los notebooks 03/04; la extensión reutiliza sus funciones |
| `results/baseline_metricas.csv` | Resultados de líneas base |
| `results/arbol_metricas.csv` | Comparación de árboles y referencias |
| `results/arbol_reglas.txt` | Reglas de un árbol ilustrativo de tres niveles |
| `PIPELINE_PROPUESTO.md` | Detalle del flujo implementado y propuestas pendientes |

## Reproducción

Desde la raíz del repositorio, instalar `requirements.txt` y ejecutar los notebooks `01` a `04` en orden, con `notebooks/` como directorio de trabajo para sus rutas relativas.

```bash
pip install -r requirements.txt
python src/preprocessing.py
```

El comando anterior genera un panel nacional; los notebooks de modelado filtran Cajamarca antes de construir su panel. `python src/preprocessing.py --exclude-week1` genera una variante de sensibilidad, pero no ejecuta una comparación de modelos.

```bash
# Rama auxiliar opcional; no es un requisito del entrenamiento actual.
python src/integracion_clima.py
```

Esta rama genera clima diario, clima mensual, correspondencias de localidades y `data/processed/dataset_cajamarca.csv`. Su existencia no implica que los modelos lo consuman.

## Modelos y resultados guardados

Los árboles utilizan las siete variables históricas y temporales. La clasificación compara una variante base con otra que añade cinco conteos relativos al umbral distrital. Usa `class_weight="balanced"`; no aplica escalamiento ni SMOTE.

| Prueba 2019–2024 | Último mes | Año anterior | Árbol |
|---|---:|---:|---:|
| RMSE | 1,142 | 1,222 | 0,907 |
| MAE | 0,537 | 0,603 | 0,576 |
| F1 de alerta | 0,213 | 0,124 | 0,222 (relativa) |
| Recall de alerta | 0,213 | 0,126 | 0,301 (relativa) |
| PR-AUC de alerta | 0,153 | 0,090 | 0,155 (relativa) |

El árbol mejora RMSE, pero no MAE frente al último mes. La variante relativa detecta 55 de 183 alertas, con 258 falsas alarmas. Estos resultados no demuestran una mejora por incorporar clima.

## Limitaciones y pendientes

- El filtro de 50 casos acumulados utiliza todo el historial, incluida la prueba; no es una selección basada exclusivamente en entrenamiento.
- La alerta es una definición experimental, no una norma sanitaria.
- En la validación interna, umbrales de alerta y variables relativas se calculan con todo el entrenamiento hasta 2018, sin recalcularlos dentro de cada pliegue. Esto introduce información posterior al corte de algunos pliegues y debe corregirse para una validación estricta.
- Las métricas corresponden a predicción de un paso con rezagos observados; no a pronósticos recursivos de todo 2019–2024 desde 2018.
- En los notebooks 03/04 siguen pendientes las demás familias. La extensión independiente ya compara regresión/logística, MLP, CNN y Random Forest; KNN, LSTM y Transformer no se implementan en esa comparación.
- Quedan comparaciones de sensibilidad, una eventual versión climática con cobertura suficiente y una salida operativa.

Los cuatro papers y sus resúmenes están en `papers/`. Fundamentan propuestas metodológicas; no acreditan implementaciones adicionales.

## Experimento adicional de clima

**Ingeniería climática adicional:** ocho representaciones (control, anomalías, multiescala, retardos no lineales, interacciones, extremos diarios, selección y PCA), con tres fuentes y los cinco modelos. Ver [metodología y fuentes](docs/INGENIERIA_CLIMA.md). No reemplaza V1/V2.

**V2 disponible:** incorpora distrito, conteos relativos, clima resumido por distrito/mes, selección de ventanas de cinco/diez años y parada temporal de MLP. Ver [guía V2](docs/EXPERIMENTO_CLIMA_V2.md). Mantiene V1 y los notebooks originales intactos.

Se añadió un experimento independiente en `src/experimento_clima.py`, con tres versiones de datos: imputación estacional del 52 % de celdas faltantes, ninguna imputación y relleno de huecos cortos con anomalías climáticas. Compara regresión/logística, árbol, Random Forest, MLP y CNN temporal, además de un control sin clima.

La metodología, cortes, reproducción y limitaciones están en [docs/EXPERIMENTO_CLIMA.md](docs/EXPERIMENTO_CLIMA.md). Este experimento no modifica los notebooks 03/04 ni sus resultados originales; las redes y regresiones mencionadas anteriormente como pendientes se implementan aquí como comparación exploratoria separada, no como parte de aquellos notebooks.
