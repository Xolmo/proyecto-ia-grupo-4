# Entregable parcial (semana 8)

**Proyecto:** Predicción de leishmaniosis cutánea en Cajamarca (Perú) · **Grupo 4** · Inteligencia Artificial Aplicada (1INF62)

**Integrantes:** Dhanny Sebastian Ramirez Rivera · Sergio Estéfano Huamán Noriega · Leonardo Jaime Flores Vera · Vilcapoma Madueño Franz Javier

## Checklist

| Requisito | Estado | Dónde está |
|---|---|---|
| README con descripción del problema, objetivo y metodología | ✅ | [`README.md`](README.md) |
| Papers revisados, mínimo 1 por integrante | ✅ 4 papers | [`papers/`](papers/) (`paper_integrante 1.pdf` … `paper_integrante 4.pdf`) |
| Resumen de cada paper | ✅ 4 resúmenes | [`papers/`](papers/) (`paper_integrante1_resumen.pdf` … `paper_integrante4_resumen.pdf`) |
| Dataset en `data/raw/` con origen, características y permisos | ✅ | [`data/raw/`](data/raw/) · [`descripcion_dataset_leishmaniosis.md`](data/raw/descripcion_dataset_leishmaniosis.md) |
| Análisis exploratorio (EDA) con visualizaciones | ✅ | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb), [`notebooks/02_eda_clima.ipynb`](notebooks/02_eda_clima.ipynb) · gráficos en [`results/plots/`](results/plots/) |
| Propuesta de modelos identificados en la literatura | ✅ | [`PIPELINE_PROPUESTO.md`](PIPELINE_PROPUESTO.md) |
| Baseline implementado y evaluado | ✅ | [`notebooks/03_baseline.ipynb`](notebooks/03_baseline.ipynb) · métricas en [`results/baseline_metricas.csv`](results/baseline_metricas.csv) |
| Dependencias | ✅ | [`requirements.txt`](requirements.txt) |

## Resumen de lo entregado

- **Problema y alcance:** anticipar los casos mensuales por distrito (regresión) y los meses de alerta (clasificación) en el departamento de Cajamarca, con el historial del CDC-MINSA 2000–2024 y el clima de SENAMHI.
- **Datos:** 156 819 casos individuales de leishmaniosis cutánea (23 departamentos), de los que se modela Cajamarca; series diarias de 26 estaciones SENAMHI de Cajamarca.
- **Preprocesamiento:** [`src/preprocessing.py`](src/preprocessing.py) construye el panel distrito × mes con rezagos, estacionalidad y alerta; [`src/integracion_clima.py`](src/integracion_clima.py) asigna a cada distrito la estación más cercana.
- **Propuesta de modelos:** líneas base (`lag1`, `lag12`), regresión lineal/logística, árbol de decisión, KNN, MLP, LSTM y un Transformer exploratorio, con las técnicas fuera del temario anotadas aparte.
- **Baseline:** dos reglas sin entrenamiento sobre la partición temporal (entrenamiento hasta 2018, prueba 2019–2024).

| Prueba 2019–2024 | B1 · último mes (`lag1`) | B2 · mismo mes del año anterior (`lag12`) |
|---|---|---|
| RMSE / MAE (casos por distrito-mes) | 1,14 / 0,54 | 1,22 / 0,60 |
| F1 de la alerta | 0,21 | 0,12 |
| PR-AUC de la alerta (al azar: 0,04) | 0,15 | 0,09 |

## Notas

- **Nombre del notebook del baseline.** La estructura propuesta indica `02_modelos.ipynb`; el baseline está en `03_baseline.ipynb`, porque `02` quedó para la exploración del clima.
- **Referencia trivial.** Predecir siempre 0 obtiene RMSE 1,05 y MAE 0,41 en la prueba, mejor que B1 en ambas métricas, porque el 75 % de las filas tiene cero casos. Por eso la comparación final usará también MASE y las métricas de alerta.
- **Clima parcial.** Solo el 55 % de las filas distrito × mes de Cajamarca tiene clima asociado, y no cubre el periodo de prueba; las líneas base usan solo variables de casos.
- **Paper del integrante 3.** Es una tesis de maestría sobre dengue en Bucaramanga, incluida por su diseño similar (casos + clima rezagado).
- **Pendientes de decisión:** mínimo de casos por distrito dentro de Cajamarca, definición final de alerta y tratamiento de los meses sin clima (ver [`PIPELINE_PROPUESTO.md`](PIPELINE_PROPUESTO.md)).

## Pendiente para el entregable final

- [ ] Entrenar y comparar los modelos propuestos (`notebooks/02_modelos.ipynb`).
- [ ] `src/models.py` y `src/evaluation.py`.
- [ ] `results/metrics.csv` y `results/model_comparison.txt`.
- [ ] Reporte académico (`report.pdf`, 3–5 páginas) y presentación (`presentation.pdf`).
- [ ] `ENTREGABLE_FINAL.md`.
