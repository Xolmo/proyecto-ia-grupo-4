# Entregable parcial (semana 8) — checklist

**Proyecto:** Predicción de leishmaniosis cutánea en Cajamarca · Grupo 4 · Inteligencia Artificial Aplicada.

**Integrantes:** Dhanny Sebastian Ramirez Rivera, Sergio Estéfano Huamán Noriega, Leonardo Jaime Flores Vera y Vilcapoma Madueño Franz Javier.

Esta actualización distingue lo implementado de lo propuesto. El título conserva la referencia al entregable parcial; el estado incluye el notebook de árboles incorporado al repositorio.

Este archivo registra lo entregado en la semana 8. Lo completado después para el entregable final está en [ENTREGABLE_FINAL.md](ENTREGABLE_FINAL.md).

## Implementado

| Componente | Evidencia |
|---|---|
| Exploración epidemiológica | `notebooks/01_eda.ipynb` |
| Exploración y cobertura climática | `notebooks/01b_eda_clima.ipynb` |
| Limpieza, panel mensual, rezagos y alertas | `src/preprocessing.py` |
| Integración climática auxiliar | `src/integracion_clima.py`; en la semana 8 no alimentaba los modelos evaluados |
| Líneas base | `notebooks/03_baseline.ipynb`, `results/baseline_metricas.csv` |
| Árboles de regresión y clasificación | `notebooks/04_arbol_decision.ipynb`, `results/arbol_metricas.csv` |
| Interpretabilidad | Importancia por permutación, gráficos y `results/arbol_reglas.txt` |
| Literatura y fuentes (1 paper por integrante) | `papers/paper_integrante1.pdf` a `paper_integrante4.pdf` y sus resúmenes, `data/raw/descripcion_dataset_leishmaniosis.md`, `data/raw/external/FUENTES.md` |

## Qué datos utilizan los resultados

Los notebooks 03 y 04 cargan casos CDC-MINSA, filtran Cajamarca y construyen el panel mensual. Utilizan distritos con al menos 50 casos acumulados, entrenamiento hasta 2018 y prueba 2019–2024.

Las entradas son siete variables de casos y calendario. La clasificación relativa añade cinco conteos divididos entre el umbral distrital. **No se utiliza clima:** los notebooks no leen `dataset_cajamarca.csv` y explican que las series climáticas no cubren la prueba.

## Resultados en prueba

| Métrica | Último mes | Año anterior | Árbol |
|---|---:|---:|---:|
| RMSE | 1,142 | 1,222 | 0,907 |
| MAE | 0,537 | 0,603 | 0,576 |
| F1 de alerta | 0,213 | 0,124 | 0,222 (relativa) |
| Recall de alerta | 0,213 | 0,126 | 0,301 (relativa) |
| PR-AUC de alerta | 0,153 | 0,090 | 0,155 (relativa) |

El árbol mejora RMSE, pero no MAE frente al último mes. La variante relativa detecta 55 de 183 alertas, con 258 falsas alarmas. La referencia constante cero obtiene RMSE 1,054 y MAE 0,413.

## Limitaciones que deben declararse

- El filtro de 50 casos utiliza el historial completo, incluida la prueba.
- La definición de alerta es experimental, no una norma del MINSA.
- Los umbrales distritales se calculan con todo el entrenamiento hasta 2018, incluso para los pliegues internos anteriores. Deben recalcularse por pliegue para una validación estricta.
- Se usan rezagos observados en prueba; no es una predicción recursiva de todo el periodo desde 2018.
- Escalamiento y SMOTE están disponibles como funciones, pero no se aplican a los árboles.

## Pendiente en la semana 8 y dónde se completó

| Pendiente | Estado en el entregable final |
|---|---|
| Entrenar regresión lineal/logística, MLP y otros modelos | Completado: 5 modelos en `src/models.py` (KNN, LSTM y Transformer no se implementaron) |
| Comparar con clima de cobertura suficiente | Completado con imputación estacional; el clima termina en 2015 (ver `results/model_comparison.txt`) |
| Selección temporal de distritos y umbrales | Completado: distritos y umbrales se calculan solo con el tramo de ajuste |
| Reporte académico y presentación | `report.pdf` y `presentation.pdf` |
| `src/models.py`, `src/evaluation.py`, `results/metrics.csv`, `results/model_comparison.txt` | Creados para el entregable final |
