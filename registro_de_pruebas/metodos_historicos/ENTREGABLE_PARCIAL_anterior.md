# Entregable parcial (semana 8): estado actualizado según código

**Proyecto:** Predicción de leishmaniosis cutánea en Cajamarca · Grupo 4 · Inteligencia Artificial Aplicada.

**Integrantes:** Dhanny Sebastian Ramirez Rivera, Sergio Estéfano Huamán Noriega, Leonardo Jaime Flores Vera y Vilcapoma Madueño Franz Javier.

Esta actualización distingue lo implementado de lo propuesto. El título conserva la referencia al entregable parcial; el estado incluye el notebook de árboles incorporado al repositorio.

**Experimento posterior independiente:** se añadió `src/experimento_clima.py` para comparar tres variantes climáticas, un control de casos y cinco familias de modelos; ver [guía](docs/EXPERIMENTO_CLIMA.md). No reemplaza las métricas ni los notebooks de este entregable.

## Implementado

| Componente | Evidencia |
|---|---|
| Exploración epidemiológica | `notebooks/01_eda.ipynb` |
| Exploración y cobertura climática | `notebooks/02_eda_clima.ipynb` |
| Limpieza, panel mensual, rezagos y alertas | `src/preprocessing.py` |
| Integración climática auxiliar | `src/integracion_clima.py`; no alimenta los modelos evaluados |
| Líneas base | `notebooks/03_baseline.ipynb`, `results/baseline_metricas.csv` |
| Árboles de regresión y clasificación | `notebooks/04_arbol_decision.ipynb`, `results/arbol_metricas.csv` |
| Interpretabilidad | Importancia por permutación, gráficos y `results/arbol_reglas.txt` |
| Literatura y fuentes | `papers/`, `data/raw/descripcion_dataset_leishmaniosis.md`, `data/raw/external/FUENTES.md` |

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

## Pendiente o propuesto

- [ ] Entrenar regresión lineal/logística, KNN, MLP, LSTM y Transformer si se mantiene ese alcance.
- [ ] Completar comparaciones de sensibilidad: semana 1, clima con cobertura suficiente y alerta alternativa.
- [ ] Revisar la validación interna y la selección temporal de distritos.
- [ ] Completar reporte académico y presentación finales; no aparecen en el repositorio revisado.
- [ ] Implementar una salida operativa si el grupo la mantiene en el alcance.

`src/models.py`, `src/evaluation.py`, `results/metrics.csv` y `results/model_comparison.txt` eran nombres propuestos; la implementación actual está en notebooks y usa los archivos de resultados indicados arriba.
