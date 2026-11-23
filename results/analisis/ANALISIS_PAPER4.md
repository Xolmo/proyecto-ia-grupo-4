# Réplica del paper 4 en Cajamarca: análisis

**Paper:** El Alaoui, O., El Khiat, A., Hakem, A., et al. (2027). *Spatiotemporal dynamics and ecological determinants of cutaneous leishmaniasis in Errachidia Province, southeastern Morocco: A 16-year municipality-level analysis with SARIMA forecasting*. Parasitology International, 116, 103363 ([PDF](../../papers/paper_integrante4.pdf), §2.6 y §4.7).
**Integrante:** Franz Javier Vilcapoma Madueño.
**Código:** [`src/modelo_paper4.py`](../../src/modelo_paper4.py) · **Pruebas:** [`src/tests/test_modelo_paper4.py`](../../src/tests/test_modelo_paper4.py) · **Notebook:** [`notebooks/07_paper_integrante4_sarima.ipynb`](../../notebooks/07_paper_integrante4_sarima.ipynb) · **Tablas:** [`results/comparacion_final/`](../comparacion_final/tabla_comparativa_final.md)

## 1. Qué se replicó

El paper modela los casos mensuales de leishmaniosis cutánea de la provincia de Errachidia (2010–2025) con un SARIMA construido por Box-Jenkins:

1. Pruebas de estacionariedad ADF y KPSS para decidir la diferenciación. El paper aplicó d = 1 y D = 1.
2. Búsqueda en cuadrícula de los órdenes, eligiendo el menor AIC. El elegido fue **SARIMA(0,1,0)(1,1,0)₁₂**.
3. Prueba de Ljung-Box sobre los residuos (p > 0,05 indica ruido blanco).
4. Evaluación con RMSE y MAE.

Aquí se implementaron dos variantes, solo con el historial de casos (el paper no usa clima dentro del SARIMA):

| Variante | Orden |
|---|---|
| **SARIMA del paper** | (0,1,0)(1,1,0)₁₂ en los 57 distritos |
| **SARIMA por AIC** | el procedimiento del paper en cada distrito: p, q, P, Q ∈ {0, 1}, d ∈ {0, 1}, D = 1, s = 12; menor AIC en el tramo de ajuste |

## 2. En qué se diferencia (y por qué)

| Paper | Aquí | Motivo |
|---|---|---|
| Una serie provincial | 57 series distritales, un SARIMA por distrito | El proyecto predice por distrito |
| Partición 80/20 | Ajuste ≤2015, validación 2016–2018, prueba 2019–2024 | Comparar en las mismas 4 104 filas que los demás modelos |
| Pronóstico de varios meses | Pronóstico a un paso con los casos observados hasta el mes anterior | Igual que los demás modelos, que usan rezagos observados |
| ADF/KPSS para elegir d y D | D = 1 fijo; d ∈ {0, 1} elegido por AIC | La estacionalidad anual es clara en todos los distritos (EDA); el AIC decide si hace falta d = 1 |
| Solo regresión | Regresión y alerta | La alerta usa los casos predichos como puntaje, con el umbral elegido en validación (igual que el paper 2) |

Para la prueba se reajustan los parámetros con ajuste + validación (≤2018), con el mismo orden. Si un ajuste falla, se usa como respaldo un paseo aleatorio estacional (0,1,0)(0,1,0)₁₂.

## 3. Resultados

**Protocolo oficial** (prueba 2019–2024, 57 distritos, 4 104 filas, 120 alertas):

| Modelo | RMSE validación | RMSE ↓ | MAE ↓ | PR-AUC ↑ | Alertas detectadas | Falsas alarmas |
|---|---:|---:|---:|---:|---:|---:|
| SARIMA por AIC | 1,158 | 0,946 | **0,556** | 0,083 | 66/120 | 1 092 |
| SARIMA del paper | 1,516 | 1,246 | 0,669 | 0,141 | 50/120 | 518 |
| *Referencia:* DFNN (paper 2), mejor por validación | 1,115 | 0,916 | 0,602 | 0,158 | — | — |
| *Referencia:* Random Forest (equipo) | 1,127 | 0,926 | 0,598 | 0,120 | 66/120 | 759 |
| *Referencia:* siempre cero | — | 1,100 | 0,432 | — | — | — |

**Protocolo histórico** (prueba 2012–2014, 46 distritos): SARIMA por AIC RMSE 1,086 y MAE 0,731; SARIMA del paper RMSE 1,373. El mejor por validación es Random Forest (prueba 1,016).

**Diagnóstico:** el AIC eligió sobre todo (1,0,1)(0,1,1)₁₂ (22 distritos) y (0,1,1)(0,1,1)₁₂ (13 distritos). Los residuos son ruido blanco (Ljung-Box p > 0,05) en el 86 % de los distritos. Detalle en `results/paper4/ordenes_por_distrito.csv`.

## 4. Interpretación

1. **El orden del paper no se traslada a otra escala.** (0,1,0)(1,1,0)₁₂, con doble diferenciación, funcionó para la serie provincial de Errachidia, con cientos de casos. En series distritales de 0–14 casos por mes, la diferenciación amplifica el ruido y el RMSE (1,246) queda peor que predecir siempre cero. Repetir el procedimiento del paper (AIC por distrito) en lugar de su resultado baja el RMSE a 0,946.
2. **SARIMA por AIC tiene el mejor MAE de todos los modelos (0,556).** Cada distrito se modela con su propio nivel. Los modelos de aprendizaje automático aprenden de todos los distritos a la vez y tienden a predecir valores intermedios en los muchos meses con 0–1 casos. Ningún modelo supera aún a "siempre cero" en MAE (0,432).
3. **En RMSE ganan los modelos no lineales.** DFNN, MLP y Random Forest (0,916–0,926) aprovechan patrones comunes entre distritos y fallan menos en los meses con más casos, que son los que más pesan en el RMSE.
4. **Como señal de alerta, el SARIMA del paper es competitivo.** Su PR-AUC (0,141) supera a todos los modelos del equipo (máx. 0,121), aunque queda por debajo de la DFNN del paper 2 (0,158). Su predicción sobrerreacciona a los meses recientes, lo que ordena bien los meses de riesgo aunque erre en la cantidad.

## 5. Limitaciones

- Una serie por distrito con pocos casos: muchos meses en cero, lo que es poco favorable para un modelo gaussiano como SARIMA.
- No se probó SARIMAX con clima. El paper solo correlaciona el clima por separado, y el clima de 2019–2024 no está observado.
- D = 1 se fijó por la estacionalidad del EDA, sin repetir ADF/KPSS en cada distrito.
- Prueba a un paso, igual que los demás modelos; no evalúa pronósticos a varios meses como el paper.
