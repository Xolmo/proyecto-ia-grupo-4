# Resultados del experimento climático

Comparación exploratoria con semilla 42. Los notebooks originales no se modificaron.

## Conclusión de esta ejecución

En prueba oficial, el menor RMSE puntual fue mlp + estacional_52: 0.9188, frente a 0.9220 de mlp sin imputación. La diferencia relativa es aproximadamente 0.349 %; no se demostró significancia estadística. La selección exclusiva por validación fue random_forest + huecos_cortos. En el histórico, el menor RMSE observado fue cnn + solo_casos (0.9993).

Para alertas oficiales, random_forest + estacional_52 alcanzó average precision 0.1541 y recall 0.5917: detectó 71 de 120 alertas y generó 734 falsas alarmas. Es un resultado descriptivo de prueba; su selección independiente se informa por separado y no debe presentarse como sistema operativo validado.

La propuesta de huecos cortos no reconstruye años enteros sin estaciones: en prueba oficial su uso máximo de rama climática fue 0.0%. El relleno del 52 % no significa haber recuperado mediciones reales; 9291 rellenos se apoyan directamente en otros ya imputados. Los resultados MLP están sujetos al diagnóstico de convergencia informado al final.

## Datasets y presupuesto

| protocol | variant | districts | rows | initial_missing_cells | filled_cells | fraction_missing_filled | recursive_sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historico | sin_imputar | 46 | 7728 | 1639 | 0 | 0.0000 | 0 |
| historico | estacional_52 | 46 | 7728 | 1639 | 852 | 0.5198 | 121 |
| historico | huecos_cortos | 46 | 7728 | 1639 | 603 | 0.3679 | 0 |
| oficial | sin_imputar | 57 | 16416 | 22586 | 0 | 0.0000 | 0 |
| oficial | estacional_52 | 57 | 16416 | 22586 | 11745 | 0.5200 | 9291 |
| oficial | huecos_cortos | 57 | 16416 | 22586 | 780 | 0.0345 | 0 |

Las fracciones se refieren a celdas originalmente nulas de lluvia total/tmax/tmin. El redondeo a celdas impide representar exactamente 0,52. Las columnas derivadas no se cuentan otra vez.

La imputación del 52 % permite fuentes previamente imputadas para alcanzar el presupuesto. Esta propagación puede fabricar regularidad y no recupera mediciones reales. La guía registra fuentes y fechas.

## Ganadores seleccionados por validación

| protocol | task | variant | family | validation_score | RMSE | MAE | PR_AUC | recall | climate_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historico | alerta | solo_casos | random_forest | 0.1595 | — | — | 0.0201 | 0.0000 | 0.0000 |
| historico | casos | solo_casos | random_forest | 1.3294 | 1.0159 | 0.7185 | — | — | 0.0000 |
| oficial | alerta | huecos_cortos | random_forest | 0.1018 | — | — | 0.1203 | 0.5500 | 0.0000 |
| oficial | casos | huecos_cortos | random_forest | 1.1270 | 0.9259 | 0.5977 | — | — | 0.0000 |

RMSE/MAE corresponden a casos; PR_AUC/recall a alertas. `climate_fraction` es la fracción de predicciones de prueba que efectivamente utilizó la rama climática. Las restantes usan el modelo de casos de la misma familia.

## Comparación por versión: historico

### casos: mejor familia de cada versión según validación

| variant | family | candidate | validation_score | RMSE | MAE | PR_AUC | F1 | recall | climate_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| solo_casos | random_forest | 1.0000 | 1.3294 | 1.0159 | 0.7185 | — | — | — | 0.0000 |
| estacional_52 | random_forest | 1.0000 | 1.3458 | 1.0394 | 0.7454 | — | — | — | 0.9457 |
| sin_imputar | random_forest | 0.0000 | 1.3475 | 1.0293 | 0.7322 | — | — | — | 0.7156 |
| huecos_cortos | random_forest | 1.0000 | 1.3475 | 1.0417 | 0.7452 | — | — | — | 0.8804 |

### alerta: mejor familia de cada versión según validación

| variant | family | candidate | validation_score | RMSE | MAE | PR_AUC | F1 | recall | climate_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| solo_casos | random_forest | 0.0000 | 0.1595 | — | — | 0.0201 | 0.0000 | 0.0000 | 0.0000 |
| estacional_52 | random_forest | 1.0000 | 0.1322 | — | — | 0.0275 | 0.0351 | 0.0435 | 0.9457 |
| sin_imputar | random_forest | 1.0000 | 0.1207 | — | — | 0.0302 | 0.0333 | 0.0435 | 0.7156 |
| huecos_cortos | cnn | 0.0000 | 0.1048 | — | — | 0.0301 | 0.0699 | 0.2174 | 0.8804 |

### Referencias de regresión sobre las mismas filas

| family | RMSE | MAE | MASE |
| --- | --- | --- | --- |
| ultimo_mes | 1.2978 | 0.7083 | 0.5390 |
| estacional | 1.4713 | 0.8279 | 0.6300 |
| cero | 1.1739 | 0.5338 | 0.4062 |

## Comparación por versión: oficial

### casos: mejor familia de cada versión según validación

| variant | family | candidate | validation_score | RMSE | MAE | PR_AUC | F1 | recall | climate_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huecos_cortos | random_forest | 1.0000 | 1.1270 | 0.9259 | 0.5977 | — | — | — | 0.0000 |
| solo_casos | random_forest | 1.0000 | 1.1273 | 0.9259 | 0.5977 | — | — | — | 0.0000 |
| sin_imputar | random_forest | 1.0000 | 1.1273 | 0.9259 | 0.5977 | — | — | — | 0.0000 |
| estacional_52 | lineal | 1.0000 | 1.1286 | 0.9399 | 0.5842 | — | — | — | 0.1806 |

### alerta: mejor familia de cada versión según validación

| variant | family | candidate | validation_score | RMSE | MAE | PR_AUC | F1 | recall | climate_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huecos_cortos | random_forest | 0.0000 | 0.1018 | — | — | 0.1203 | 0.1397 | 0.5500 | 0.0000 |
| solo_casos | random_forest | 0.0000 | 0.1017 | — | — | 0.1203 | 0.1397 | 0.5500 | 0.0000 |
| sin_imputar | random_forest | 0.0000 | 0.1017 | — | — | 0.1203 | 0.1397 | 0.5500 | 0.0000 |
| estacional_52 | arbol | 1.0000 | 0.0989 | — | — | 0.1349 | 0.1943 | 0.3417 | 0.1806 |

### Referencias de regresión sobre las mismas filas

| family | RMSE | MAE | MASE |
| --- | --- | --- | --- |
| ultimo_mes | 1.1874 | 0.5609 | 0.5454 |
| estacional | 1.2521 | 0.6174 | 0.6003 |
| cero | 1.0996 | 0.4323 | 0.4203 |

## Mejor resultado observado en prueba (descriptivo)

Esta tabla compara resultados ya evaluados; no debe confundirse con selección independiente por validación ni garantiza que vuelva a ganar fuera de muestra.

| protocol | task | variant | family | RMSE | MAE | PR_AUC | F1 | recall | climate_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historico | alerta | huecos_cortos | mlp | — | — | 0.0452 | 0.0976 | 0.0870 | 0.8804 |
| historico | casos | solo_casos | cnn | 0.9993 | 0.7032 | — | — | — | 0.0000 |
| oficial | alerta | estacional_52 | random_forest | — | — | 0.1541 | 0.1535 | 0.5917 | 0.1806 |
| oficial | casos | estacional_52 | mlp | 0.9188 | 0.5732 | — | — | — | 0.1806 |

## Interpretación y límites

- Se compararon 5 familias × 4 condiciones × 2 protocolos × 2 tareas: 80 combinaciones, con 2 candidatos por combinación. El control de casos se reutiliza cuando las entradas son idénticas.
- Histórico: ajuste 2001–2009, validación 2010–2011, prueba 2012–2014. Oficial: ajuste 2001–2015, validación 2016–2018, prueba 2019–2024.
- La selección de distritos y estaciones usa solo el ajuste inicial: 46 distritos históricos y 57 oficiales; no son los 70 del notebook original. Por ello los números nuevos no se comparan directamente con `arbol_metricas.csv`.
- Las etiquetas de alerta se calculan nuevamente en cada corte de entrenamiento. Los umbrales probabilísticos se eligen en validación y se conservan en prueba.
- El clima de 2019–2024 carece de observaciones; la variante 52 puede consumir estimaciones propagadas. Un buen resultado allí no demuestra haber recuperado el clima real.
- No se imputa ningún nulo restante silenciosamente; el respaldo de casos mantiene exactamente las mismas filas de evaluación por protocolo.
- Una sola semilla, pocos hiperparámetros, MLP con máximo de 1000 iteraciones y CNN de 50 épocas. Son comparaciones exploratorias sin significancia estadística ni validación clínica.

MLP: los 42 ajustes convergieron antes del límite de 1000 iteraciones (iteraciones usadas: 24–187).

Ver `metricas.csv`, `validacion.csv`, `predicciones.csv`, `seleccion_por_validacion.csv`, `diagnostico_ajuste.csv` y `ejecucion.json` para la evidencia completa.
