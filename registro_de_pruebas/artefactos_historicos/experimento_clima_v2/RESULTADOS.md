# Resultados V2

Se conservaron los datasets V1, sus etiquetas de prueba y exactamente sus 230 400 predicciones objetivo. Se probaron seis candidatos (ventana/configuración) por combinación. No se implementó un modelo en dos etapas.

## Lectura de la ejecución

En prueba oficial, el menor RMSE de V1 fue 0.9229 y el de V2 0.9294. V2 no mejoró la regresión: conviene conservar el regresor V1 mientras se estudian otras mejoras.

En alertas, la mejor PR-AUC observada en V1 fue 0.1541; la combinación V2 elegida por validación obtuvo 0.1940 en prueba. Esta última utiliza clima en 0.0% de las filas, por lo que no se atribuye el avance a mediciones climáticas inexistentes.

Los siguientes cortes pertenecen al mismo clasificador; muestran cuánto cambia la sensibilidad al exigir más al puntaje:

| policy | threshold | precision | recall | predicted_alerts | false_alerts |
| --- | --- | --- | --- | --- | --- |
| f1 | 0.5758 | 0.1219 | 0.5667 | 558.0000 | 490.0000 |
| precision_20 | 0.8047 | 0.5833 | 0.0583 | 12.0000 | 5.0000 |

## Selección por validación

| protocol | task | variant | family | window | validation_score | RMSE | MAE | PR_AUC | F1 | recall | precision | climate_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historico | alerta | solo_casos | random_forest | 5 | 0.2904 | — | — | 0.0896 | 0.1200 | 0.2609 | 0.0779 | 0.0000 |
| historico | casos | solo_casos | random_forest | 5 | 1.3221 | 0.9973 | 0.7158 | — | — | — | — | 0.0000 |
| oficial | alerta | huecos_cortos | random_forest | 10 | 0.1422 | — | — | 0.1940 | 0.2006 | 0.5667 | 0.1219 | 0.0000 |
| oficial | casos | solo_casos | random_forest | 5 | 1.1103 | 0.9305 | 0.5963 | — | — | — | — | 0.0000 |

## Mejor resultado observado en prueba (exploratorio)

| protocol | task | variant | family | window | RMSE | MAE | PR_AUC | F1 | recall | precision | false_alerts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historico | alerta | sin_imputar | cnn | 5 | — | — | 0.1366 | 0.2154 | 0.3043 | 0.1667 | 35.0000 |
| historico | casos | huecos_cortos | random_forest | 5 | 0.9966 | 0.7180 | — | — | — | — | — |
| oficial | alerta | solo_casos | random_forest | 10 | — | — | 0.1940 | 0.2006 | 0.5667 | 0.1219 | 490.0000 |
| oficial | casos | estacional_52 | random_forest | 5 | 0.9294 | 0.5965 | — | — | — | — | — |

## Oficial: todos los modelos, casos

| variant | family | window | RMSE | MAE |
| --- | --- | --- | --- | --- |
| solo_casos | lineal | 10 | 0.9451 | 0.5947 |
| solo_casos | arbol | 5 | 0.9539 | 0.6039 |
| solo_casos | random_forest | 5 | 0.9305 | 0.5963 |
| solo_casos | mlp | 10 | 0.9565 | 0.5995 |
| solo_casos | cnn | 10 | 0.9641 | 0.6049 |
| sin_imputar | lineal | 10 | 0.9451 | 0.5947 |
| sin_imputar | arbol | 5 | 0.9539 | 0.6039 |
| sin_imputar | random_forest | 5 | 0.9305 | 0.5963 |
| sin_imputar | mlp | 10 | 0.9565 | 0.5995 |
| sin_imputar | cnn | 10 | 0.9641 | 0.6049 |
| estacional_52 | lineal | 10 | 0.9448 | 0.5934 |
| estacional_52 | arbol | 10 | 0.9395 | 0.6033 |
| estacional_52 | random_forest | 5 | 0.9294 | 0.5965 |
| estacional_52 | mlp | 10 | 0.9593 | 0.6000 |
| estacional_52 | cnn | 10 | 0.9732 | 0.6090 |
| huecos_cortos | lineal | 10 | 0.9451 | 0.5947 |
| huecos_cortos | arbol | 5 | 0.9539 | 0.6039 |
| huecos_cortos | random_forest | 5 | 0.9305 | 0.5963 |
| huecos_cortos | mlp | 10 | 0.9565 | 0.5995 |
| huecos_cortos | cnn | 10 | 0.9641 | 0.6049 |

## Oficial: todos los modelos, alerta

| variant | family | window | policy | threshold | validation_feasible | PR_AUC | precision | recall | F1 | false_alerts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| solo_casos | lineal | 10 | f1 | 0.7301 | True | 0.1855 | 0.1374 | 0.4167 | 0.2066 | 314.0000 |
| solo_casos | lineal | 10 | precision_20 | inf | False | 0.1855 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| solo_casos | arbol | todo | f1 | 0.6202 | True | 0.1322 | 0.1381 | 0.4833 | 0.2148 | 362.0000 |
| solo_casos | arbol | todo | precision_20 | inf | False | 0.1322 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| solo_casos | random_forest | 10 | f1 | 0.5758 | True | 0.1940 | 0.1219 | 0.5667 | 0.2006 | 490.0000 |
| solo_casos | random_forest | 10 | precision_20 | 0.8047 | True | 0.1940 | 0.5833 | 0.0583 | 0.1061 | 5.0000 |
| solo_casos | mlp | todo | f1 | 0.6615 | True | 0.0562 | 0.0850 | 0.1750 | 0.1144 | 226.0000 |
| solo_casos | mlp | todo | precision_20 | 0.8663 | True | 0.0562 | 0.1228 | 0.0583 | 0.0791 | 50.0000 |
| solo_casos | cnn | 5 | f1 | 0.6890 | True | 0.0569 | 0.0724 | 0.2250 | 0.1095 | 346.0000 |
| solo_casos | cnn | 5 | precision_20 | 0.8424 | True | 0.0569 | 0.0630 | 0.0667 | 0.0648 | 119.0000 |
| sin_imputar | lineal | 10 | f1 | 0.7301 | True | 0.1855 | 0.1374 | 0.4167 | 0.2066 | 314.0000 |
| sin_imputar | lineal | 10 | precision_20 | inf | False | 0.1855 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sin_imputar | arbol | todo | f1 | 0.6202 | True | 0.1322 | 0.1381 | 0.4833 | 0.2148 | 362.0000 |
| sin_imputar | arbol | todo | precision_20 | inf | False | 0.1322 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sin_imputar | random_forest | 10 | f1 | 0.5758 | True | 0.1940 | 0.1219 | 0.5667 | 0.2006 | 490.0000 |
| sin_imputar | random_forest | 10 | precision_20 | 0.8047 | True | 0.1940 | 0.5833 | 0.0583 | 0.1061 | 5.0000 |
| sin_imputar | mlp | todo | f1 | 0.6615 | True | 0.0562 | 0.0850 | 0.1750 | 0.1144 | 226.0000 |
| sin_imputar | mlp | todo | precision_20 | 0.8663 | True | 0.0562 | 0.1228 | 0.0583 | 0.0791 | 50.0000 |
| sin_imputar | cnn | 5 | f1 | 0.6890 | True | 0.0569 | 0.0724 | 0.2250 | 0.1095 | 346.0000 |
| sin_imputar | cnn | 5 | precision_20 | 0.8424 | True | 0.0569 | 0.0630 | 0.0667 | 0.0648 | 119.0000 |
| estacional_52 | lineal | todo | f1 | 0.3921 | True | 0.1718 | 0.0656 | 0.7333 | 0.1204 | 1254.0000 |
| estacional_52 | lineal | todo | precision_20 | inf | False | 0.1718 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| estacional_52 | arbol | todo | f1 | 0.6491 | True | 0.1112 | 0.1056 | 0.5333 | 0.1763 | 542.0000 |
| estacional_52 | arbol | todo | precision_20 | inf | False | 0.1112 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| estacional_52 | random_forest | 5 | f1 | 0.4990 | True | 0.1294 | 0.0989 | 0.7417 | 0.1745 | 811.0000 |
| estacional_52 | random_forest | 5 | precision_20 | 0.7768 | True | 0.1294 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| estacional_52 | mlp | todo | f1 | 0.2852 | True | 0.0569 | 0.0553 | 0.3667 | 0.0962 | 751.0000 |
| estacional_52 | mlp | todo | precision_20 | inf | False | 0.0569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| estacional_52 | cnn | todo | f1 | 0.2979 | True | 0.1173 | 0.0583 | 0.7500 | 0.1082 | 1454.0000 |
| estacional_52 | cnn | todo | precision_20 | 0.7283 | True | 0.1173 | 0.1134 | 0.1833 | 0.1401 | 172.0000 |
| huecos_cortos | lineal | 10 | f1 | 0.7301 | True | 0.1855 | 0.1374 | 0.4167 | 0.2066 | 314.0000 |
| huecos_cortos | lineal | 10 | precision_20 | inf | False | 0.1855 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| huecos_cortos | arbol | todo | f1 | 0.6202 | True | 0.1322 | 0.1381 | 0.4833 | 0.2148 | 362.0000 |
| huecos_cortos | arbol | todo | precision_20 | inf | False | 0.1322 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| huecos_cortos | random_forest | 10 | f1 | 0.5758 | True | 0.1940 | 0.1219 | 0.5667 | 0.2006 | 490.0000 |
| huecos_cortos | random_forest | 10 | precision_20 | 0.8047 | True | 0.1940 | 0.5833 | 0.0583 | 0.1061 | 5.0000 |
| huecos_cortos | mlp | todo | f1 | 0.6615 | True | 0.0562 | 0.0850 | 0.1750 | 0.1144 | 226.0000 |
| huecos_cortos | mlp | todo | precision_20 | 0.8663 | True | 0.0562 | 0.1228 | 0.0583 | 0.0791 | 50.0000 |
| huecos_cortos | cnn | 5 | f1 | 0.6890 | True | 0.0569 | 0.0724 | 0.2250 | 0.1095 | 346.0000 |
| huecos_cortos | cnn | 5 | precision_20 | 0.8424 | True | 0.0569 | 0.0630 | 0.0667 | 0.0648 | 119.0000 |

## MLP: diagnóstico

Se registraron 98 ajustes de MLP. 1 ajustes de selección alcanzaron el límite de 600 épocas. Mediana de épocas seleccionadas: 40. La parada usa validación cronológica y no significa haber demostrado un óptimo numérico.

## Qué significa el umbral

El umbral distrital de casos define la alerta real (media + 2 desviaciones). El corte del clasificador es otro número, entre 0 y 1: es un hiperparámetro de decisión aplicado a un puntaje no necesariamente calibrado. El objetivo alternativo de precisión 20 % se aplica solo a validación y no garantiza esa precisión en prueba.

## Alcance

- Las variables de casos anteriores y estacionalidad se conservaron; se añadieron distrito, conteos relativos y clima resumido/anómalo por distrito y mes.
- Las ventanas modifican los ejemplos usados para entrenar; no eliminan lag1/lag12 ni cambian las etiquetas entre modelos.
- El clima oficial sigue sin nuevas observaciones: una mejora con imputación no demuestra reconstrucción del clima real.
- El conjunto de prueba ya fue examinado en V1: V2 es desarrollo exploratorio, no una nueva prueba externa independiente.
- Los resultados anteriores y los notebooks originales permanecen intactos.
