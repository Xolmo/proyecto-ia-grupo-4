# Versión 2: conservar los valores observados

No se rellena ninguna celda climática. Se conservan las observaciones y los nulos originales en lluvia total, tmax y tmin; las variables derivadas y sus rezagos se calculan sin completar huecos.

## Archivos

- `data/processed/experimento_clima/historico/sin_imputar.csv`.
- `data/processed/experimento_clima/oficial/sin_imputar.csv`.

`resumen_imputacion.csv` debe mostrar cero celdas rellenadas y la misma cantidad de nulos primarios iniciales y finales. Las columnas `*_imputado` son falsas en todas las filas. Los meses sin medición se conservan como nulos, nunca como cero.

## Entrenamiento y evaluación

La rama de clima solo utiliza filas con todos los rezagos climáticos requeridos observados. Para conservar una evaluación comparable, las filas incompletas se predicen con un modelo de casos de la misma familia; no se eliminan del conjunto de prueba ni se completa su clima con medianas.

En 2019–2024 el respaldo de casos debe cubrir todas las filas por ausencia de estaciones. El protocolo histórico permite evaluar clima observado en 2012–2014. Ver [cortes, modelos y métricas](EXPERIMENTO_CLIMA.md).

