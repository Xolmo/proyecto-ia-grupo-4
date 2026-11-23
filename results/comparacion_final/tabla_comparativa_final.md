# Comparación final de modelos (equipo + paper 2 + paper 4)

Generado por `src/comparacion_final.py`. Prueba oficial: 2019-2024 (57 distritos, 4 104 filas, 120 alertas). Prueba histórica: 2012-2014 (46 distritos, 1 656 filas), con clima observado.

## Protocolo oficial

**Mejor por RMSE de validación:** Paper 2 · DFNN · Solo casos (7) (validación 1.115, prueba 0.916).

### Regresión (casos del mes siguiente)

| origen | modelo | entradas | RMSE_validacion | RMSE | RMSE_media_semillas | RMSE_sd_semillas | MAE | MASE |
|---|---|---|---|---|---|---|---|---|
| Paper 2 | DFNN | Casos (7) + clima | 1.115 | 0.916 | 0.922 | 0.016 | 0.602 | 0.585 |
| Paper 2 | DFNN | Solo casos (7) | 1.115 | 0.916 | 0.922 | 0.016 | 0.602 | 0.585 |
| Equipo | MLP | Solo casos (7) | 1.131 | 0.922 | — | — | 0.575 | 0.559 |
| Equipo | MLP | Casos (7) + clima | 1.131 | 0.922 | — | — | 0.575 | 0.559 |
| Equipo | Random Forest | Solo casos (7) | 1.127 | 0.926 | — | — | 0.598 | 0.581 |
| Equipo | Random Forest | Casos (7) + clima | 1.127 | 0.926 | — | — | 0.598 | 0.581 |
| Equipo | CNN temporal | Casos (7) + clima | 1.128 | 0.938 | — | — | 0.621 | 0.604 |
| Equipo | CNN temporal | Solo casos (7) | 1.128 | 0.938 | — | — | 0.621 | 0.604 |
| Equipo | Ridge / logística | Casos (7) + clima | 1.129 | 0.940 | — | — | 0.583 | 0.567 |
| Equipo | Ridge / logística | Solo casos (7) | 1.129 | 0.940 | — | — | 0.583 | 0.567 |
| Equipo | Árbol de decisión | Solo casos (7) | 1.153 | 0.941 | — | — | 0.611 | 0.594 |
| Equipo | Árbol de decisión | Casos (7) + clima | 1.153 | 0.941 | — | — | 0.611 | 0.594 |
| Paper 4 | SARIMA, orden por AIC por distrito | Solo casos (serie) | 1.158 | 0.946 | — | — | 0.556 | 0.541 |
| Paper 2 | SFNN2 | Solo casos (7) | 1.116 | 0.960 | 0.932 | 0.020 | 0.687 | 0.668 |
| Paper 2 | SFNN2 | Casos (7) + clima | 1.116 | 0.960 | 0.932 | 0.020 | 0.687 | 0.668 |
| Paper 2 | SFNN1 | Solo casos (7) | 1.121 | 0.971 | 0.938 | 0.022 | 0.704 | 0.685 |
| Paper 2 | SFNN1 | Casos (7) + clima | 1.121 | 0.971 | 0.938 | 0.022 | 0.704 | 0.685 |
| Paper 2 | SFNN1 | Solo casos (3 lags) | 1.147 | 0.976 | 0.955 | 0.014 | 0.701 | 0.681 |
| Paper 2 | SFNN1 | Casos (3 lags) + clima | 1.147 | 0.976 | 0.955 | 0.014 | 0.701 | 0.681 |
| Paper 2 | SFNN2 | Solo casos (3 lags) | 1.148 | 0.983 | 0.957 | 0.018 | 0.710 | 0.690 |
| Paper 2 | SFNN2 | Casos (3 lags) + clima | 1.148 | 0.983 | 0.957 | 0.018 | 0.710 | 0.690 |
| Paper 2 | DFNN | Solo casos (3 lags) | 1.145 | 1.002 | 0.956 | 0.026 | 0.749 | 0.728 |
| Paper 2 | DFNN | Casos (3 lags) + clima | 1.145 | 1.002 | 0.956 | 0.026 | 0.749 | 0.728 |
| Referencia | Regla: siempre cero | — | — | 1.100 | — | — | 0.432 | 0.420 |
| Referencia | Regla: último mes | — | — | 1.187 | — | — | 0.561 | 0.545 |
| Paper 4 | SARIMA (0,1,0)(1,1,0)12 del paper | Solo casos (serie) | 1.516 | 1.246 | — | — | 0.669 | 0.650 |
| Referencia | Regla: mismo mes año anterior | — | — | 1.252 | — | — | 0.617 | 0.600 |


`RMSE` usa la semilla 42; `RMSE_media_semillas` y `RMSE_sd_semillas` resumen 5 semillas (solo redes del paper 2; los modelos del equipo se entrenaron una vez y el SARIMA del paper 4 es determinista). Diferencias menores que la desviación no son interpretables. Las filas «+ clima» del protocolo oficial son idénticas a «solo casos» porque en 2019-2024 no hay clima observado y se usa el respaldo sin clima.

### Alerta (brote)

| origen | modelo | entradas | PR_AUC | ROC_AUC | EER | accuracy | recall | precision | FPR | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Paper 2 | DFNN | Solo casos (7) | 0.158 | 0.796 | 0.259 | 0.826 | 0.583 | 0.095 | 0.167 | 0.164 | 70 | 664 | 50 | 3320 |
| Paper 2 | DFNN | Casos (7) + clima | 0.158 | 0.796 | 0.259 | 0.826 | 0.583 | 0.095 | 0.167 | 0.164 | 70 | 664 | 50 | 3320 |
| Paper 4 | SARIMA (0,1,0)(1,1,0)12 del paper | Solo casos (serie) | 0.141 | 0.725 | 0.333 | 0.857 | 0.417 | 0.088 | 0.130 | 0.145 | 50 | 518 | 70 | 3466 |
| Paper 2 | SFNN2 | Solo casos (7) | 0.138 | 0.794 | 0.267 | 0.814 | 0.558 | 0.086 | 0.178 | 0.150 | 67 | 709 | 53 | 3275 |
| Paper 2 | SFNN2 | Casos (7) + clima | 0.138 | 0.794 | 0.267 | 0.814 | 0.558 | 0.086 | 0.178 | 0.150 | 67 | 709 | 53 | 3275 |
| Paper 2 | SFNN1 | Solo casos (7) | 0.137 | 0.771 | 0.326 | 0.819 | 0.550 | 0.088 | 0.173 | 0.151 | 66 | 688 | 54 | 3296 |
| Paper 2 | SFNN1 | Casos (7) + clima | 0.137 | 0.771 | 0.326 | 0.819 | 0.550 | 0.088 | 0.173 | 0.151 | 66 | 688 | 54 | 3296 |
| Paper 2 | SFNN1 | Solo casos (3 lags) | 0.125 | 0.727 | 0.316 | 0.779 | 0.600 | 0.077 | 0.215 | 0.137 | 72 | 858 | 48 | 3126 |
| Paper 2 | SFNN1 | Casos (3 lags) + clima | 0.125 | 0.727 | 0.316 | 0.779 | 0.600 | 0.077 | 0.215 | 0.137 | 72 | 858 | 48 | 3126 |
| Paper 2 | SFNN2 | Casos (3 lags) + clima | 0.124 | 0.726 | 0.321 | 0.803 | 0.542 | 0.079 | 0.189 | 0.138 | 65 | 754 | 55 | 3230 |
| Paper 2 | SFNN2 | Solo casos (3 lags) | 0.124 | 0.726 | 0.321 | 0.803 | 0.542 | 0.079 | 0.189 | 0.138 | 65 | 754 | 55 | 3230 |
| Paper 2 | DFNN | Solo casos (3 lags) | 0.124 | 0.727 | 0.327 | 0.659 | 0.700 | 0.058 | 0.342 | 0.107 | 84 | 1363 | 36 | 2621 |
| Paper 2 | DFNN | Casos (3 lags) + clima | 0.124 | 0.727 | 0.327 | 0.659 | 0.700 | 0.058 | 0.342 | 0.107 | 84 | 1363 | 36 | 2621 |
| Equipo | MLP | Solo casos (7) | 0.121 | 0.752 | 0.301 | 0.816 | 0.542 | 0.085 | 0.175 | 0.147 | 65 | 699 | 55 | 3285 |
| Equipo | MLP | Casos (7) + clima | 0.121 | 0.752 | 0.301 | 0.816 | 0.542 | 0.085 | 0.175 | 0.147 | 65 | 699 | 55 | 3285 |
| Equipo | Random Forest | Solo casos (7) | 0.120 | 0.747 | 0.307 | 0.802 | 0.550 | 0.080 | 0.191 | 0.140 | 66 | 759 | 54 | 3225 |
| Equipo | Random Forest | Casos (7) + clima | 0.120 | 0.747 | 0.307 | 0.802 | 0.550 | 0.080 | 0.191 | 0.140 | 66 | 759 | 54 | 3225 |
| Equipo | CNN temporal | Solo casos (7) | 0.112 | 0.745 | 0.309 | 0.853 | 0.417 | 0.086 | 0.134 | 0.142 | 50 | 532 | 70 | 3452 |
| Equipo | CNN temporal | Casos (7) + clima | 0.112 | 0.745 | 0.309 | 0.853 | 0.417 | 0.086 | 0.134 | 0.142 | 50 | 532 | 70 | 3452 |
| Equipo | Ridge / logística | Casos (7) + clima | 0.107 | 0.690 | 0.408 | 0.874 | 0.392 | 0.096 | 0.111 | 0.154 | 47 | 443 | 73 | 3541 |
| Equipo | Ridge / logística | Solo casos (7) | 0.107 | 0.690 | 0.408 | 0.874 | 0.392 | 0.096 | 0.111 | 0.154 | 47 | 443 | 73 | 3541 |
| Equipo | Árbol de decisión | Casos (7) + clima | 0.093 | 0.702 | 0.375 | 0.889 | 0.358 | 0.102 | 0.095 | 0.158 | 43 | 380 | 77 | 3604 |
| Equipo | Árbol de decisión | Solo casos (7) | 0.093 | 0.702 | 0.375 | 0.889 | 0.358 | 0.102 | 0.095 | 0.158 | 43 | 380 | 77 | 3604 |
| Paper 4 | SARIMA, orden por AIC por distrito | Solo casos (serie) | 0.083 | 0.702 | 0.360 | 0.721 | 0.550 | 0.057 | 0.274 | 0.103 | 66 | 1092 | 54 | 2892 |


## Protocolo historico

**Mejor por RMSE de validación:** Equipo · Random Forest · Solo casos (7) (validación 1.329, prueba 1.016).

### Regresión (casos del mes siguiente)

| origen | modelo | entradas | RMSE_validacion | RMSE | RMSE_media_semillas | RMSE_sd_semillas | MAE | MASE |
|---|---|---|---|---|---|---|---|---|
| Paper 2 | DFNN | Solo casos (7) | 1.330 | 0.998 | 1.007 | 0.028 | 0.728 | 0.554 |
| Equipo | CNN temporal | Solo casos (7) | 1.335 | 0.999 | — | — | 0.703 | 0.535 |
| Equipo | Random Forest | Solo casos (7) | 1.329 | 1.016 | — | — | 0.718 | 0.547 |
| Paper 2 | SFNN1 | Solo casos (7) | 1.347 | 1.016 | 1.020 | 0.011 | 0.750 | 0.570 |
| Equipo | Árbol de decisión | Solo casos (7) | 1.394 | 1.016 | — | — | 0.722 | 0.549 |
| Equipo | MLP | Solo casos (7) | 1.339 | 1.017 | — | — | 0.722 | 0.549 |
| Equipo | CNN temporal | Casos (7) + clima | 1.363 | 1.022 | — | — | 0.728 | 0.554 |
| Paper 2 | SFNN2 | Solo casos (7) | 1.336 | 1.024 | 1.016 | 0.021 | 0.749 | 0.570 |
| Paper 2 | DFNN | Casos (7) + clima | 1.352 | 1.024 | 1.032 | 0.018 | 0.757 | 0.576 |
| Equipo | Random Forest | Casos (7) + clima | 1.347 | 1.029 | — | — | 0.732 | 0.557 |
| Equipo | Ridge / logística | Solo casos (7) | 1.352 | 1.031 | — | — | 0.715 | 0.544 |
| Paper 2 | SFNN1 | Casos (7) + clima | 1.349 | 1.033 | 1.035 | 0.005 | 0.734 | 0.558 |
| Equipo | Árbol de decisión | Casos (7) + clima | 1.380 | 1.035 | — | — | 0.736 | 0.560 |
| Equipo | Ridge / logística | Casos (7) + clima | 1.356 | 1.035 | — | — | 0.719 | 0.547 |
| Equipo | MLP | Casos (7) + clima | 1.351 | 1.037 | — | — | 0.732 | 0.557 |
| Paper 2 | SFNN2 | Casos (7) + clima | 1.352 | 1.044 | 1.037 | 0.008 | 0.735 | 0.559 |
| Paper 2 | SFNN2 | Casos (3 lags) + clima | 1.367 | 1.045 | 1.061 | 0.014 | 0.770 | 0.586 |
| Paper 2 | SFNN1 | Casos (3 lags) + clima | 1.366 | 1.046 | 1.067 | 0.020 | 0.783 | 0.596 |
| Paper 2 | SFNN1 | Solo casos (3 lags) | 1.375 | 1.050 | 1.050 | 0.008 | 0.797 | 0.607 |
| Paper 2 | SFNN2 | Solo casos (3 lags) | 1.379 | 1.050 | 1.046 | 0.015 | 0.760 | 0.579 |
| Paper 2 | DFNN | Solo casos (3 lags) | 1.377 | 1.057 | 1.047 | 0.020 | 0.788 | 0.600 |
| Paper 2 | DFNN | Casos (3 lags) + clima | 1.371 | 1.068 | 1.060 | 0.006 | 0.787 | 0.599 |
| Paper 4 | SARIMA, orden por AIC por distrito | Solo casos (serie) | 1.672 | 1.086 | — | — | 0.731 | 0.556 |
| Referencia | Regla: siempre cero | — | — | 1.174 | — | — | 0.534 | 0.406 |
| Referencia | Regla: último mes | — | — | 1.298 | — | — | 0.708 | 0.539 |
| Paper 4 | SARIMA (0,1,0)(1,1,0)12 del paper | Solo casos (serie) | 1.879 | 1.373 | — | — | 0.837 | 0.637 |
| Referencia | Regla: mismo mes año anterior | — | — | 1.471 | — | — | 0.828 | 0.630 |


`RMSE` usa la semilla 42; `RMSE_media_semillas` y `RMSE_sd_semillas` resumen 5 semillas (solo redes del paper 2; los modelos del equipo se entrenaron una vez y el SARIMA del paper 4 es determinista). Diferencias menores que la desviación no son interpretables. Las filas «+ clima» del protocolo oficial son idénticas a «solo casos» porque en 2019-2024 no hay clima observado y se usa el respaldo sin clima.

### Alerta (brote)

| origen | modelo | entradas | PR_AUC | ROC_AUC | EER | accuracy | recall | precision | FPR | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Paper 2 | DFNN | Solo casos (7) | 0.037 | 0.689 | 0.348 | 0.969 | 0.130 | 0.088 | 0.019 | 0.105 | 3 | 31 | 20 | 1602 |
| Equipo | MLP | Solo casos (7) | 0.037 | 0.649 | 0.391 | 0.938 | 0.174 | 0.046 | 0.051 | 0.073 | 4 | 83 | 19 | 1550 |
| Paper 4 | SARIMA (0,1,0)(1,1,0)12 del paper | Solo casos (serie) | 0.034 | 0.616 | 0.434 | 0.892 | 0.304 | 0.041 | 0.100 | 0.073 | 7 | 163 | 16 | 1470 |
| Paper 2 | SFNN2 | Solo casos (3 lags) | 0.033 | 0.637 | 0.395 | 0.973 | 0.087 | 0.077 | 0.015 | 0.082 | 2 | 24 | 21 | 1609 |
| Paper 2 | SFNN1 | Solo casos (3 lags) | 0.033 | 0.618 | 0.443 | 0.976 | 0.087 | 0.100 | 0.011 | 0.093 | 2 | 18 | 21 | 1615 |
| Paper 2 | SFNN1 | Solo casos (7) | 0.033 | 0.669 | 0.355 | 0.968 | 0.087 | 0.059 | 0.020 | 0.070 | 2 | 32 | 21 | 1601 |
| Paper 2 | DFNN | Solo casos (3 lags) | 0.033 | 0.637 | 0.395 | 0.972 | 0.087 | 0.074 | 0.015 | 0.080 | 2 | 25 | 21 | 1608 |
| Paper 2 | SFNN2 | Solo casos (7) | 0.032 | 0.680 | 0.394 | 0.952 | 0.130 | 0.048 | 0.036 | 0.071 | 3 | 59 | 20 | 1574 |
| Paper 2 | DFNN | Casos (3 lags) + clima | 0.032 | 0.624 | 0.434 | 0.978 | 0.043 | 0.062 | 0.009 | 0.051 | 1 | 15 | 22 | 1618 |
| Paper 2 | SFNN1 | Casos (7) + clima | 0.032 | 0.659 | 0.421 | 0.973 | 0.043 | 0.042 | 0.014 | 0.043 | 1 | 23 | 22 | 1610 |
| Equipo | CNN temporal | Solo casos (7) | 0.031 | 0.665 | 0.395 | 0.928 | 0.130 | 0.029 | 0.061 | 0.048 | 3 | 99 | 20 | 1534 |
| Paper 2 | SFNN1 | Casos (3 lags) + clima | 0.030 | 0.629 | 0.400 | 0.978 | 0.043 | 0.067 | 0.009 | 0.053 | 1 | 14 | 22 | 1619 |
| Equipo | Random Forest | Casos (7) + clima | 0.030 | 0.688 | 0.394 | 0.965 | 0.043 | 0.027 | 0.022 | 0.033 | 1 | 36 | 22 | 1597 |
| Paper 2 | SFNN2 | Casos (3 lags) + clima | 0.030 | 0.629 | 0.435 | 0.975 | 0.087 | 0.091 | 0.012 | 0.089 | 2 | 20 | 21 | 1613 |
| Equipo | Ridge / logística | Solo casos (7) | 0.030 | 0.600 | 0.438 | 0.943 | 0.130 | 0.038 | 0.046 | 0.059 | 3 | 75 | 20 | 1558 |
| Equipo | Ridge / logística | Casos (7) + clima | 0.030 | 0.595 | 0.479 | 0.944 | 0.087 | 0.027 | 0.043 | 0.042 | 2 | 71 | 21 | 1562 |
| Paper 2 | SFNN2 | Casos (7) + clima | 0.030 | 0.647 | 0.390 | 0.963 | 0.087 | 0.047 | 0.025 | 0.061 | 2 | 41 | 21 | 1592 |
| Equipo | CNN temporal | Casos (7) + clima | 0.026 | 0.677 | 0.352 | 0.925 | 0.130 | 0.028 | 0.064 | 0.046 | 3 | 104 | 20 | 1529 |
| Equipo | MLP | Casos (7) + clima | 0.026 | 0.631 | 0.434 | 0.935 | 0.130 | 0.033 | 0.054 | 0.053 | 3 | 88 | 20 | 1545 |
| Paper 2 | DFNN | Casos (7) + clima | 0.025 | 0.643 | 0.395 | 0.976 | 0.043 | 0.056 | 0.010 | 0.049 | 1 | 17 | 22 | 1616 |
| Paper 4 | SARIMA, orden por AIC por distrito | Solo casos (serie) | 0.023 | 0.634 | 0.445 | 0.631 | 0.435 | 0.016 | 0.366 | 0.032 | 10 | 598 | 13 | 1035 |
| Equipo | Árbol de decisión | Casos (7) + clima | 0.021 | 0.625 | 0.417 | 0.937 | 0.087 | 0.023 | 0.051 | 0.037 | 2 | 84 | 21 | 1549 |
| Equipo | Árbol de decisión | Solo casos (7) | 0.020 | 0.597 | 0.474 | 0.960 | 0.043 | 0.022 | 0.027 | 0.029 | 1 | 44 | 22 | 1589 |
| Equipo | Random Forest | Solo casos (7) | 0.020 | 0.596 | 0.435 | 0.962 | 0.000 | 0.000 | 0.024 | 0.000 | 0 | 40 | 23 | 1593 |

