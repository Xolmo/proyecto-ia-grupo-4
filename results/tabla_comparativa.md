# Tabla comparativa de modelos — protocolo oficial

Prueba 2019–2024, 57 distritos de Cajamarca, 4 104 filas distrito × mes. Ajuste 2001–2015 y validación 2016–2018. ↓ menor es mejor; ↑ mayor es mejor.

| Origen | Modelo | Entradas | RMSE ↓ | MAE ↓ | MASE ↓ | PR-AUC ↑ | F1 ↑ | Recall ↑ | Precisión ↑ | Alertas detectadas | Falsas alarmas |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equipo | Lineal (Ridge / Logística) | Solo casos | 0,940 | 0,583 | 0,567 | 0,107 | 0,154 | 39 % | 10 % | 47/120 | 443 |
| Equipo | Lineal (Ridge / Logística) | Casos + clima | 0,940 | 0,584 | 0,568 | 0,112 | 0,071 | 62 % | 4 % | 74/120 | 1890 |
| Equipo | Árbol de decisión | Solo casos | 0,941 | 0,611 | 0,594 | 0,093 | 0,158 | 36 % | 10 % | 43/120 | 380 |
| Equipo | Árbol de decisión | Casos + clima | 0,944 | 0,613 | 0,596 | 0,135 | 0,194 | 34 % | 14 % | 41/120 | 261 |
| Equipo | Random Forest | Solo casos | 0,926 | 0,598 | 0,581 | 0,120 | 0,140 | 55 % | 8 % | 66/120 | 759 |
| Equipo | Random Forest | Casos + clima | 0,923 | 0,596 | 0,580 | 0,154 | 0,154 | 59 % | 9 % | 71/120 | 734 |
| Equipo | Red neuronal MLP | Solo casos | 0,922 | 0,575 | 0,559 | 0,121 | 0,147 | 54 % | 9 % | 65/120 | 699 |
| Equipo | Red neuronal MLP | Casos + clima | 0,919 | 0,573 | 0,557 | 0,132 | 0,132 | 49 % | 8 % | 59/120 | 713 |
| Equipo | Red convolucional CNN | Solo casos | 0,938 | 0,621 | 0,604 | 0,112 | 0,142 | 42 % | 9 % | 50/120 | 532 |
| Equipo | Red convolucional CNN | Casos + clima | 0,941 | 0,632 | 0,615 | 0,104 | 0,159 | 19 % | 14 % | 23/120 | 146 |
| Paper 2 | SFNN1 (1 capa oculta) | Solo casos (3 rezagos, como el paper) | 0,976 | 0,701 | 0,681 | 0,125 | 0,137 | 60 % | 8 % | 72/120 | 858 |
| Paper 2 | SFNN2 (1 capa oculta) | Solo casos (3 rezagos, como el paper) | 0,983 | 0,710 | 0,690 | 0,124 | 0,138 | 54 % | 8 % | 65/120 | 754 |
| Paper 2 | DFNN (2 capas ocultas) | Solo casos (3 rezagos, como el paper) | 1,002 | 0,749 | 0,728 | 0,124 | 0,107 | 70 % | 6 % | 84/120 | 1363 |
| Paper 2 | SFNN1 (1 capa oculta) | Solo casos (7 variables del equipo) | 0,971 | 0,704 | 0,685 | 0,137 | 0,151 | 55 % | 9 % | 66/120 | 688 |
| Paper 2 | SFNN2 (1 capa oculta) | Solo casos (7 variables del equipo) | 0,960 | 0,687 | 0,668 | 0,138 | 0,150 | 56 % | 9 % | 67/120 | 709 |
| Paper 2 | DFNN (2 capas ocultas) | Solo casos (7 variables del equipo) | 0,916 | 0,602 | 0,585 | 0,158 | 0,164 | 58 % | 10 % | 70/120 | 664 |
| Paper 4 | SARIMA (0,1,0)(1,1,0)12 del paper | Solo casos (serie por distrito) | 1,246 | 0,669 | 0,650 | 0,141 | 0,145 | 42 % | 9 % | 50/120 | 518 |
| Paper 4 | SARIMA, orden por AIC por distrito | Solo casos (serie por distrito) | 0,946 | 0,556 | 0,541 | 0,083 | 0,103 | 55 % | 6 % | 66/120 | 1092 |
| Referencia | Ref.: último mes | Regla fija | 1,187 | 0,561 | 0,545 | — | — | — | — | — | — |
| Referencia | Ref.: mismo mes año anterior | Regla fija | 1,252 | 0,617 | 0,600 | — | — | — | — | — | — |
| Referencia | Ref.: siempre cero | Regla fija | 1,100 | 0,432 | 0,420 | — | — | — | — | — | — |
| Referencia | Ref.: clasificador al azar | Regla fija | — | — | — | 0,029 | — | — | — | — | — |

**Mejor del equipo por validación:** casos → Random Forest (RMSE de validación 1,1270); alertas → Random Forest (PR-AUC de validación 0,1018). La variante elegida no usó clima en prueba, así que equivale a la fila «Solo casos».

**Modelo final (menor RMSE de validación entre todos los modelos, equipo + papers):** Paper 2 · DFNN (2 capas ocultas), Solo casos (7 variables del equipo) (RMSE de validación 1,1148).

**Mejor valor observado en prueba (descriptivo):** RMSE 0,9157 con Paper 2 · DFNN (2 capas ocultas) (Solo casos (7 variables del equipo)); PR-AUC 0,1584 con Paper 2 · DFNN (2 capas ocultas) (Solo casos (7 variables del equipo)). No se usa para elegir modelo porque la prueba ya fue observada.

Notas:
- Las filas «Paper 2» y «Paper 4» son las réplicas de las redes SFNN/DFNN y del SARIMA, entrenadas con la misma partición y evaluadas en las mismas filas. Sus variantes con clima no tuvieron clima en prueba y no se repiten. `RMSE_media_semillas` ± `RMSE_sd_semillas` resume 5 semillas de las redes del paper 2.
- El XGBoost del paper 1 se evaluó en otro conjunto (70 distritos, 5 040 filas) y se resume en `model_comparison.txt`, no en esta tabla.
- RMSE/MAE/MASE miden el error en casos por distrito y mes; MASE divide el MAE entre el error medio que tenía la regla «mismo mes del año anterior» en el entrenamiento (menor que 1 = mejor que esa regla).
- PR-AUC es la precisión media (`average_precision_score`); el clasificador al azar obtiene la prevalencia de alertas (0,029).
- Alerta = mes con casos mayores que la media distrital + 2 desviaciones estándar del entrenamiento.
- «Casos + clima» usa la imputación estacional del 52 %; en prueba solo el 18 % de las predicciones tuvo clima completo. Las variantes sin imputar y de huecos cortos no tuvieron clima en prueba y son idénticas a «Solo casos».
