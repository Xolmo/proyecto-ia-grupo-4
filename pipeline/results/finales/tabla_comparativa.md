# Tabla comparativa de modelos — protocolo oficial

Prueba 2019–2024, 57 distritos de Cajamarca, 4 104 filas distrito × mes. Ajuste 2001–2015 y validación 2016–2018. ↓ menor es mejor; ↑ mayor es mejor.

| Modelo | Entradas | RMSE ↓ | MAE ↓ | MASE ↓ | PR-AUC ↑ | F1 ↑ | Recall ↑ | Precisión ↑ | Alertas detectadas | Falsas alarmas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lineal (Ridge / Logística) | Solo casos | 0,940 | 0,583 | 0,567 | 0,107 | 0,154 | 39 % | 10 % | 47/120 | 443 |
| Lineal (Ridge / Logística) | Casos + clima | 0,940 | 0,584 | 0,568 | 0,112 | 0,071 | 62 % | 4 % | 74/120 | 1890 |
| Árbol de decisión | Solo casos | 0,941 | 0,611 | 0,594 | 0,093 | 0,158 | 36 % | 10 % | 43/120 | 380 |
| Árbol de decisión | Casos + clima | 0,944 | 0,613 | 0,596 | 0,135 | 0,194 | 34 % | 14 % | 41/120 | 261 |
| Random Forest | Solo casos | 0,926 | 0,598 | 0,581 | 0,120 | 0,140 | 55 % | 8 % | 66/120 | 759 |
| Random Forest | Casos + clima | 0,923 | 0,596 | 0,580 | 0,154 | 0,154 | 59 % | 9 % | 71/120 | 734 |
| Red neuronal MLP | Solo casos | 0,922 | 0,575 | 0,559 | 0,121 | 0,147 | 54 % | 9 % | 65/120 | 699 |
| Red neuronal MLP | Casos + clima | 0,919 | 0,573 | 0,557 | 0,132 | 0,132 | 49 % | 8 % | 59/120 | 713 |
| Red convolucional CNN | Solo casos | 0,938 | 0,621 | 0,604 | 0,112 | 0,142 | 42 % | 9 % | 50/120 | 532 |
| Red convolucional CNN | Casos + clima | 0,941 | 0,632 | 0,615 | 0,104 | 0,159 | 19 % | 14 % | 23/120 | 146 |
| Ref.: último mes | Regla fija | 1,187 | 0,561 | 0,545 | — | — | — | — | — | — |
| Ref.: mismo mes año anterior | Regla fija | 1,252 | 0,617 | 0,600 | — | — | — | — | — | — |
| Ref.: siempre cero | Regla fija | 1,100 | 0,432 | 0,420 | — | — | — | — | — | — |
| Ref.: clasificador al azar | Regla fija | — | — | — | 0,029 | — | — | — | — | — |

**Selección por validación (criterio del pipeline):** casos → Random Forest (RMSE de validación 1,1270); alertas → Random Forest (PR-AUC de validación 0,1018). La variante elegida no usó clima en prueba, así que equivale a la fila «Solo casos».

**Mejor valor observado en prueba (descriptivo):** RMSE 0,9188 con Red neuronal MLP (Casos + clima); PR-AUC 0,1541 con Random Forest (Casos + clima). No se usa para elegir modelo porque la prueba ya fue observada.

Notas:
- RMSE/MAE/MASE miden el error en casos por distrito y mes; MASE divide el MAE entre el error medio que tenía la regla «mismo mes del año anterior» en el entrenamiento (menor que 1 = mejor que esa regla).
- PR-AUC es la precisión media (`average_precision_score`); el clasificador al azar obtiene la prevalencia de alertas (0,029).
- Alerta = mes con casos mayores que la media distrital + 2 desviaciones estándar del entrenamiento.
- «Casos + clima» usa la imputación estacional del 52 %; en prueba solo el 18 % de las predicciones tuvo clima completo. Las variantes sin imputar y de huecos cortos no tuvieron clima en prueba y son idénticas a «Solo casos».
