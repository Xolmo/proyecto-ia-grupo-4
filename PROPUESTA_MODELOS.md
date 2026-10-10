# Propuesta de modelos

> **Borrador.** Falta incorporar los papers de los integrantes 3 y 4 y confirmar el alcance final (distritos, variables, definición de alerta).

## Problema

Anticipar los casos mensuales de leishmaniosis cutánea por distrito en el Perú (regresión) y detectar los meses de alerta (clasificación), usando el historial de casos del CDC-MINSA 2000–2024.

## Evidencia de la literatura

| Paper | Problema | Modelos | Qué aporta a nuestro proyecto |
|---|---|---|---|
| **Integrante 1** — Adams et al. (2025), *Environ. Res.: Health* | Leishmaniosis visceral en 113 municipios de Brasil | XGBoost en ventana deslizante (entrena 24 meses, valida 3); regresión de conteos + clasificación binaria; SHAP | Ventana deslizante, lags 1–12 sin fuga, umbral operativo definido por la autoridad sanitaria, doble enfoque regresión/clasificación |
| **Integrante 2** — Donizette, Rocco & de Queiroz (2025), *Oper. Res. Health Care* | Leishmaniosis visceral y tegumentaria en 6 ciudades de Brasil | SFNN y DFNN (MLP), LSTM, SVR; RMSE como métrica; también probaron RF, GB, XGBoost y LightGBM (peores) | Comparación MLP / LSTM / SVR, lags 1–3 de casos como entradas, evaluar el ajuste gráfico además del RMSE |
| **Integrante 3** | _pendiente_ | _pendiente_ | _pendiente_ |
| **Integrante 4** | _pendiente_ | _pendiente_ | _pendiente_ |

## Modelos a implementar

| # | Modelo | Tarea | Origen | Papel | Técnica del curso |
|---|---|---|---|---|---|
| B1 | Último valor observado (`lag1`) | Regresión y alerta | EDA del grupo | Baseline | — |
| B2 | Mismo mes del año anterior (`lag12`) | Regresión y alerta | EDA del grupo | Baseline | — |
| M1 | Regresión lineal / logística | Regresión / alerta | Papers 1 y 2 (modelos de referencia) | Modelo simple interpretable | Regresión lineal, regresión logística |
| M2 | Árbol de decisión | Regresión / alerta | Curso | Reglas interpretables | Árboles |
| M3 | KNN | Regresión / alerta | Curso | Comparación no paramétrica | KNN |
| M4 | MLP (1 y 2 capas ocultas) | Regresión / alerta | Paper 2 (SFNN, DFNN) | Modelo principal de red neuronal | MLP |
| M5 | LSTM | Regresión | Paper 2 | Secuencias temporales | RNN |

**Fuera de alcance por ahora:** SVR (paper 2) y XGBoost (paper 1) se usaron en la literatura, pero no están entre las técnicas vistas en clase que exigen las indicaciones de la TA. Se pueden retomar si el curso las cubre o si el profesor las autoriza.

## Diseño experimental propuesto

- **Variables:** `lag1`, `lag2`, `lag3`, `lag12`, `media3`, `mes_sin`, `mes_cos` (ya implementadas en `src/preprocessing.py`). Pendiente decidir si se agrega clima (ERA5/CHIRPS, ONI) y población del INEI, como hacen ambos papers.
- **Partición:** temporal, entrenamiento ≤ 2018 y prueba 2019–2024. Validación con ventana deslizante hacia adelante, nunca aleatoria.
- **Escalado:** solo con entrenamiento (`log1p` + estandarización para KNN, regresión logística y MLP; sin escalar para árboles).
- **Desbalance (alerta):** pesos de clase como opción base y SMOTE como comparación, solo en entrenamiento.
- **Métricas:**
  - Regresión: RMSE y MAE, comparados contra B1 y B2.
  - Alerta: recall, precisión, F1 y área bajo la curva precisión-recall (no accuracy).
  - Revisión gráfica de predicho vs. observado, porque el paper 2 muestra que un RMSE bajo puede ocultar un mal ajuste.
- **Sensibilidad:** repetir con y sin la semana 1 (`--exclude-week1`).

## Decisiones abiertas

1. Qué distritos se modelan (umbral de casos acumulados u otro criterio).
2. Definición final de alerta: la actual (media + 2 desviaciones) es provisional; los papers sugieren un umbral operativo de la autoridad sanitaria.
3. Si se incorporan variables climáticas y de población.
