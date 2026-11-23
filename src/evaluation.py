"""Métricas y evaluación de los modelos.

- ``labels``: define la etiqueta de alerta (brote) con estadísticas del entrenamiento.
- ``choose_threshold``: elige el umbral de alerta que maximiza F1 en validación.
- ``metrics``: calcula las métricas de prueba de cada tarea.

Regresión (``casos``): RMSE, MAE y MASE. MASE divide el MAE entre el error
medio de la regla «mismo mes del año anterior» en el entrenamiento
(Hyndman y Koehler, 2006); menor que 1 = mejor que esa regla.

Clasificación (``alerta``): precisión media (PR-AUC, ``average_precision_score``),
F1, recall, precisión, ROC-AUC y conteos de alertas reales, emitidas y falsas.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def labels(d, train):
    """1 si los casos del mes superan media + 2 desviaciones estándar del distrito.

    La media y la desviación se calculan solo con ``train`` para no usar
    información del periodo que se evalúa.
    """
    stats = train.groupby("ubigeo").casos.agg(["mean", "std"])
    threshold = d.ubigeo.map(stats["mean"] + 2 * stats["std"])
    return (d.casos > threshold).astype(int).to_numpy()


def choose_threshold(y, p):
    """Umbral de probabilidad que maximiza F1 (se aplica sobre validación)."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])]) if len(thresholds) else .5


def metrics(task, y, p, threshold, scale):
    """Métricas de prueba.

    Args:
        task: ``"casos"`` o ``"alerta"``.
        y: valores observados (conteos o etiquetas 0/1).
        p: predicciones (conteos o probabilidades).
        threshold: umbral de alerta elegido en validación (solo clasificación).
        scale: MAE de la regla estacional en entrenamiento, para MASE.
    """
    if task == "casos":
        mae = mean_absolute_error(y, p)
        return dict(RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=float(mae), MASE=float(mae / scale))
    pred = p >= threshold
    return dict(PR_AUC=float(average_precision_score(y, p)), F1=float(f1_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                ROC_AUC=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                true_alerts=int(np.sum(y)), predicted_alerts=int(pred.sum()),
                false_alerts=int(np.sum(pred & (y == 0))))
