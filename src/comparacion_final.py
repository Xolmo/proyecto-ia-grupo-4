"""Comparación final: modelos del equipo (V1), redes del paper 2 y SARIMA del paper 4.

Une, para cada protocolo (``oficial`` 2019-2024 e ``historico`` 2012-2014):

- los cinco modelos del equipo (``results/experimento_clima/``), y
- las redes SFNN1, SFNN2 y DFNN del paper 2 (``results/paper2*/``, ver ``modelo_paper2.py``), y
- el SARIMA del paper 4 (``results/paper4*/``, ver ``modelo_paper4.py``).

Regresión: RMSE, MAE y MASE de prueba; RMSE de validación (criterio de elección).
Alerta: PR-AUC, ROC-AUC, EER, accuracy, recall, precisión, FPR y F1, más la matriz
de confusión con el umbral elegido en validación. La alerta de las redes del paper
y del SARIMA usa los casos predichos como puntaje.

EER (equal error rate): tasa de error donde la de falsas alarmas (FPR) iguala a la
de brotes no detectados (FNR); menor es mejor. Accuracy se informa, pero con ~3 %
de alertas la regla «nunca alertar» ya obtiene ~97 %, por eso no se usa para elegir.

No modifica ningún archivo existente: escribe en ``results/comparacion_final/``.

Uso (desde la raíz del repositorio)::

    python src/comparacion_final.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/comparacion_final"
TEAM = ROOT / "results/experimento_clima"
PAPER2 = {"oficial": ROOT / "results/paper2", "historico": ROOT / "results/paper2_historico"}
PAPER4 = {"oficial": ROOT / "results/paper4", "historico": ROOT / "results/paper4_historico"}
P4_NAMES = {"SARIMA_paper": "SARIMA (0,1,0)(1,1,0)12 del paper", "SARIMA_AIC": "SARIMA, orden por AIC por distrito"}

FAMILIES = {"lineal": "Ridge / logística", "arbol": "Árbol de decisión", "random_forest": "Random Forest",
            "mlp": "MLP", "cnn": "CNN temporal"}
TEAM_VARIANTS = {"solo_casos": "Solo casos (7)", "sin_imputar": "Casos (7) + clima"}
P2_INPUTS = {"casos3": "Solo casos (3 lags)", "casos3_clima": "Casos (3 lags) + clima",
             "casos7": "Solo casos (7)", "casos7_clima": "Casos (7) + clima"}


def eer(y, score):
    """Equal error rate: punto donde FPR = FNR (promedio de ambos en el cruce)."""
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def alert_metrics(y, score, threshold):
    """Métricas de clasificación de la alerta con el umbral fijado en validación."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(PR_AUC=float(average_precision_score(y, score)), ROC_AUC=float(roc_auc_score(y, score)),
                EER=eer(y, score), accuracy=float((tp + tn) / len(y)),
                recall=float(tp / max(tp + fn, 1)), precision=float(tp / max(tp + fp, 1)),
                FPR=float(fp / max(fp + tn, 1)), F1=float(f1_score(y, pred, zero_division=0)),
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def team_rows(protocol):
    metr = pd.read_csv(TEAM / "metricas.csv")
    metr = metr[metr.protocol == protocol]
    pred = pd.read_csv(TEAM / "predicciones.csv", dtype={"ubigeo": str})
    pred = pred[pred.protocol == protocol]
    rows = []
    for family, fname in FAMILIES.items():
        for variant, vname in TEAM_VARIANTS.items():
            sel = metr[(metr.family == family) & (metr.variant == variant)]
            c, a = sel[sel.task == "casos"].iloc[0], sel[sel.task == "alerta"].iloc[0]
            p = pred[(pred.family == family) & (pred.variant == variant) & (pred.task == "alerta")]
            rows.append(dict(origen="Equipo", modelo=fname, entradas=vname,
                             RMSE_validacion=c.validation_score, RMSE=c.RMSE, MAE=c.MAE, MASE=c.MASE,
                             RMSE_media_semillas=np.nan, RMSE_sd_semillas=np.nan, uso_clima_prueba=c.climate_fraction,
                             **alert_metrics(p.observed, p.predicted, a.threshold)))
    return rows


def paper2_rows(protocol):
    folder = PAPER2[protocol]
    metr = pd.read_csv(folder / "metricas.csv")
    pred = pd.read_csv(folder / "predicciones.csv", dtype={"ubigeo": str})
    rows = []
    for _, r in metr.iterrows():
        p = pred[(pred.modelo == r.modelo) & (pred.entradas == r.entradas)]
        rows.append(dict(origen="Paper 2", modelo=r.modelo, entradas=P2_INPUTS[r.entradas],
                         RMSE_validacion=r.RMSE_validacion, RMSE=r.RMSE, MAE=r.MAE, MASE=r.MASE,
                         RMSE_media_semillas=r.RMSE_5semillas_media, RMSE_sd_semillas=r.RMSE_5semillas_sd, uso_clima_prueba=r.uso_clima_prueba,
                         **alert_metrics(p.alerta_observada, p.casos_predichos, p.umbral_alerta.iloc[0])))
    return rows


def paper4_rows(protocol):
    folder = PAPER4[protocol]
    if not (folder / "predicciones.csv").exists():
        print(f"Falta {folder}; se omite el SARIMA del paper 4 en {protocol}")
        return []
    metr = pd.read_csv(folder / "metricas.csv")
    pred = pd.read_csv(folder / "predicciones.csv", dtype={"ubigeo": str})
    rows = []
    for _, r in metr.iterrows():
        p = pred[pred.modelo == r.modelo]
        rows.append(dict(origen="Paper 4", modelo=P4_NAMES[r.modelo], entradas="Solo casos (serie)",
                         RMSE_validacion=r.RMSE_validacion, RMSE=r.RMSE, MAE=r.MAE, MASE=r.MASE,
                         RMSE_media_semillas=np.nan, RMSE_sd_semillas=np.nan, uso_clima_prueba=0.0,
                         **alert_metrics(p.alerta_observada, p.casos_predichos, p.umbral_alerta.iloc[0])))
    return rows


def references(protocol):
    metr = pd.read_csv(TEAM / "metricas.csv")
    ref = metr[(metr.protocol == protocol) & (metr.variant == "referencia")]
    names = {"ultimo_mes": "Regla: último mes", "estacional": "Regla: mismo mes año anterior",
             "cero": "Regla: siempre cero"}
    return pd.DataFrame([dict(origen="Referencia", modelo=names[r.family], entradas="—",
                              RMSE=r.RMSE, MAE=r.MAE, MASE=r.MASE) for _, r in ref.iterrows()])


def best_by_validation(table):
    """Mejor modelo por RMSE de validación, sin mirar la prueba."""
    t = table[table.origen != "Referencia"]
    return t.sort_values("RMSE_validacion", kind="stable").iloc[0]


def plot(table, refs, protocol):
    """RMSE de prueba (con ±1 sd entre semillas en el paper 2) y PR-AUC de la alerta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Equipo": "#4c78a8", "Paper 2": "#e8833a", "Paper 4": "#54a24b"}
    t = table.sort_values("RMSE", ascending=False).reset_index(drop=True)
    labels_ = [f"{r.modelo} · {r.entradas}" for r in t.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.32 * len(t) + 2.2), sharey=True)
    ax = axes[0]
    ax.barh(range(len(t)), t.RMSE, xerr=t.RMSE_sd_semillas.fillna(0), color=[colors[o] for o in t.origen],
            capsize=2)
    for r in refs.itertuples():
        if r.modelo in ("Regla: siempre cero", "Regla: último mes"):
            ax.axvline(r.RMSE, ls="--", lw=1, color="gray")
            ax.text(r.RMSE, len(t) - 0.4, r.modelo.replace("Regla: ", ""), rotation=90, va="top", ha="right",
                    fontsize=8, color="gray")
    ax.set_xlim(t.RMSE.min() * 0.9, max(refs.RMSE.max(), t.RMSE.max()) * 1.02)
    ax.set_yticks(range(len(t)), labels_, fontsize=8)
    ax.set_xlabel("RMSE de prueba (menor es mejor)")
    ax = axes[1]
    ax.barh(range(len(t)), t.PR_AUC, color=[colors[o] for o in t.origen])
    ax.set_xlabel("PR-AUC de la alerta (mayor es mejor)")
    present = [o for o in colors if o in set(t.origen)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[o]) for o in present]
    fig.legend(handles, present, loc="upper right")
    fig.suptitle(f"Comparación de modelos, protocolo {protocol}")
    fig.tight_layout()
    fig.savefig(OUT / f"comparacion_{protocol}.png", dpi=130)
    plt.close(fig)


def to_markdown(df, cols, fmt):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(("—" if pd.isna(r[c]) else fmt.get(c, "{}").format(r[c])) for c in cols) + " |\n"
    return head + body


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fmt = {c: "{:.3f}" for c in ["RMSE_validacion", "RMSE", "MAE", "MASE", "PR_AUC", "ROC_AUC", "EER",
                                  "accuracy", "recall", "precision", "FPR", "F1", "RMSE_media_semillas", "RMSE_sd_semillas"]}
    md = ["# Comparación final de modelos (equipo + paper 2 + paper 4)", "",
          "Generado por `src/comparacion_final.py`. Prueba oficial: 2019-2024 (57 distritos, 4 104 filas, 120 alertas). "
          "Prueba histórica: 2012-2014 (46 distritos, 1 656 filas), con clima observado.", ""]
    for protocol in PAPER2:
        if not (PAPER2[protocol] / "predicciones.csv").exists():
            print(f"Falta {PAPER2[protocol]}; se omite el protocolo {protocol}")
            continue
        table = pd.DataFrame(team_rows(protocol) + paper2_rows(protocol) + paper4_rows(protocol))
        full = pd.concat([table, references(protocol)], ignore_index=True)
        full.insert(0, "protocolo", protocol)
        full.to_csv(OUT / f"metricas_{protocol}.csv", index=False)
        plot(table, references(protocol), protocol)
        best = best_by_validation(table)
        md += [f"## Protocolo {protocol}", "",
               f"**Mejor por RMSE de validación:** {best.origen} · {best.modelo} · {best.entradas} "
               f"(validación {best.RMSE_validacion:.3f}, prueba {best.RMSE:.3f}).", "",
               "### Regresión (casos del mes siguiente)", "",
               to_markdown(full.sort_values("RMSE"), ["origen", "modelo", "entradas", "RMSE_validacion", "RMSE",
                                                      "RMSE_media_semillas", "RMSE_sd_semillas", "MAE", "MASE"], fmt), "",
               "`RMSE` usa la semilla 42; `RMSE_media_semillas` y `RMSE_sd_semillas` resumen 5 semillas "
               "(solo redes del paper 2; los modelos del equipo se entrenaron una vez y el SARIMA del paper 4 es "
               "determinista). Diferencias menores "
               "que la desviación no son interpretables. Las filas «+ clima» del protocolo oficial son idénticas "
               "a «solo casos» porque en 2019-2024 no hay clima observado y se usa el respaldo sin clima.", "",
               "### Alerta (brote)", "",
               to_markdown(table.sort_values("PR_AUC", ascending=False),
                           ["origen", "modelo", "entradas", "PR_AUC", "ROC_AUC", "EER", "accuracy", "recall",
                            "precision", "FPR", "F1", "TP", "FP", "FN", "TN"], fmt), ""]
        print(f"{protocol}: mejor por validación -> {best.origen} {best.modelo} {best.entradas} "
              f"(val {best.RMSE_validacion:.3f}, prueba {best.RMSE:.3f})")
    (OUT / "tabla_comparativa_final.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
