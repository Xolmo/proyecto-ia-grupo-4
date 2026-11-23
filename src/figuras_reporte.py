"""Figuras 2 y 3 del reporte final, considerando todos los modelos.

El modelo final es el de menor RMSE de validación entre todos los modelos
evaluados con el protocolo oficial (5 del equipo y réplicas de los papers 2 y 4):
la DFNN del paper 2 con las 7 variables de casos.

Escribe en results/plots/:
  - fig2_observado_vs_predicho_dfnn.png: casos mensuales totales, observados y
    predichos por la DFNN (prueba 2019-2024, 57 distritos).
  - fig3_curvas_pr_todos.png: curvas precisión-recall de la alerta de brote de
    cinco modelos representativos (equipo y papers).

Ejecutar desde la raíz: python src/figuras_reporte.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PLOTS = RES / "plots"

# Misma paleta y estilo que src/resultados_finales.py
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
    "axes.axisbelow": True, "legend.frameon": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def coma(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def miles(x):
    return f"{x:,.0f}".replace(",", "\u202f")


def load():
    """Devuelve {nombre: (casos DataFrame, alerta DataFrame)} con columnas fecha_mes, observed, predicted."""
    e = pd.read_csv(RES / "experimento_clima/predicciones.csv", dtype={"ubigeo": str}, parse_dates=["fecha_mes"])
    e = e[(e.protocol == "oficial") & (e.variant == "solo_casos")]
    out = {}
    for fam, name in [("random_forest", "Random Forest"), ("mlp", "MLP")]:
        g = e[e.family == fam]
        out[name] = (g[g.task == "casos"], g[g.task == "alerta"])
    p2 = pd.read_csv(RES / "paper2/predicciones.csv", dtype={"ubigeo": str}, parse_dates=["fecha_mes"])
    p4 = pd.read_csv(RES / "paper4/predicciones.csv", dtype={"ubigeo": str}, parse_dates=["fecha_mes"])
    for df, mod, ent, name in [(p2, "DFNN", "casos7", "P2 · DFNN (7 var.)"),
                               (p2, "SFNN2", "casos7", "P2 · SFNN2 (7 var.)"),
                               (p4, "SARIMA_AIC", "casos", "P4 · SARIMA por AIC")]:
        g = df[(df.modelo == mod) & (df.entradas == ent)]
        casos = g.rename(columns={"casos_observados": "observed", "casos_predichos": "predicted"})
        # Los papers solo hacen regresión: la alerta usa los casos predichos como puntaje.
        alerta = g.rename(columns={"alerta_observada": "observed", "casos_predichos": "predicted"})
        out[name] = (casos, alerta)
    return out


def fig_observed(models):
    casos = models["P2 · DFNN (7 var.)"][0]
    obs = casos.groupby("fecha_mes")["observed"].sum()
    pred = casos.groupby("fecha_mes")["predicted"].sum()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(obs.index, obs.values, color=INK, lw=1.8, label=f"Observado ({miles(obs.sum())} casos)")
    ax.plot(pred.index, pred.values, color="#2a78d6", lw=1.8,
            label=f"DFNN, 7 variables ({miles(pred.sum())} casos)")
    ax.set_title("Casos mensuales totales, 57 distritos (prueba 2019–2024)")
    ax.set_ylabel("casos por mes")
    ax.set_ylim(0, None)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.32), ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig2_observado_vs_predicho_dfnn.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pr(models):
    order = [("P2 · DFNN (7 var.)", "#2a78d6"), ("P2 · SFNN2 (7 var.)", "#eb6834"),
             ("MLP", "#eda100"), ("Random Forest", "#1baf7a"), ("P4 · SARIMA por AIC", "#e87ba4")]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    prevalence = None
    for name, color in order:
        g = models[name][1]
        prec, rec, _ = precision_recall_curve(g.observed, g.predicted)
        ap = average_precision_score(g.observed, g.predicted)
        ax.step(rec, prec, where="post", color=color, lw=1.8, label=f"{name} (PR-AUC {coma(ap)})")
        prevalence = g.observed.mean()
    ax.axhline(prevalence, color=INK2, ls="--", lw=1.2, label=f"Al azar (PR-AUC {coma(prevalence)})")
    ax.set_xlabel("Recall (alertas reales detectadas)")
    ax.set_ylabel("Precisión (alertas emitidas reales)")
    ax.set_title("Detección de alertas — prueba 2019–2024")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(.6, ax.get_ylim()[1]))
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig3_curvas_pr_todos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    models = load()
    fig_observed(models)
    fig_pr(models)
    print("Figuras escritas en", PLOTS)


if __name__ == "__main__":
    main()
