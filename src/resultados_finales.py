"""Tabla comparativa y gráficos del entregable final.

Lee los resultados ya generados por `ejecutar.py --modo experimento-v1` (protocolo
oficial: ajuste 2001-2015, validación 2016-2018, prueba 2019-2024, 57 distritos)
y escribe en results/:

- metrics.csv / tabla_comparativa.md: 5 modelos del equipo x {solo casos, con clima},
  redes del paper 2 y SARIMA del paper 4 (mismas filas de prueba) + referencias.
- model_comparison.txt: resumen de resultados y análisis detallado, incluido el
  XGBoost del paper 1 (evaluado aparte, en 70 distritos).
- plots/ (además de los gráficos del EDA y del árbol original):
  - fig1_error_por_modelo.png: RMSE y MAE por modelo frente a las referencias.
  - fig2_observado_vs_predicho.png: casos mensuales totales observados y predichos.
  - fig3_curvas_precision_recall.png: detección de alertas por modelo.
  - fig4_importancia_variables_rf.png: importancia de variables del Random Forest.
  - fig5_casos_por_distrito.png: error del mejor modelo por distrito.

Ejecutar desde la raíz: python src/resultados_finales.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

import comparacion_final as cf
import experimento_clima as ex
import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/experimento_clima"
OUT = ROOT / "results"
PLOTS = OUT / "plots"
PROTOCOL = "oficial"

MODELS = ["lineal", "arbol", "random_forest", "mlp", "cnn"]
NAMES = {"lineal": "Lineal (Ridge / Logística)", "arbol": "Árbol de decisión",
         "random_forest": "Random Forest", "mlp": "Red neuronal MLP", "cnn": "Red convolucional CNN",
         "ultimo_mes": "Ref.: último mes", "estacional": "Ref.: mismo mes año anterior",
         "cero": "Ref.: siempre cero"}
SHORT = {"lineal": "Lineal", "arbol": "Árbol", "random_forest": "Random Forest",
         "mlp": "MLP", "cnn": "CNN"}
VARIANTS = {"solo_casos": "Solo casos", "estacional_52": "Casos + clima"}
# Papers replicados que comparten protocolo y filas de prueba con el equipo.
# Las variantes «+ clima» del paper 2 no tuvieron clima en 2019-2024 (son
# idénticas a «solo casos»), por eso solo se conservan las de casos.
P2_INPUTS = {"Solo casos (3 lags)": "Solo casos (3 rezagos, como el paper)",
             "Solo casos (7)": "Solo casos (7 variables del equipo)"}
P2_NAMES = {"SFNN1": "SFNN1 (1 capa oculta)", "SFNN2": "SFNN2 (1 capa oculta)",
            "DFNN": "DFNN (2 capas ocultas)"}
P1_RESULTS = ROOT / "results/p1_adams"
COLUMNS = ["origen", "modelo", "entradas", "RMSE", "MAE", "MASE", "RMSE_validacion", "PR_AUC", "F1",
           "recall", "precision", "alertas_detectadas", "falsas_alarmas", "uso_clima_prueba",
           "RMSE_media_semillas", "RMSE_sd_semillas"]
# Paleta categórica validada (validate_palette.js, modo claro): orden fijo.
COLOR = {"lineal": "#2a78d6", "arbol": "#eb6834", "random_forest": "#1baf7a",
         "mlp": "#eda100", "cnn": "#e87ba4"}
VCOLOR = {"solo_casos": "#2a78d6", "estacional_52": "#eb6834"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
    "axes.axisbelow": True, "legend.frameon": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load():
    m = pd.read_csv(RES / "metricas.csv")
    m = m[m.protocol == PROTOCOL]
    sel = pd.read_csv(RES / "seleccion_por_validacion.csv")
    sel = sel[sel.protocol == PROTOCOL].copy()
    # Si la variante ganadora no usó clima en prueba, es idéntica a «solo casos».
    sel.loc[sel.climate_fraction == 0, "variant"] = "solo_casos"
    return m, sel


def paper_rows():
    """Redes del paper 2 y SARIMA del paper 4, con las mismas métricas y filas de prueba.

    Usa las funciones de ``comparacion_final`` para que ambas tablas coincidan.
    """
    rows = []
    p2 = cf.paper2_rows(PROTOCOL) if (cf.PAPER2[PROTOCOL] / "predicciones.csv").exists() else []
    for r in p2 + cf.paper4_rows(PROTOCOL):
        if r["origen"] == "Paper 2":
            if r["entradas"] not in P2_INPUTS:
                continue
            modelo, entradas = P2_NAMES[r["modelo"]], P2_INPUTS[r["entradas"]]
        else:
            modelo, entradas = r["modelo"], "Solo casos (serie por distrito)"
        rows.append({"origen": r["origen"], "modelo": modelo, "entradas": entradas,
                     "RMSE": r["RMSE"], "MAE": r["MAE"], "MASE": r["MASE"],
                     "RMSE_validacion": r["RMSE_validacion"],
                     "PR_AUC": r["PR_AUC"], "F1": r["F1"], "recall": r["recall"], "precision": r["precision"],
                     "alertas_detectadas": f"{r['TP']}/{r['TP'] + r['FN']}", "falsas_alarmas": r["FP"],
                     "uso_clima_prueba": r["uso_clima_prueba"],
                     "RMSE_media_semillas": r["RMSE_media_semillas"], "RMSE_sd_semillas": r["RMSE_sd_semillas"]})
    if not p2:
        print(f"Falta {cf.PAPER2[PROTOCOL]}; las redes del paper 2 no entran en metrics.csv")
    return rows


def comparison_table(m, sel):
    rows = []
    for family in MODELS:
        for variant, vname in VARIANTS.items():
            c = m[(m.family == family) & (m.variant == variant) & (m.task == "casos")].iloc[0]
            a = m[(m.family == family) & (m.variant == variant) & (m.task == "alerta")].iloc[0]
            rows.append({"origen": "Equipo", "modelo": NAMES[family], "entradas": vname,
                         "RMSE": c.RMSE, "MAE": c.MAE, "MASE": c.MASE,
                         "RMSE_validacion": c.validation_score,
                         "PR_AUC": a.PR_AUC, "F1": a.F1, "recall": a.recall, "precision": a.precision,
                         "alertas_detectadas": f"{int(round(a.recall * a.true_alerts))}/{int(a.true_alerts)}",
                         "falsas_alarmas": int(a.false_alerts),
                         "uso_clima_prueba": c.climate_fraction})
    rows += paper_rows()
    for family in ["ultimo_mes", "estacional", "cero"]:
        c = m[(m.family == family) & (m.variant == "referencia")].iloc[0]
        rows.append({"origen": "Referencia", "modelo": NAMES[family], "entradas": "Regla fija",
                     "RMSE": c.RMSE, "MAE": c.MAE, "MASE": c.MASE})
    alerts = m[(m.task == "alerta")].iloc[0]
    prevalence = alerts.true_alerts / alerts.n_test
    rows.append({"origen": "Referencia", "modelo": "Ref.: clasificador al azar", "entradas": "Regla fija",
                 "PR_AUC": prevalence})
    t = pd.DataFrame(rows).reindex(columns=COLUMNS)
    t["falsas_alarmas"] = t["falsas_alarmas"].astype("Int64")
    PLOTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "metrics.csv", index=False)

    # Ganadores por validación (selección honesta) y por prueba (descriptivo).
    winners = {r.task: r for r in sel.itertuples()}
    best_all = best_by_validation(t)
    best_test_rmse = t.loc[t.RMSE.idxmin()]
    best_test_ap = t.loc[t.PR_AUC.idxmax()]

    def f(x, d=3):
        return "—" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")

    def pct(x):
        return "—" if pd.isna(x) else f"{100 * x:.0f} %"

    md = ["# Tabla comparativa de modelos — protocolo oficial", "",
          "Prueba 2019–2024, 57 distritos de Cajamarca, 4 104 filas distrito × mes. "
          "Ajuste 2001–2015 y validación 2016–2018. ↓ menor es mejor; ↑ mayor es mejor.", "",
          "| Origen | Modelo | Entradas | RMSE ↓ | MAE ↓ | MASE ↓ | PR-AUC ↑ | F1 ↑ | Recall ↑ | Precisión ↑ | Alertas detectadas | Falsas alarmas |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples():
        md.append(f"| {r.origen} | {r.modelo} | {r.entradas} | {f(r.RMSE)} | {f(r.MAE)} | {f(r.MASE)} | {f(r.PR_AUC)} | "
                  f"{f(r.F1)} | {pct(r.recall)} | {pct(r.precision)} | "
                  f"{'—' if pd.isna(r.alertas_detectadas) else r.alertas_detectadas} | "
                  f"{'—' if pd.isna(r.falsas_alarmas) else int(r.falsas_alarmas)} |")
    wc, wa = winners["casos"], winners["alerta"]
    md += ["",
           "**Mejor del equipo por validación:** "
           f"casos → {NAMES[wc.family]} (RMSE de validación {f(wc.validation_score, 4)}); "
           f"alertas → {NAMES[wa.family]} (PR-AUC de validación {f(wa.validation_score, 4)}). "
           "La variante elegida no usó clima en prueba, así que equivale a la fila «Solo casos».", "",
           f"**Modelo final (menor RMSE de validación entre todos los modelos, equipo + papers):** {best_all.origen} · "
           f"{best_all.modelo}, {best_all.entradas} (RMSE de validación {f(best_all.RMSE_validacion, 4)}).", "",
           f"**Mejor valor observado en prueba (descriptivo):** RMSE {f(best_test_rmse.RMSE, 4)} con "
           f"{best_test_rmse.origen} · {best_test_rmse.modelo} ({best_test_rmse.entradas}); PR-AUC "
           f"{f(best_test_ap.PR_AUC, 4)} con {best_test_ap.origen} · {best_test_ap.modelo} ({best_test_ap.entradas}). "
           "No se usa para elegir modelo porque la prueba ya fue observada.", "",
           "Notas:",
           "- Las filas «Paper 2» y «Paper 4» son las réplicas de las redes SFNN/DFNN y del SARIMA, entrenadas con la misma "
           "partición y evaluadas en las mismas filas. Sus variantes con clima no tuvieron clima en prueba y no se repiten. "
           "`RMSE_media_semillas` ± `RMSE_sd_semillas` resume 5 semillas de las redes del paper 2.",
           "- El XGBoost del paper 1 se evaluó en otro conjunto (70 distritos, 5 040 filas) y se resume en "
           "`model_comparison.txt`, no en esta tabla.",
           "- RMSE/MAE/MASE miden el error en casos por distrito y mes; MASE divide el MAE entre el error medio que tenía la regla «mismo mes del año anterior» en el entrenamiento (menor que 1 = mejor que esa regla).",
           "- PR-AUC es la precisión media (`average_precision_score`); el clasificador al azar obtiene la prevalencia de alertas "
           f"({f(prevalence)}).",
           "- Alerta = mes con casos mayores que la media distrital + 2 desviaciones estándar del entrenamiento.",
           "- «Casos + clima» usa la imputación estacional del 52 %; en prueba solo el "
           f"{pct(t[t.entradas == 'Casos + clima'].uso_clima_prueba.iloc[0])} de las predicciones tuvo clima completo. "
           "Las variantes sin imputar y de huecos cortos no tuvieron clima en prueba y son idénticas a «Solo casos».",
           ""]
    (OUT / "tabla_comparativa.md").write_text("\n".join(md), encoding="utf-8")
    return t


def best_by_validation(t):
    """Modelo con menor RMSE de validación entre equipo y papers, sin mirar la prueba."""
    cand = t[t.origen != "Referencia"].dropna(subset=["RMSE_validacion"])
    return cand.sort_values("RMSE_validacion", kind="stable").iloc[0]


def fig_error(t):
    models = t[t.entradas.isin(VARIANTS.values())]
    refs = t[t.entradas == "Regla fija"].dropna(subset=["RMSE"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(MODELS))
    h = .36
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        for k, (variant, vname) in enumerate(VARIANTS.items()):
            vals = [models[(models.modelo == NAMES[f]) & (models.entradas == vname)][metric].iloc[0] for f in MODELS]
            ys = y + (k - .5) * (h + .04)
            ax.barh(ys, vals, height=h, color=VCOLOR[variant], label=vname, edgecolor=SURFACE, linewidth=1)
            for yy, v in zip(ys, vals):
                ax.text(v + .005, yy, f"{v:.3f}".replace(".", ","), va="center", fontsize=8, color=INK2)
        for r, ls in zip(refs.itertuples(), ["--", ":", "-."]):
            v = getattr(r, metric)
            ax.axvline(v, color=INK2, ls=ls, lw=1.4,
                       label=f"{r.modelo.replace('Ref.: ', 'Regla: ')}" if metric == "RMSE" else None)
        ax.set_title(f"{metric} en prueba (menor es mejor)")
        ax.set_xlabel("casos por distrito y mes")
        lo = min(models[metric].min(), refs[metric].min())
        hi = max(models[metric].max(), refs[metric].max())
        ax.set_xlim(lo * .85, hi * 1.06)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [SHORT[f] for f in MODELS])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.08), ncol=5, fontsize=8.5,
               handlelength=2.4)
    fig.suptitle("Error de los cinco modelos frente a reglas simples — prueba 2019–2024, 57 distritos",
                 x=.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig1_error_por_modelo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_observed(pred, sel):
    wc = sel[sel.task == "casos"].iloc[0]
    p = pred[(pred.task == "casos")]
    obs = p[(p.family == wc.family) & (p.variant == wc.variant)].groupby("fecha_mes")["observed"].sum()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(obs.index, obs.values, color=INK, lw=2, label="Observado")
    series = [(wc.family, wc.variant, f"{SHORT[wc.family]} (elegido por validación)"),
              ("mlp", "estacional_52", "MLP casos + clima (menor RMSE en prueba)")]
    for family, variant, label in series:
        s = p[(p.family == family) & (p.variant == variant)].groupby("fecha_mes")["predicted"].sum()
        ax.plot(s.index, s.values, color=COLOR[family], lw=2, label=label)
    ax.set_title("Casos mensuales totales en los 57 distritos: observados vs. predichos (prueba 2019–2024)")
    ax.set_ylabel("casos por mes")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.28), ncol=3)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig2_observado_vs_predicho.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pr(pred, m):
    p = pred[(pred.task == "alerta") & (pred.variant == "solo_casos")]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    prevalence = None
    for family in MODELS:
        g = p[p.family == family]
        prec, rec, _ = precision_recall_curve(g.observed, g.predicted)
        ap = m[(m.family == family) & (m.variant == "solo_casos") & (m.task == "alerta")].PR_AUC.iloc[0]
        ax.step(rec, prec, where="post", color=COLOR[family], lw=2,
                label=f"{SHORT[family]} (PR-AUC {ap:.3f})".replace(".", ","))
        prevalence = g.observed.mean()
    ax.axhline(prevalence, color=INK2, ls="--", lw=1.2)
    ax.text(1, prevalence + .01, f"al azar ({prevalence:.3f})".replace(".", ","), ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("Recall (alertas reales detectadas)")
    ax.set_ylabel("Precisión (alertas emitidas que eran reales)")
    ax.set_title("Detección de alertas — solo casos, prueba 2019–2024")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(.6, ax.get_ylim()[1]))
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig3_curvas_precision_recall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_importance(sel):
    cfg = ex.PROTOCOLS[PROTOCOL]
    d = pd.read_csv(ex.DATA / PROTOCOL / "estacional_52.csv", dtype={"ubigeo": str},
                    parse_dates=["fecha_mes"], low_memory=False)
    full = d[d.fecha_mes.dt.year <= cfg["val"]]
    x = ex.features(full, True)
    keep = x.notna().all(axis=1)
    candidate = int(sel[sel.task == "casos"].candidate.iloc[0])
    model = ex.make_model("random_forest", "casos", candidate, True).fit(x[keep], full.casos[keep])
    imp = pd.Series(model.feature_importances_, index=x.columns).sort_values()
    labels = {"lag1": "casos mes −1", "lag2": "casos mes −2", "lag3": "casos mes −3",
              "lag12": "casos mismo mes año −1", "media3": "media casos 3 meses",
              "mes_sin": "estacionalidad (seno)", "mes_cos": "estacionalidad (coseno)"}
    for v, n in [("pp_total_mm", "lluvia"), ("tmax_media", "temp. máx."), ("tmin_media", "temp. mín.")]:
        for k in (1, 2, 3):
            labels[f"{v}_lag{k}"] = f"{n} mes −{k}"
    climate = [c.endswith(("lag1", "lag2", "lag3")) and not c.startswith("lag") for c in imp.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([labels[c] for c in imp.index], imp.values, height=.7,
            color=["#eb6834" if c else "#2a78d6" for c in climate], edgecolor=SURFACE)
    for i, v in enumerate(imp.values):
        ax.text(v + .003, i, f"{100 * v:.1f} %".replace(".", ","), va="center", fontsize=8, color=INK2)
    share = imp[climate].sum()
    ax.set_title(f"Importancia de variables — Random Forest con clima\n"
                 f"variables de casos {100 * (1 - share):.0f} % · variables de clima {100 * share:.0f} %",
                 fontsize=10.5)
    ax.set_xlabel("importancia (reducción media de impureza)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2a78d6", label="Historial de casos"),
                       Patch(color="#eb6834", label="Clima rezagado")], loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig4_importancia_variables_rf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    imp.rename("importancia").to_csv(OUT / "importancia_variables_rf.csv")
    return share, int(keep.sum()), len(keep)


def fig_districts(pred, sel):
    wc = sel[sel.task == "casos"].iloc[0]
    p = pred[(pred.task == "casos") & (pred.family == wc.family) & (pred.variant == wc.variant)].copy()
    names = pd.read_csv(ex.DATA / PROTOCOL / "sin_imputar.csv", dtype={"ubigeo": str},
                        usecols=["ubigeo", "distrito", "provincia"]).drop_duplicates("ubigeo")
    g = p.groupby("ubigeo").agg(observados=("observed", "sum"), predichos=("predicted", "sum"))
    g = g.join(names.set_index("ubigeo"))
    top = g.sort_values("observados", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    yy = np.arange(len(top))
    ax.barh(yy + .2, top.observados, height=.38, color=INK2, label="Observados", edgecolor=SURFACE)
    ax.barh(yy - .2, top.predichos, height=.38, color=COLOR[wc.family],
            label=f"Predichos ({SHORT[wc.family]})", edgecolor=SURFACE)
    ax.set_yticks(yy, [d.title() for d in top.distrito])
    ax.set_xlabel("casos acumulados 2019–2024")
    ax.set_title("15 distritos con más casos: total observado vs. predicho")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig5_casos_por_distrito.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    g.sort_values("observados", ascending=False).to_csv(OUT / "casos_por_distrito.csv")


def write_comparison(t, m, sel, pred, share):
    """results/model_comparison.txt: resumen de resultados y análisis detallado."""
    models = t[t.entradas.isin(VARIANTS.values())]
    solo = models[models.entradas == "Solo casos"].set_index("modelo")
    clima = models[models.entradas == "Casos + clima"].set_index("modelo")
    refs = t[t.entradas == "Regla fija"].set_index("modelo")
    wc = sel[sel.task == "casos"].iloc[0]
    best = solo.loc[NAMES[wc.family]]
    zero, last = refs.loc["Ref.: siempre cero"], refs.loc["Ref.: último mes"]
    rnd = refs.loc["Ref.: clasificador al azar"].PR_AUC
    a = m[(m.task == "alerta") & (m.family == wc.family) & (m.variant == "solo_casos")].iloc[0]
    p = pred[(pred.task == "casos") & (pred.variant == "solo_casos")]
    totals = p.groupby("family").predicted.sum()
    observed = int(p[p.family == wc.family].observed.sum())
    zeros = float((p[p.family == wc.family].observed == 0).mean())
    d = pd.read_csv(ex.DATA / PROTOCOL / "sin_imputar.csv", usecols=["fecha_mes", "casos"], parse_dates=["fecha_mes"])
    yearly = d.groupby(d.fecha_mes.dt.year).casos.sum()
    cfg = ex.PROTOCOLS[PROTOCOL]
    fit_mean = yearly.loc[:cfg["fit"]].mean()
    val_mean = yearly.loc[cfg["fit"] + 1:cfg["val"]].mean()
    test_mean = yearly.loc[cfg["val"] + 1:].mean()
    clima_frac = float(m[(m.variant == "estacional_52")].climate_fraction.max())
    rmse_gap = (solo.RMSE.max() - solo.RMSE.min())
    papers = t[t.origen.str.startswith("Paper")]
    best_all = best_by_validation(t)
    f = lambda x, n=3: f"{x:.{n}f}"

    def paper_label(r):
        if r.origen == "Paper 2":
            name = r.modelo.split(" (")[0]
            inputs = "3 rezagos" if "3 rezagos" in r.entradas else "7 variables"
        else:
            name, inputs = ("SARIMA del paper" if "del paper" in r.modelo else "SARIMA por AIC"), "serie"
        return f"{r.origen} · {name} / {inputs}"

    def row(name, r):
        cells = [f(r.RMSE) if pd.notna(r.RMSE) else "-", f(r.MAE) if pd.notna(r.MAE) else "-",
                 f(r.PR_AUC) if pd.notna(r.PR_AUC) else "-",
                 str(r.alertas_detectadas) if isinstance(r.alertas_detectadas, str) else "-"]
        return f"{name:<38}{cells[0]:>8}{cells[1]:>8}{cells[2]:>9}{cells[3]:>11}"

    lines = [
        "RESUMEN DE RESULTADOS Y ANÁLISIS — Grupo 4, leishmaniosis cutánea en Cajamarca",
        "=" * 78, "",
        "Protocolo: ajuste 2001-2015, validación 2016-2018, prueba 2019-2024.",
        f"Prueba: {int(m.n_test.max())} filas distrito x mes de 57 distritos. Semilla 42.",
        "Tareas: casos (regresión, mes siguiente) y alerta (clasificación de brote).",
        "", "1. TABLA COMPARATIVA (prueba)", "-" * 78,
        f"{'Modelo / entradas':<38}{'RMSE':>8}{'MAE':>8}{'PR-AUC':>9}{'Alertas':>11}",
    ]
    for r in models.itertuples():
        short = next(SHORT[k] for k, v in NAMES.items() if v == r.modelo)
        lines.append(row(f"{short} / {r.entradas}", r))
    for r in papers.itertuples():
        lines.append(row(paper_label(r), r))
    for name, r in refs.iterrows():
        lines.append(row(name.replace("Ref.: ", "Regla: "), r))
    lines += [
        "", "Paper 2 = redes SFNN/DFNN (Donizette et al.); Paper 4 = SARIMA (El Alaoui et al.). Misma partición",
        "y mismas filas de prueba que el equipo. El paper 1 (XGBoost) se evaluó en 70 distritos: sección 3.",
        "Tabla completa: results/metrics.csv. Gráficos: results/plots/fig1..fig5 y comparacion_final_*.png.",
        "", "2. MEJOR MODELO", "-" * 78,
        *final_lines(best_all, zero, last, rnd, f),
        "",
        f"Mejor del equipo por validación: {NAMES[wc.family]} con solo casos "
        f"(RMSE de validación {f(wc.validation_score, 4)}).",
        f"- Casos: RMSE {f(best.RMSE)}, {100 * (1 - best.RMSE / zero.RMSE):.0f} % menor que 'siempre cero' y "
        f"{100 * (1 - best.RMSE / last.RMSE):.0f} % menor que 'último mes'.",
        f"- Alertas: PR-AUC {f(best.PR_AUC)}, {best.PR_AUC / rnd:.1f} veces la del azar ({f(rnd)}); detecta "
        f"{best.alertas_detectadas} meses de brote con {int(best.falsas_alarmas)} falsas alarmas.",
        f"- La MLP obtiene el menor RMSE en prueba del equipo ({f(models.RMSE.min())}); la diferencia entre los cinco modelos "
        f"con solo casos es de {f(rmse_gap)} casos. Random Forest y MLP se consideran empatados.",
        *tie_lines(best_all, best, NAMES[wc.family], f),
        "", "3. MODELOS DE LOS PAPERS", "-" * 78,
        *paper_lines(t, papers, solo, zero, zeros, f),
        "", "4. POR QUÉ", "-" * 78,
        f"- El historial reciente de casos domina: las variables de casos suman el {100 * (1 - share):.0f} % de la",
        "  importancia del Random Forest; la media de los últimos 3 meses es la principal.",
        f"- Los conteos son pequeños y ruidosos: el {100 * zeros:.0f} % de las filas de prueba tiene 0 casos.",
        "  Random Forest (promedio de 100 árboles) y MLP (función continua) suavizan ese ruido;",
        "  un árbol solo salta entre pocos grupos y queda último.",
        "- La CNN solo ve 3 meses y hay pocos datos por distrito: no aprovecha la convolución.",
        "- El modelo lineal queda cerca porque la señal principal es la persistencia.",
        "", "5. APORTE DEL CLIMA", "-" * 78,
        f"- En casos, el clima cambia el RMSE como máximo {f((clima.RMSE - solo.RMSE).abs().max(), 4)}. En alertas ayuda a",
        "  Random Forest y al árbol, pero empeora a la CNN y dispara falsas alarmas del lineal.",
        f"- El clima suma el {100 * share:.0f} % de la importancia (sobre todo lluvia de 2-3 meses antes).",
        "- Los datos SENAMHI disponibles terminan en octubre de 2015; en 2019-2024 el clima",
        f"  es imputado y solo el {100 * clima_frac:.0f} % de las predicciones de prueba tuvo clima completo.",
        *paper1_climate_lines(f),
        "  Con estos datos no se puede afirmar ni descartar que el clima mejore la predicción.",
        "", "6. INSIGHTS", "-" * 78,
        f"- En MAE ganan las reglas simples ('siempre cero' {f(zero.MAE)}); los modelos ganan en RMSE porque",
        "  fallan menos en los meses con varios casos, los relevantes para vigilancia.",
        f"- Los modelos del equipo sobreestiman el total: {observed} casos observados frente a "
        f"{totals.min():.0f}-{totals.max():.0f} predichos.",
        f"  La incidencia anual cayó de {fit_mean:.0f} (ajuste) a {val_mean:.0f} (validación) y {test_mean:.0f} (prueba).",
        "- Se anticipa el pico de enero de cada año, pero no su tamaño.",
        "- Los distritos con más casos se subestiman y los pequeños se sobreestiman.",
        *alert_cost_lines(best_all, a, NAMES[wc.family]),
        "", "7. LIMITACIONES", "-" * 78,
        "- Clima sin observaciones después de 2015; la imputación no recupera mediciones reales.",
        "- Una semilla en los modelos del equipo (5 en las redes del paper 2), un periodo de prueba y",
        "  2 configuraciones por modelo del equipo; sin intervalos de confianza.",
        "- El periodo de prueba ya se revisó en versiones anteriores: no es validación externa nueva.",
        "- Se excluyen distritos con menos de 50 casos hasta 2015.",
        "- La alerta (media + 2 DE) es una convención; el puntaje no está calibrado.",
        "- Predicción a un mes con los casos reales del mes anterior; no se evalúa a varios meses.",
        "", "Análisis extendido: results/analisis/ANALISIS_RESULTADOS.md. Paper: report.pdf.", "",
    ]
    (OUT / "model_comparison.txt").write_text("\n".join(lines), encoding="utf-8")


def model_label(r):
    """Nombre corto con origen: «Paper 2 · DFNN» o el nombre del modelo del equipo."""
    name = r.modelo.split(" (")[0]
    return name if r.origen == "Equipo" else f"{r.origen} · {name}"


def final_lines(best_all, zero, last, rnd, f):
    """Modelo final: menor RMSE de validación entre equipo y papers 2 y 4 (criterio de report.pdf)."""
    lines = [f"Modelo final: {model_label(best_all)} con entradas «{best_all.entradas}». Es el de menor RMSE",
             f"de validación entre todos los modelos (equipo + papers 2 y 4): {f(best_all.RMSE_validacion, 4)}."]
    seeds = (f"; {f(best_all.RMSE_media_semillas)} ± {f(best_all.RMSE_sd_semillas)} con 5 semillas"
             if pd.notna(best_all.RMSE_media_semillas) else "")
    lines += [f"- Casos: RMSE {f(best_all.RMSE)}{' con la semilla 42' if seeds else ''}, "
              f"{100 * (1 - best_all.RMSE / zero.RMSE):.0f} % menor que 'siempre cero' y "
              f"{100 * (1 - best_all.RMSE / last.RMSE):.0f} % menor que 'último mes'{seeds}."]
    if pd.notna(best_all.PR_AUC):
        lines.append(f"- Alertas: PR-AUC {f(best_all.PR_AUC)}, {best_all.PR_AUC / rnd:.1f} veces la del azar; detecta "
                     f"{best_all.alertas_detectadas} brotes con {int(best_all.falsas_alarmas)} falsas alarmas.")
    return lines


def alert_cost_lines(best_all, a, team_name):
    """Falsas alarmas por alerta verdadera del modelo final y del mejor del equipo."""
    lines = []
    if best_all.origen != "Equipo" and pd.notna(best_all.recall):
        tp = int(str(best_all.alertas_detectadas).split("/")[0])
        lines.append(f"- Para detectar el {100 * best_all.recall:.0f} % de los brotes, {model_label(best_all)} emite ~"
                     f"{best_all.falsas_alarmas / max(tp, 1):.1f} alertas falsas por cada verdadera;")
        lines.append(f"  el {team_name}, el {100 * a.recall:.0f} % con ~"
                     f"{a.false_alerts / max(a.recall * a.true_alerts, 1):.1f}.")
    else:
        lines.append(f"- Para detectar el {100 * a.recall:.0f} % de los brotes, el modelo emite ~"
                     f"{a.false_alerts / max(a.recall * a.true_alerts, 1):.1f} alertas falsas por cada verdadera.")
    return lines


def tie_lines(best_all, team_best, team_name, f):
    """Compara el mejor del equipo con el modelo final, si este viene de un paper."""
    if best_all.origen == "Equipo":
        return []
    final = model_label(best_all)
    if pd.notna(best_all.RMSE_media_semillas):
        gap = team_best.RMSE - best_all.RMSE_media_semillas
        tie = abs(gap) < best_all.RMSE_sd_semillas
        lines = [f"- Frente a la media de 5 semillas de {final}, el {team_name} difiere en {f(gap)} de RMSE,",
                 f"  {'menos' if tie else 'más'} que la variación entre semillas"
                 f"{': en casos quedan empatados.' if tie else '.'}"]
    else:
        lines = [f"- {final} obtiene RMSE {f(best_all.RMSE)} frente a {f(team_best.RMSE)} del {team_name}."]
    better = "mejor" if best_all.PR_AUC > team_best.PR_AUC else "peor"
    return lines + [f"- En alertas, {final} es {better} (PR-AUC {f(best_all.PR_AUC)} frente a {f(team_best.PR_AUC)}).",
                    f"- La importancia de variables (secciones 4 y 5) se mide con el {team_name}, porque las redes no la dan."]


def paper_lines(t, papers, solo, zero, zeros, f):
    """Sección 3 de model_comparison.txt: réplica de cada paper revisado."""
    lines = []
    e1_path = P1_RESULTS / "e1_metricas.csv"
    lines.append("Paper 1 — XGBoost (Adams et al.). Evaluado en los 70 distritos de la comparación original")
    if e1_path.exists():
        e1 = pd.read_csv(e1_path)
        gain = e1.columns[-1]
        lines += ["(5 040 filas, prueba 2019-2024); por eso no está en la tabla 1.",
                  f"  {'Modelo':<58}{'RMSE':>8}{'MAE':>8}"]
        for r in e1.itertuples(index=False):
            lines.append(f"  {r.modelo[:57]:<58}{f(r.RMSE):>8}{f(r.MAE):>8}")
        get = lambda p: e1[e1.modelo.str.startswith(p)].iloc[0]
        rf, e1a, e1b, e1c = get("Equipo · Random Forest (solo"), get("E1a"), get("E1b"), get("E1c")
        verb = lambda g: "empata con" if abs(g) < 1 else ("empeora" if g > 0 else "mejora")
        lines += [f"- Con las mismas 7 variables, XGBoost (E1a) {verb(e1a[gain])} el Random Forest del equipo "
                  f"({f(e1a.RMSE)} vs {f(rf.RMSE)}).",
                  f"- El método del paper con un modelo para todos (E1b) {verb(e1b[gain])} el Random Forest "
                  f"({f(e1b.RMSE)}); con un modelo",
                  f"  por distrito (E1c) el RMSE cambia {e1c[gain]:+.1f} %, aunque el MAE baja a {f(e1c.MAE)}."]
    else:
        lines.append(f"(faltan sus resultados en {P1_RESULTS.relative_to(ROOT)}; ejecutar el notebook 05).")

    p2, p4 = papers[papers.origen == "Paper 2"], papers[papers.origen == "Paper 4"]
    if len(p2):
        three = p2[p2.entradas.str.contains("3 rezagos")]
        seven = p2[p2.entradas.str.contains("7 variables")]
        worse = three.RMSE.min() > solo.RMSE.max()
        input_gain = three.RMSE_media_semillas.mean() - seven.RMSE_media_semillas.mean()
        arch_gap = seven.RMSE_media_semillas.max() - seven.RMSE_media_semillas.min()
        lines += ["", "Paper 2 — redes SFNN1, SFNN2 y DFNN (Donizette et al.).",
                  f"- Con los 3 rezagos del paper: RMSE {f(three.RMSE.min())}-{f(three.RMSE.max())}"
                  + (f", peor que los cinco modelos del equipo ({f(solo.RMSE.min())}-{f(solo.RMSE.max())})." if worse else "."),
                  f"- Con las 7 variables del equipo, el RMSE medio de 5 semillas cambia de "
                  f"{f(three.RMSE_media_semillas.mean())} a {f(seven.RMSE_media_semillas.mean())};",
                  f"  entre arquitecturas la diferencia es de {f(arch_gap)}."
                  + (" Importan más las variables que la arquitectura." if input_gain > arch_gap else ""),
                  "- Sus variantes con clima no tuvieron clima en 2019-2024 y dan lo mismo que solo casos."]
    if len(p4):
        paper = p4[p4.modelo.str.contains("del paper")].iloc[0]
        aic = p4[p4.modelo.str.contains("AIC")].iloc[0]
        lowest = aic.PR_AUC <= t[t.origen != "Referencia"].PR_AUC.min()
        lines += ["", "Paper 4 — SARIMA (El Alaoui et al.), una serie por distrito.",
                  f"- Con el orden del paper, (0,1,0)(1,1,0)12, en todos los distritos: RMSE {f(paper.RMSE)}"
                  + (f", peor que 'siempre cero' ({f(zero.RMSE)})." if paper.RMSE > zero.RMSE else "."),
                  f"- Eligiendo el orden por AIC en cada distrito: RMSE {f(aic.RMSE)}, cerca del equipo, pero PR-AUC "
                  f"{f(aic.PR_AUC)}{', la más baja de la tabla' if lowest else ''},",
                  f"  con {int(aic.falsas_alarmas)} falsas alarmas."
                  + (f" Con el {100 * zeros:.0f} % de los meses en cero, la serie sigue la persistencia,"
                     if lowest else "")]
        if lowest:
            lines.append("  pero anticipa los brotes peor que los demás modelos.")
    lines += ["", "Paper 3 — Araque y Paipa (tesis sobre dengue). No se replicó: se usó como antecedente de los",
              "efectos climáticos rezagados, y sus resultados son de otra enfermedad."]
    return lines


def paper1_climate_lines(f):
    """Resultado del paper 1 con clima observado (protocolo histórico, 2012-2014)."""
    path = P1_RESULTS / "e3_metricas.csv"
    if not path.exists():
        return []
    e3 = pd.read_csv(path)
    e3 = e3[e3["horizonte (meses)"] == 1].set_index("modelo")
    solo_x, clima_x = e3.loc["XGBoost · solo casos"], e3.loc["XGBoost · con clima sin rellenar"]
    return [f"- La réplica del paper 1 con clima observado (2012-2014, 46 distritos) llega a lo mismo: XGBoost",
            f"  con clima (RMSE {f(clima_x.RMSE)}) no mejora a XGBoost con solo casos ({f(solo_x.RMSE)})."
            if clima_x.RMSE >= solo_x.RMSE else
            f"  con clima baja el RMSE de {f(solo_x.RMSE)} a {f(clima_x.RMSE)}."]


def main():
    m, sel = load()
    t = comparison_table(m, sel)
    pred = pd.read_csv(RES / "predicciones.csv", dtype={"ubigeo": str}, parse_dates=["fecha_mes"])
    pred = pred[pred.protocol == PROTOCOL]
    fig_error(t)
    fig_observed(pred, sel)
    fig_pr(pred, m)
    share, rows_clima, rows_total = fig_importance(sel)
    fig_districts(pred, sel)
    write_comparison(t, m, sel, pred, share)
    print((OUT / "tabla_comparativa.md").read_text(encoding="utf-8"))
    print(f"Importancia total del clima en RF: {share:.3f} ({rows_clima}/{rows_total} filas de ajuste con clima)")


if __name__ == "__main__":
    main()
