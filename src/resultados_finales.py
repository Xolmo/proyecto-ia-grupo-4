"""Tabla comparativa y gráficos del entregable final.

Lee los resultados ya generados por `ejecutar.py --modo experimento-v1` (protocolo
oficial: ajuste 2001-2015, validación 2016-2018, prueba 2019-2024, 57 distritos)
y escribe en results/:

- metrics.csv / tabla_comparativa.md: 5 modelos x {solo casos, con clima} + referencias.
- model_comparison.txt: resumen de resultados y análisis detallado.
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


def comparison_table(m, sel):
    rows = []
    for family in MODELS:
        for variant, vname in VARIANTS.items():
            c = m[(m.family == family) & (m.variant == variant) & (m.task == "casos")].iloc[0]
            a = m[(m.family == family) & (m.variant == variant) & (m.task == "alerta")].iloc[0]
            rows.append({"modelo": NAMES[family], "entradas": vname,
                         "RMSE": c.RMSE, "MAE": c.MAE, "MASE": c.MASE,
                         "RMSE_validacion": c.validation_score,
                         "PR_AUC": a.PR_AUC, "F1": a.F1, "recall": a.recall, "precision": a.precision,
                         "alertas_detectadas": f"{int(round(a.recall * a.true_alerts))}/{int(a.true_alerts)}",
                         "falsas_alarmas": int(a.false_alerts),
                         "uso_clima_prueba": c.climate_fraction})
    prevalence = None
    for family in ["ultimo_mes", "estacional", "cero"]:
        c = m[(m.family == family) & (m.variant == "referencia")].iloc[0]
        rows.append({"modelo": NAMES[family], "entradas": "Regla fija",
                     "RMSE": c.RMSE, "MAE": c.MAE, "MASE": c.MASE})
    alerts = m[(m.task == "alerta")].iloc[0]
    prevalence = alerts.true_alerts / alerts.n_test
    rows.append({"modelo": "Ref.: clasificador al azar", "entradas": "Regla fija",
                 "PR_AUC": prevalence})
    t = pd.DataFrame(rows)
    PLOTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "metrics.csv", index=False)

    # Ganadores por validación (selección honesta) y por prueba (descriptivo).
    winners = {r.task: r for r in sel.itertuples()}
    best_test_rmse = t.loc[t.RMSE.idxmin()]
    best_test_ap = t.loc[t.PR_AUC.idxmax()]

    def f(x, d=3):
        return "—" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")

    def pct(x):
        return "—" if pd.isna(x) else f"{100 * x:.0f} %"

    md = ["# Tabla comparativa de modelos — protocolo oficial", "",
          "Prueba 2019–2024, 57 distritos de Cajamarca, 4 104 filas distrito × mes. "
          "Ajuste 2001–2015 y validación 2016–2018. ↓ menor es mejor; ↑ mayor es mejor.", "",
          "| Modelo | Entradas | RMSE ↓ | MAE ↓ | MASE ↓ | PR-AUC ↑ | F1 ↑ | Recall ↑ | Precisión ↑ | Alertas detectadas | Falsas alarmas |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples():
        md.append(f"| {r.modelo} | {r.entradas} | {f(r.RMSE)} | {f(r.MAE)} | {f(r.MASE)} | {f(r.PR_AUC)} | "
                  f"{f(r.F1)} | {pct(r.recall)} | {pct(r.precision)} | "
                  f"{'—' if pd.isna(r.alertas_detectadas) else r.alertas_detectadas} | "
                  f"{'—' if pd.isna(r.falsas_alarmas) else int(r.falsas_alarmas)} |")
    wc, wa = winners["casos"], winners["alerta"]
    md += ["",
           "**Selección por validación (criterio del pipeline):** "
           f"casos → {NAMES[wc.family]} (RMSE de validación {f(wc.validation_score, 4)}); "
           f"alertas → {NAMES[wa.family]} (PR-AUC de validación {f(wa.validation_score, 4)}). "
           "La variante elegida no usó clima en prueba, así que equivale a la fila «Solo casos».", "",
           f"**Mejor valor observado en prueba (descriptivo):** RMSE {f(best_test_rmse.RMSE, 4)} con "
           f"{best_test_rmse.modelo} ({best_test_rmse.entradas}); PR-AUC {f(best_test_ap.PR_AUC, 4)} con "
           f"{best_test_ap.modelo} ({best_test_ap.entradas}). No se usa para elegir modelo porque la prueba ya fue observada.", "",
           "Notas:",
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
    cf = float(m[(m.variant == "estacional_52")].climate_fraction.max())
    rmse_gap = (solo.RMSE.max() - solo.RMSE.min())
    f = lambda x, n=3: f"{x:.{n}f}"

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
    for name, r in refs.iterrows():
        lines.append(row(name.replace("Ref.: ", "Regla: "), r))
    lines += [
        "", "Tabla completa: results/metrics.csv. Gráficos: results/plots/fig1..fig5.",
        "", "2. MEJOR MODELO", "-" * 78,
        f"Elegido por validación: {NAMES[wc.family]} con solo casos (RMSE de validación {f(wc.validation_score, 4)}).",
        f"- Casos: RMSE {f(best.RMSE)}, {100 * (1 - best.RMSE / zero.RMSE):.0f} % menor que 'siempre cero' y "
        f"{100 * (1 - best.RMSE / last.RMSE):.0f} % menor que 'último mes'.",
        f"- Alertas: PR-AUC {f(best.PR_AUC)}, {best.PR_AUC / rnd:.1f} veces la del azar ({f(rnd)}); detecta "
        f"{best.alertas_detectadas} meses de brote con {int(best.falsas_alarmas)} falsas alarmas.",
        f"- La MLP obtiene el menor RMSE en prueba ({f(models.RMSE.min())}); la diferencia entre los cinco modelos "
        f"con solo casos es de {f(rmse_gap)} casos. Random Forest y MLP se consideran empatados.",
        "", "3. POR QUÉ", "-" * 78,
        f"- El historial reciente de casos domina: las variables de casos suman el {100 * (1 - share):.0f} % de la",
        "  importancia del Random Forest; la media de los últimos 3 meses es la principal.",
        f"- Los conteos son pequeños y ruidosos: el {100 * zeros:.0f} % de las filas de prueba tiene 0 casos.",
        "  Random Forest (promedio de 100 árboles) y MLP (función continua) suavizan ese ruido;",
        "  un árbol solo salta entre pocos grupos y queda último.",
        "- La CNN solo ve 3 meses y hay pocos datos por distrito: no aprovecha la convolución.",
        "- El modelo lineal queda cerca porque la señal principal es la persistencia.",
        "", "4. APORTE DEL CLIMA", "-" * 78,
        f"- En casos, el clima cambia el RMSE como máximo {f((clima.RMSE - solo.RMSE).abs().max(), 4)}. En alertas ayuda a",
        "  Random Forest y al árbol, pero empeora a la CNN y dispara falsas alarmas del lineal.",
        f"- El clima suma el {100 * share:.0f} % de la importancia (sobre todo lluvia de 2-3 meses antes).",
        "- Los datos SENAMHI disponibles terminan en octubre de 2015; en 2019-2024 el clima",
        f"  es imputado y solo el {100 * cf:.0f} % de las predicciones de prueba tuvo clima completo.",
        "  Con estos datos no se puede afirmar ni descartar que el clima mejore la predicción.",
        "", "5. INSIGHTS", "-" * 78,
        f"- En MAE ganan las reglas simples ('siempre cero' {f(zero.MAE)}); los modelos ganan en RMSE porque",
        "  fallan menos en los meses con varios casos, los relevantes para vigilancia.",
        f"- Todos sobreestiman el total: {observed} casos observados frente a {totals.min():.0f}-{totals.max():.0f} predichos.",
        f"  La incidencia anual cayó de {fit_mean:.0f} (ajuste) a {val_mean:.0f} (validación) y {test_mean:.0f} (prueba).",
        "- Se anticipa el pico de enero de cada año, pero no su tamaño.",
        "- Los distritos con más casos se subestiman y los pequeños se sobreestiman.",
        f"- Para detectar el {100 * a.recall:.0f} % de los brotes, el modelo emite ~"
        f"{a.false_alerts / max(a.recall * a.true_alerts, 1):.1f} alertas falsas por cada verdadera.",
        "", "6. LIMITACIONES", "-" * 78,
        "- Clima sin observaciones después de 2015; la imputación no recupera mediciones reales.",
        "- Una semilla, un periodo de prueba y 2 configuraciones por modelo; sin intervalos de confianza.",
        "- El periodo de prueba ya se revisó en versiones anteriores: no es validación externa nueva.",
        "- Se excluyen distritos con menos de 50 casos hasta 2015.",
        "- La alerta (media + 2 DE) es una convención; el puntaje no está calibrado.",
        "- Predicción a un mes con los casos reales del mes anterior; no se evalúa a varios meses.",
        "", "Análisis extendido: results/analisis/ANALISIS_RESULTADOS.md. Paper: report.pdf.", "",
    ]
    (OUT / "model_comparison.txt").write_text("\n".join(lines), encoding="utf-8")


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
