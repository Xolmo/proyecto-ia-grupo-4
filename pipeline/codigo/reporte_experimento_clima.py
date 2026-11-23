"""Resume resultados existentes, sin reentrenar ni seleccionar mirando prueba."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experimento_clima"


def table(d, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, r in d.iterrows():
        vals = []
        for c in columns:
            v = r[c]
            vals.append("—" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    MLP_MAX = int(pd.read_csv(OUT / "diagnostico_ajuste.csv").max_iter.max())
    m = pd.read_csv(OUT / "metricas.csv")
    summary = pd.read_csv(ROOT / "data/processed/experimento_clima/resumen_imputacion.csv")
    chosen = []
    for (_, task), g in m[m.variant != "referencia"].groupby(["protocol", "task"]):
        chosen.append(g.sort_values("validation_score", ascending=task == "casos", kind="stable").iloc[0])
    selected = pd.DataFrame(chosen)
    selected.to_csv(OUT / "seleccion_por_validacion.csv", index=False)
    candidates = m[m.variant != "referencia"]
    def best(protocol, task, variant=None):
        g = candidates[(candidates.protocol == protocol) & (candidates.task == task)]
        if variant is not None:
            g = g[g.variant == variant]
        return g.sort_values("RMSE" if task == "casos" else "PR_AUC", ascending=task == "casos", kind="stable").iloc[0]
    reg = best("oficial", "casos")
    plain = best("oficial", "casos", "sin_imputar")
    alert = best("oficial", "alerta")
    hist = best("historico", "casos")
    val = selected[(selected.protocol == "oficial") & (selected.task == "casos")].iloc[0]
    relative = (plain.RMSE - reg.RMSE) / plain.RMSE * 100
    recursive = int(summary[(summary.protocol == "oficial") & (summary.variant == "estacional_52")].recursive_sources.iloc[0])
    short_usage = candidates[(candidates.protocol == "oficial") & (candidates.variant == "huecos_cortos")].climate_fraction.max()
    lines = ["# Resultados del experimento climático", "",
             "Comparación exploratoria con semilla 42. Los notebooks originales no se modificaron.", "",
             "## Conclusión de esta ejecución", "",
             f"En prueba oficial, el menor RMSE puntual fue {reg.family} + {reg.variant}: {reg.RMSE:.4f}, frente a {plain.RMSE:.4f} de {plain.family} sin imputación. La diferencia relativa es aproximadamente {relative:.3f} %; no se demostró significancia estadística. La selección exclusiva por validación fue {val.family} + {val.variant}. En el histórico, el menor RMSE observado fue {hist.family} + {hist.variant} ({hist.RMSE:.4f}).", "",
             f"Para alertas oficiales, {alert.family} + {alert.variant} alcanzó average precision {alert.PR_AUC:.4f} y recall {alert.recall:.4f}: detectó {round(alert.recall * alert.true_alerts)} de {int(alert.true_alerts)} alertas y generó {int(alert.false_alerts)} falsas alarmas. Es un resultado descriptivo de prueba; su selección independiente se informa por separado y no debe presentarse como sistema operativo validado.", "",
             f"La propuesta de huecos cortos no reconstruye años enteros sin estaciones: en prueba oficial su uso máximo de rama climática fue {short_usage:.1%}. El relleno del 52 % no significa haber recuperado mediciones reales; {recursive} rellenos se apoyan directamente en otros ya imputados. Los resultados MLP están sujetos al diagnóstico de convergencia informado al final.", "",
             "## Datasets y presupuesto", "",
             table(summary, ["protocol", "variant", "districts", "rows", "initial_missing_cells", "filled_cells", "fraction_missing_filled", "recursive_sources"]), "",
             "Las fracciones se refieren a celdas originalmente nulas de lluvia total/tmax/tmin. El redondeo a celdas impide representar exactamente 0,52. Las columnas derivadas no se cuentan otra vez.", "",
             "La imputación del 52 % permite fuentes previamente imputadas para alcanzar el presupuesto. Esta propagación puede fabricar regularidad y no recupera mediciones reales. La guía registra fuentes y fechas.", "",
             "## Ganadores seleccionados por validación", "",
             table(selected, ["protocol", "task", "variant", "family", "validation_score", "RMSE", "MAE", "PR_AUC", "recall", "climate_fraction"]), "",
             "RMSE/MAE corresponden a casos; PR_AUC/recall a alertas. `climate_fraction` es la fracción de predicciones de prueba que efectivamente utilizó la rama climática. Las restantes usan el modelo de casos de la misma familia.", ""]
    for protocol in ["historico", "oficial"]:
        lines += [f"## Comparación por versión: {protocol}", ""]
        for task in ["casos", "alerta"]:
            g = m[(m.protocol == protocol) & (m.task == task) & (m.variant != "referencia")]
            winners = g.sort_values("validation_score", ascending=task == "casos", kind="stable").groupby("variant", sort=False).head(1)
            lines += [f"### {task}: mejor familia de cada versión según validación", "",
                      table(winners, ["variant", "family", "candidate", "validation_score", "RMSE", "MAE", "PR_AUC", "F1", "recall", "climate_fraction"]), ""]
        refs = m[(m.protocol == protocol) & (m.variant == "referencia")]
        lines += ["### Referencias de regresión sobre las mismas filas", "", table(refs, ["family", "RMSE", "MAE", "MASE"]), ""]
    lines += ["## Mejor resultado observado en prueba (descriptivo)", "",
              "Esta tabla compara resultados ya evaluados; no debe confundirse con selección independiente por validación ni garantiza que vuelva a ganar fuera de muestra.", ""]
    descriptive = []
    for (protocol, task), g in m[m.variant != "referencia"].groupby(["protocol", "task"]):
        descriptive.append(g.sort_values("RMSE" if task == "casos" else "PR_AUC", ascending=task == "casos").iloc[0])
    lines += [table(pd.DataFrame(descriptive), ["protocol", "task", "variant", "family", "RMSE", "MAE", "PR_AUC", "F1", "recall", "climate_fraction"]), "",
              "## Interpretación y límites", "",
              "- Se compararon 5 familias × 4 condiciones × 2 protocolos × 2 tareas: 80 combinaciones, con 2 candidatos por combinación. El control de casos se reutiliza cuando las entradas son idénticas.",
              "- Histórico: ajuste 2001–2009, validación 2010–2011, prueba 2012–2014. Oficial: ajuste 2001–2015, validación 2016–2018, prueba 2019–2024.",
              "- La selección de distritos y estaciones usa solo el ajuste inicial: 46 distritos históricos y 57 oficiales; no son los 70 del notebook original. Por ello los números nuevos no se comparan directamente con `arbol_metricas.csv`.",
              "- Las etiquetas de alerta se calculan nuevamente en cada corte de entrenamiento. Los umbrales probabilísticos se eligen en validación y se conservan en prueba.",
              "- El clima de 2019–2024 carece de observaciones; la variante 52 puede consumir estimaciones propagadas. Un buen resultado allí no demuestra haber recuperado el clima real.",
              "- No se imputa ningún nulo restante silenciosamente; el respaldo de casos mantiene exactamente las mismas filas de evaluación por protocolo.",
              f"- Una sola semilla, pocos hiperparámetros, MLP con máximo de {MLP_MAX} iteraciones y CNN de 50 épocas. Son comparaciones exploratorias sin significancia estadística ni validación clínica.", ""]
    diag = pd.read_csv(OUT / "diagnostico_ajuste.csv")
    hits = int(diag.hit_iteration_limit.sum())
    mlp_note = (f"MLP: {hits} de {len(diag)} ajustes alcanzaron el límite de {MLP_MAX} iteraciones; requieren mayor optimización antes de extraer conclusiones definitivas."
                if hits else f"MLP: los {len(diag)} ajustes convergieron antes del límite de {MLP_MAX} iteraciones (iteraciones usadas: {int(diag.n_iter.min())}–{int(diag.n_iter.max())}).")
    lines += [mlp_note, "",
              "Ver `metricas.csv`, `validacion.csv`, `predicciones.csv`, `seleccion_por_validacion.csv`, `diagnostico_ajuste.csv` y `ejecucion.json` para la evidencia completa.", ""]
    (OUT / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(selected[["protocol", "task", "variant", "family", "RMSE", "MAE", "PR_AUC", "recall", "climate_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
