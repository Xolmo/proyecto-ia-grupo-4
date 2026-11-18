"""Comparación de regresión en las mismas 5 040 filas del notebook original."""
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
import experimento_clima as ex
import integracion_clima as ic


def main():
    output = ex.ROOT / "results/comparacion_70_distritos"
    output.mkdir(parents=True, exist_ok=True)
    raw = ex.pp.clean_cases(ex.pp.load_raw(ex.RAW))
    raw = raw[raw.departamento == "CAJAMARCA"]
    panel = ex.pp.build_monthly_panel(raw)
    climate = pd.read_csv(ex.ROOT/"data/processed/dataset_cajamarca.csv", dtype={"ubigeo": str}, parse_dates=["fecha_mes"])
    base = climate.drop(columns=ex.pp.FEATURES).merge(panel[["ubigeo", "fecha_mes"]+ex.pp.FEATURES],
               on=["ubigeo", "fecha_mes"], validate="one_to_one")
    monthly = pd.read_csv(ex.ROOT/"data/processed/clima_mensual_cajamarca.csv", parse_dates=["fecha_mes"])
    assignment = base.drop_duplicates("ubigeo")[["ubigeo", "codigo_pp", "codigo_t"]]
    hist = ic.clima_por_distrito(monthly, assignment)
    hist = hist[hist.fecha_mes < base.fecha_mes.min()]
    filled, audit, stats = ex.seasonal_fill(base, "estacional_52", hist)
    filled = filled.sort_values(["fecha_mes", "ubigeo"]).reset_index(drop=True)
    year = filled.fecha_mes.dt.year
    fit, val = filled[year <= 2015], filled[(year >= 2016)&(year <= 2018)]
    train, test = filled[year <= 2018], filled[year >= 2019]
    assert test.ubigeo.nunique() == 70 and len(test) == 5040
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42).fit(train[ex.pp.FEATURES], train.casos)
    original = model.predict(test[ex.pp.FEATURES])
    original_rmse = float(np.sqrt(ex.mean_squared_error(test.casos, original)))
    reference = pd.read_csv(ex.ROOT/"results/arbol_metricas.csv")
    saved = reference[(reference.Tarea == "Regresión")&(reference.Conjunto == "Prueba")&reference.Modelo.str.contains("M2")].iloc[0].RMSE
    assert abs(original_rmse-saved) < 1e-9, (original_rmse, saved)
    rows = [dict(model="arbol_original_solo_casos", candidate=5, n_test=len(test), districts=70,
                 RMSE=original_rmse, MAE=float(ex.mean_absolute_error(test.casos, original)), climate_fraction=0.)]
    predictions = test[["ubigeo", "fecha_mes", "casos"]].copy()
    predictions["arbol_original"] = original
    for variant in ["solo_casos", "estacional_52"]:
        options = []
        for candidate in (0, 1):
            p, _ = ex.train_predict(fit, val, "random_forest", "casos", candidate, variant)
            options.append((float(np.sqrt(ex.mean_squared_error(val.casos, p))), candidate))
        score, candidate = min(options)
        p, used = ex.train_predict(train, test, "random_forest", "casos", candidate, variant)
        rows.append(dict(model="random_forest_"+variant, candidate=candidate, validation_RMSE=score,
                         n_test=len(test), districts=70, RMSE=float(np.sqrt(ex.mean_squared_error(test.casos,p))),
                         MAE=float(ex.mean_absolute_error(test.casos,p)), climate_fraction=float(used.mean())))
        predictions[variant] = p
    pd.DataFrame(rows).to_csv(output/"metricas.csv", index=False)
    predictions.to_csv(output/"predicciones.csv", index=False)
    audit.to_csv(output/"auditoria_imputacion.csv", index=False)
    (output/"ejecucion.json").write_text(json.dumps(dict(imputation=stats, original_tree_reproduced=True,
                 cutoff_train=2018, test="2019–2024", note="Comparación exploratoria con el alcance original de 70 distritos"), indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
