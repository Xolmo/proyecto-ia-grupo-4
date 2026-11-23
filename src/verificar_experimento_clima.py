"""Auditoría independiente de los CSV finales y de la comparación completa."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/experimento_clima"
OUT = ROOT / "results/experimento_clima"
VARS = ["pp_total_mm", "tmax_media", "tmin_media"]


def main():
    rows = []
    for protocol in ["historico", "oficial"]:
        base = pd.read_csv(DATA / protocol / "sin_imputar.csv", dtype={"ubigeo": str}, low_memory=False)
        for variant in ["sin_imputar", "estacional_52", "huecos_cortos"]:
            d = pd.read_csv(DATA / protocol / f"{variant}.csv", dtype={"ubigeo": str}, low_memory=False)
            pd.testing.assert_frame_equal(base[["ubigeo", "fecha_mes", "casos"]], d[["ubigeo", "fecha_mes", "casos"]])
            assert not d.duplicated(["ubigeo", "fecha_mes"]).any()
            filled = 0
            for v in VARS:
                known = base[v].notna()
                np.testing.assert_array_equal(base.loc[known, v], d.loc[known, v])
                changed = base[v].isna() & d[v].notna()
                np.testing.assert_array_equal(changed, d[v + "_imputado"])
                filled += int(changed.sum())
            if variant == "estacional_52":
                assert filled == round(int(base[VARS].isna().sum().sum()) * .52)
            if variant == "sin_imputar":
                assert filled == 0
            else:
                audit = pd.read_csv(DATA / protocol / f"{variant}_auditoria.csv", parse_dates=["fecha_mes", "fuente_fecha"])
                assert len(audit) == filled
                assert (audit.fuente_fecha < audit.fecha_mes).all()
                if variant == "huecos_cortos":
                    assert not audit.fuente_imputada.any()
            invalid_t = int((d.tmax_media < d.tmin_media).sum())
            negative_rain = int((d.pp_total_mm < 0).sum())
            assert invalid_t == negative_rain == 0
            rows.append(dict(protocol=protocol, variant=variant, rows=len(d), filled_cells=filled,
                             invalid_temperature=invalid_t, negative_rain=negative_rain))
    m = pd.read_csv(OUT / "metricas.csv")
    p = pd.read_csv(OUT / "predicciones.csv", dtype={"ubigeo": str})
    actual = m[m.variant != "referencia"]
    assert len(actual) == 80
    assert not actual.duplicated(["protocol", "variant", "family", "task"]).any()
    assert np.isfinite(p.predicted).all()
    assert (p.loc[p.task == "casos", "predicted"] >= 0).all()
    assert p.loc[p.task == "alerta", "predicted"].between(0, 1).all()
    assert (p.groupby(["protocol", "task", "ubigeo", "fecha_mes"])["observed"].nunique() == 1).all()
    sizes = p.groupby(["protocol", "task", "variant", "family"]).size()
    assert (sizes.groupby(level=["protocol", "task"]).nunique() == 1).all()
    for protocol in ["historico", "oficial"]:
        for task in ["casos", "alerta"]:
            group = p[(p.protocol == protocol) & (p.task == task)]
            reference = None
            for _, g in group.groupby(["variant", "family"]):
                keys = set(zip(g.ubigeo, g.fecha_mes))
                assert len(g) == len(keys)
                if reference is None:
                    reference = keys
                assert keys == reference
    no_obs = actual[(actual.protocol == "oficial") & actual.variant.isin(["sin_imputar", "huecos_cortos"])]
    assert (no_obs.climate_fraction == 0).all()
    pd.DataFrame(rows).to_csv(OUT / "integridad_datasets.csv", index=False)
    report = dict(status="OK", combinations=80, prediction_rows=len(p),
                  same_test_rows=True, original_observations_preserved=True,
                  physical_constraints=True, audited_sources_strictly_past=True,
                  official_without_observations_uses_case_fallback=True)
    (OUT / "verificacion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
