"""Réplica del modelo del paper 4 (El Alaoui et al., 2027).

El paper pronostica los casos mensuales de leishmaniosis cutánea de la provincia
de Errachidia (Marruecos) con un modelo SARIMA construido con el método de
Box-Jenkins:

1. pruebas de estacionariedad (ADF y KPSS) para decidir la diferenciación;
2. búsqueda en cuadrícula de los órdenes, eligiendo el menor AIC;
3. prueba de Ljung-Box sobre los residuos (p > 0,05 = ruido blanco);
4. evaluación con RMSE y MAE.

El modelo elegido en el paper fue SARIMA(0,1,0)(1,1,0)12.

Aquí se implementan dos variantes, ambas solo con el historial de casos (como el
paper) y una serie por distrito:

- ``SARIMA_paper``: el orden del paper, (0,1,0)(1,1,0)12, en todos los distritos.
- ``SARIMA_AIC``: el procedimiento del paper aplicado a cada distrito, es decir,
  cuadrícula p, q, P, Q en {0, 1} y d en {0, 1}, con D = 1 y s = 12, eligiendo
  el menor AIC en el tramo de ajuste.

Diferencias con el paper (por la naturaleza de este proyecto):

- 57 series distritales en lugar de una serie provincial; se ajusta un SARIMA
  por distrito.
- Partición temporal del proyecto (ajuste, validación y prueba) en lugar de
  80/20, para comparar con los demás modelos en las mismas filas.
- Pronóstico a un paso: cada mes se predice con los parámetros ya ajustados y
  los casos observados hasta el mes anterior, igual que los demás modelos (que
  usan los rezagos observados). Para la prueba se reajusta con ajuste +
  validación, con el mismo orden.
- El paper solo hace regresión; la alerta se evalúa usando los casos predichos
  como puntaje, con el umbral elegido en validación (igual que el paper 2).

Uso (desde la raíz del repositorio)::

    python src/modelo_paper4.py                        # protocolo oficial -> results/paper4/
    python src/modelo_paper4.py --protocolo historico  # -> results/paper4_historico/
"""
from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from evaluation import choose_threshold, labels, metrics
from experimento_clima import DATA, PROTOCOLS

ROOT = Path(__file__).resolve().parents[1]
S = 12
PAPER_ORDER = ((0, 1, 0), (1, 1, 0, S))  # SARIMA(0,1,0)(1,1,0)12 elegido en el paper
GRID = [((p, d, q), (P, 1, Q, S)) for p, d, q, P, Q in itertools.product((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))]
FALLBACK_ORDER = ((0, 1, 0), (0, 1, 0, S))  # si un ajuste falla: paseo aleatorio estacional


def _model(y, order):
    return SARIMAX(y, order=order[0], seasonal_order=order[1],
                   enforce_stationarity=False, enforce_invertibility=False)


def fit(y, order):
    """Ajusta SARIMA sobre ``y`` (serie mensual de un distrito); None si falla."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return _model(y, order).fit(disp=False, maxiter=200)
        except Exception:
            return None


def select_order(y):
    """Box-Jenkins del paper: menor AIC de la cuadrícula y Ljung-Box de sus residuos."""
    best = None
    for order in GRID:
        res = fit(y, order)
        if res is None or not np.isfinite(res.aic):
            continue
        if best is None or res.aic < best[1].aic:
            best = (order, res)
    if best is None:
        return FALLBACK_ORDER, np.nan, np.nan
    order, res = best
    resid = res.resid[S + 1:]  # se descartan los primeros meses afectados por la diferenciación
    lb = float(acorr_ljungbox(resid, lags=[S], return_df=True).lb_pvalue.iloc[0]) if len(resid) > S else np.nan
    return order, float(res.aic), lb


def one_step(y_fit, y_all, order):
    """Pronóstico a un paso para los meses de ``y_all`` posteriores a ``y_fit``.

    Los parámetros se estiman solo con ``y_fit``; luego se aplican (sin reajustar)
    a la serie completa, de modo que cada mes se predice con los casos
    observados hasta el mes anterior.
    """
    res = fit(y_fit, order) or fit(y_fit, FALLBACK_ORDER)
    if res is None:  # último recurso: repetir el mismo mes del año anterior
        return y_all.shift(S).iloc[len(y_fit):].fillna(0).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = res.apply(y_all.to_numpy(dtype=float))
        pred = full.get_prediction(start=len(y_fit), end=len(y_all) - 1, dynamic=False).predicted_mean
    return np.maximum(np.asarray(pred, dtype=float), 0)


def district(ubigeo, y, cfg):
    """Ajuste, validación y prueba de un distrito, con ambas variantes."""
    y = y.astype(float)
    years = y.index.year
    y_tr, y_val_end = y[years <= cfg["fit"]], y[years <= cfg["val"]]
    y_full = y[years <= cfg["end"]]
    order_aic, aic, lb = select_order(y_tr)
    out = {}
    for name, order in (("SARIMA_paper", PAPER_ORDER), ("SARIMA_AIC", order_aic)):
        pv = one_step(y_tr, y_val_end, order)
        pt = one_step(y_val_end, y_full, order)
        out[name] = (order, pv, pt)
    return ubigeo, out, dict(ubigeo=ubigeo, orden_AIC=f"{order_aic[0]}x{order_aic[1]}", AIC=aic,
                             ljung_box_p=lb)


def order_label(order):
    (p, d, q), (P, D, Q, s) = order
    return f"({p},{d},{q})({P},{D},{Q}){s}"


def main(protocol: str, out: Path, n_jobs: int = 4):
    cfg = PROTOCOLS[protocol]
    d = pd.read_csv(DATA / protocol / "sin_imputar.csv", dtype={"ubigeo": str},
                    parse_dates=["fecha_mes"], low_memory=False)
    d = d.sort_values(["fecha_mes", "ubigeo"]).reset_index(drop=True)
    year = d.fecha_mes.dt.year
    tr, va = d[year <= cfg["fit"]], d[(year > cfg["fit"]) & (year <= cfg["val"])]
    full, te = d[year <= cfg["val"]], d[year > cfg["val"]]
    scale = max(float((full.casos - full.lag12).abs().mean()), 1e-12)
    yv_alert, yt_alert = labels(va, tr), labels(te, full)
    series = {u: g.set_index("fecha_mes").casos.asfreq("MS") for u, g in d.groupby("ubigeo")}
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        results = Parallel(n_jobs=n_jobs)(delayed(district)(u, y, cfg) for u, y in series.items())
    print(f"{protocol}: {len(results)} distritos ajustados en {time.time() - t0:.0f} s", flush=True)
    diag = pd.DataFrame([r[2] for r in results])
    out.mkdir(parents=True, exist_ok=True)
    diag.to_csv(out / "ordenes_por_distrito.csv", index=False)

    rows, predictions, tuning = [], [], []
    for name in ("SARIMA_paper", "SARIMA_AIC"):
        # Reordena las predicciones por distrito a las filas (mes, distrito) de validación y prueba.
        val_map, test_map = {}, {}
        for ubigeo, res, _ in results:
            order, pv, pt = res[name]
            val_dates = series[ubigeo].index[(series[ubigeo].index.year > cfg["fit"]) &
                                             (series[ubigeo].index.year <= cfg["val"])]
            test_dates = series[ubigeo].index[(series[ubigeo].index.year > cfg["val"]) &
                                              (series[ubigeo].index.year <= cfg["end"])]
            val_map.update({(ubigeo, t): v for t, v in zip(val_dates, pv)})
            test_map.update({(ubigeo, t): v for t, v in zip(test_dates, pt)})
        pv = np.array([val_map[(u, t)] for u, t in zip(va.ubigeo, va.fecha_mes)])
        p = np.array([test_map[(u, t)] for u, t in zip(te.ubigeo, te.fecha_mes)])
        val_rmse = float(np.sqrt(mean_squared_error(va.casos, pv)))
        threshold = choose_threshold(yv_alert, pv)
        orden = order_label(PAPER_ORDER) if name == "SARIMA_paper" else "menor AIC por distrito"
        tuning.append(dict(modelo=name, entradas="casos", orden=orden, val_RMSE=val_rmse))
        rows.append(dict(modelo=name, entradas="casos", RMSE_validacion=val_rmse, orden=orden,
                         RMSE_5semillas_media=np.nan, RMSE_5semillas_sd=np.nan, uso_clima_prueba=0.0,
                         **metrics("casos", te.casos.to_numpy(), p, np.nan, scale),
                         **{f"alerta_{k}": v for k, v in metrics("alerta", yt_alert, p, threshold, scale).items()}))
        predictions.append(te[["ubigeo", "fecha_mes"]].assign(
            modelo=name, entradas="casos", casos_observados=te.casos.to_numpy(), casos_predichos=p,
            alerta_observada=yt_alert, umbral_alerta=threshold))
        print(f"{name}: val RMSE {val_rmse:.3f} test RMSE {rows[-1]['RMSE']:.3f} MAE {rows[-1]['MAE']:.3f} "
              f"PR-AUC {rows[-1]['alerta_PR_AUC']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "metricas.csv", index=False)
    pd.DataFrame(tuning).to_csv(out / "validacion.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "predicciones.csv", index=False)
    return pd.DataFrame(rows), diag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocolo", choices=list(PROTOCOLS), default="oficial")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    main(args.protocolo, ROOT / ("results/paper4" if args.protocolo == "oficial" else f"results/paper4_{args.protocolo}"),
         args.n_jobs)
