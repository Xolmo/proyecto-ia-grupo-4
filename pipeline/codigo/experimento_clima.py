"""Experimento aislado: tres datasets climáticos, cinco familias, dos tareas.

Ejecutar desde la raíz: python src/experimento_clima.py
No modifica los notebooks ni sus resultados existentes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

import integracion_clima as ic
import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/experimento_clima"
OUT = ROOT / "results/experimento_clima"
RAW = ROOT / "data/raw/datos_abiertos_vigilancia_leishmaniosis_2000_2024.csv"
CLIMATE = ["pp_total_mm", "tmax_media", "tmin_media"]
CF = [f"{v}_lag{k}" for v in CLIMATE for k in (1, 2, 3)]
VARIANTS = ["sin_imputar", "estacional_52", "huecos_cortos"]
PROTOCOLS = {
    "historico": {"fit": 2009, "val": 2011, "end": 2014},
    "oficial": {"fit": 2015, "val": 2018, "end": 2024},
}
SEED = 42
# Revisión final: con 150 iteraciones 41/41 ajustes llegaban al límite sin converger.
# Se usa early stopping: 10 % del tramo de ajuste se reserva internamente y el
# entrenamiento se detiene tras 20 épocas sin mejora (nunca usa validación ni prueba).
MLP_MAX_ITER = 1000
FALLBACK_CACHE = {}
FIT_DIAGNOSTICS = []


def seasonal_fill(frame, mode, history=None):
    """Relleno causal, trazable por celda; nunca utiliza valores futuros.

    52%: mismo mes a 12/24/36/48 meses, permite antecedentes imputados
    y los marca. Cortos: mediana de antecedentes OBSERVADOS, máximo dos
    meses consecutivos; no recurre a estimaciones anteriores.
    """
    d = frame.sort_values(["ubigeo", "fecha_mes"]).reset_index(drop=True).copy()
    original = d[CLIMATE].copy()
    initial = int(original.isna().sum().sum())
    quota = round(initial * .52) if mode == "estacional_52" else initial
    lookup = {(r.ubigeo, r.fecha_mes): i for i, r in d.iterrows()}
    prior = {} if history is None else {
        (r.ubigeo, r.fecha_mes): r for _, r in history.iterrows()}
    audit = []
    for v in CLIMATE:
        d[v + "_imputado"] = False
        d[v + "_fuente_fecha"] = pd.NaT
        d[v + "_fuente_imputada"] = False
    def valid_value(i, variable, value):
        if not np.isfinite(value):
            return False
        if variable == "pp_total_mm":
            return value >= 0
        other = "tmin_media" if variable == "tmax_media" else "tmax_media"
        bound = d.loc[i, other]
        return pd.isna(bound) or (value >= bound if variable == "tmax_media" else value <= bound)
    # Orden global cronológico: el presupuesto no privilegia un distrito.
    for i in d.sort_values(["fecha_mes", "ubigeo"]).index:
        r = d.loc[i]
        for v in CLIMATE:
            if pd.notna(original.loc[i, v]) or len(audit) >= quota or mode == "sin_imputar":
                continue
            if mode == "huecos_cortos":
                prev = [lookup.get((r.ubigeo, r.fecha_mes - pd.DateOffset(months=k))) for k in (1, 2)]
                # Primer o segundo hueco; al tercero se conserva nulo.
                if all(j is None or pd.isna(original.loc[j, v]) for j in prev):
                    continue
            sources = []
            for k in (12, 24, 36, 48):
                date = r.fecha_mes - pd.DateOffset(months=k)
                key = (r.ubigeo, date)
                j = lookup.get(key)
                if j is None:
                    if key in prior and pd.notna(prior[key][v]) and valid_value(i, v, prior[key][v]):
                        sources.append((date, float(prior[key][v]), False))
                        if mode == "estacional_52":
                            break
                    continue
                value = d.loc[j, v] if mode == "estacional_52" else original.loc[j, v]
                if pd.notna(value) and valid_value(i, v, value):
                    sources.append((date, float(value), bool(d.loc[j, v + "_imputado"])))
                    if mode == "estacional_52":
                        break
            if not sources:
                continue
            source_date = sources[0][0]
            value = sources[0][1] if mode == "estacional_52" else float(np.median([s[1] for s in sources]))
            d.loc[i, v] = value
            d.loc[i, v + "_imputado"] = True
            d.loc[i, v + "_fuente_fecha"] = source_date
            d.loc[i, v + "_fuente_imputada"] = sources[0][2]
            audit.append({"ubigeo": r.ubigeo, "fecha_mes": r.fecha_mes, "variable": v,
                          "valor": value, "fuente_fecha": source_date,
                          "fuente_imputada": sources[0][2],
                          "n_fuentes": len(sources)})
    # Variables derivadas consistentes; no se imputan independientemente.
    d["pp_media_mm"] = d.pp_total_mm / d.fecha_mes.dt.days_in_month
    d["tmed_media"] = (d.tmax_media + d.tmin_media) / 2
    for v in CLIMATE + ["tmed_media"]:
        for k in (1, 2, 3):
            d[f"{v}_lag{k}"] = d.groupby("ubigeo")[v].shift(k)
    d["tiene_clima"] = d.pp_total_mm.notna()
    # No publicar la antigua etiqueta con otro corte como si fuera actual.
    d = d.drop(columns=["umbral", "alerta"], errors="ignore")
    stats = {"variant": mode, "rows": len(d), "initial_missing_cells": initial,
             "target_cells": quota if mode == "estacional_52" else None,
             "filled_cells": len(audit), "fraction_missing_filled": len(audit) / initial,
             "remaining_missing_cells": int(d[CLIMATE].isna().sum().sum()),
             "recursive_sources": sum(a["fuente_imputada"] for a in audit)}
    if mode == "estacional_52" and len(audit) != quota:
        raise ValueError(f"No hay suficientes antecedentes para 52%: {stats}")
    # Invariantes de integridad y disponibilidad.
    assert d[CLIMATE].where(original.notna()).equals(original)
    assert all(a["fuente_fecha"] < a["fecha_mes"] for a in audit)
    assert not (d.tmax_media < d.tmin_media).any()
    assert not (d.pp_total_mm < 0).any()
    if mode == "sin_imputar":
        assert not audit
    return d, pd.DataFrame(audit), stats


def build_datasets():
    DATA.mkdir(parents=True, exist_ok=True)
    cases = pp.clean_cases(pp.load_raw(RAW))
    cases = cases[cases.departamento == "CAJAMARCA"].copy()
    ext = ROOT / "data/raw/external"
    stations = pd.read_csv(ext / "estaciones_senamhi.csv")
    daily = ic.clima_diario(ROOT / "data/raw/clima", stations)
    monthly = ic.clima_mensual(daily)
    ccpp = pd.read_csv(ext / "centros_poblados_cajamarca.csv", dtype=str)
    districts = pd.read_csv(ext / "distritos_cajamarca.csv", dtype=str)
    result, summary = {}, []
    for protocol, cfg in PROTOCOLS.items():
        cutoff = pd.Timestamp(f"{cfg['fit']}-12-31")
        station_summary = ic.resumen_estaciones(monthly[monthly.fecha_mes <= cutoff])
        # Selección espacial usando SOLO el tramo de ajuste, no validación/prueba.
        eligible = cases[cases.fecha <= cutoff].groupby("ubigeo").size()
        eligible = eligible[eligible >= pp.UMBRAL_CASOS].index
        selected = cases[cases.ubigeo.isin(eligible)].copy()
        ic.TRAIN_END = str(cutoff.date())
        loc = ic.estacion_mas_cercana(ic.coordenadas_localidades(
            selected[selected.fecha <= cutoff], ccpp, districts), station_summary)
        assignment = ic.asignar_distritos(loc)
        base = ic.dataset_final(selected, monthly, assignment, train_end=str(cutoff.date()))
        base = base[base.fecha_mes.dt.year <= cfg["end"]].copy()
        history = ic.clima_por_distrito(monthly, assignment)
        history = history[history.fecha_mes < base.fecha_mes.min()]
        for variant in VARIANTS:
            d, audit, stats = seasonal_fill(base, variant, history)
            path = DATA / protocol
            path.mkdir(exist_ok=True)
            d.to_csv(path / f"{variant}.csv", index=False)
            audit.to_csv(path / f"{variant}_auditoria.csv", index=False)
            stats.update(protocol=protocol, districts=d.ubigeo.nunique())
            summary.append(stats)
            result[protocol, variant] = d
        assignment.to_csv(DATA / protocol / "asignacion_estaciones.csv", index=False)
    pd.DataFrame(summary).to_csv(DATA / "resumen_imputacion.csv", index=False)
    return result


class TemporalCNN:
    """Conv1D sobre tres meses anteriores; contexto anual en una rama densa."""
    def __init__(self, climate, task, width):
        self.climate, self.task, self.width = climate, task, width

    def fit(self, X, y):
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        self.scaler = StandardScaler().fit(X)
        a = self.scaler.transform(X).astype("float32")
        channels = 4 if self.climate else 1
        # F=[lag1,lag2,lag3,lag12,media3,sin,cos], luego 3 lags por clima.
        self.seq_cols = [0, 1, 2] + (list(range(7, 16)) if self.climate else [])
        width, context = self.width, 4
        self.conv = torch.nn.Sequential(torch.nn.Conv1d(channels, width, 2), torch.nn.ReLU(), torch.nn.Flatten())
        self.head = torch.nn.Sequential(torch.nn.Linear(width * 2 + context, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        self.model = torch.nn.ModuleList([self.conv, self.head])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.005, weight_decay=.001)
        tx = torch.from_numpy(a)
        ty = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        if self.task == "alerta":
            pos = max(float(ty.sum()), 1.)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y)-pos)/pos))
        else:
            loss_fn = torch.nn.MSELoss()
        generator = torch.Generator().manual_seed(SEED)
        self.model.train()
        for _ in range(50):
            for ids in torch.randperm(len(y), generator=generator).split(512):
                optimizer.zero_grad()
                seq = tx[ids][:, self.seq_cols].reshape(-1, channels, 3).flip(-1)
                z = self.head(torch.cat([self.conv(seq), tx[ids][:, 3:7]], dim=1))
                loss = loss_fn(z, ty[ids])
                loss.backward()
                optimizer.step()
        return self

    def predict_score(self, X):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            seq = x[:, self.seq_cols].reshape(-1, 4 if self.climate else 1, 3).flip(-1)
            z = self.head(torch.cat([self.conv(seq), x[:, 3:7]], dim=1)).flatten()
            return (torch.sigmoid(z) if self.task == "alerta" else z).numpy()


def make_model(family, task, candidate, climate):
    cls = task == "alerta"
    if family == "cnn":
        return TemporalCNN(climate, task, [8, 16][candidate])
    if family == "lineal":
        m = (LogisticRegression(C=[.1, 1.][candidate], max_iter=1000, class_weight="balanced")
             if cls else Ridge(alpha=[1., 10.][candidate]))
    elif family == "arbol":
        params = dict(max_depth=[3, 5][candidate], min_samples_leaf=20, random_state=SEED)
        m = DecisionTreeClassifier(**params, class_weight="balanced") if cls else DecisionTreeRegressor(**params)
    elif family == "random_forest":
        params = dict(n_estimators=100, max_depth=8, min_samples_leaf=[5, 20][candidate],
                      random_state=SEED, n_jobs=2)
        m = RandomForestClassifier(**params, class_weight="balanced") if cls else RandomForestRegressor(**params)
    elif family == "mlp":
        params = dict(hidden_layer_sizes=[(32,), (32, 16)][candidate], alpha=.01,
                      max_iter=MLP_MAX_ITER, early_stopping=True, validation_fraction=.1,
                      n_iter_no_change=20, shuffle=False, random_state=SEED)
        m = MLPClassifier(**params) if cls else MLPRegressor(**params)
    else:
        raise ValueError(family)
    return make_pipeline(StandardScaler(), m) if family in ("lineal", "mlp") else m


def features(d, climate, short=False, train=None):
    f = d[pp.FEATURES + (CF if climate else [])].copy()
    # Propuesta propia: anomalías respecto a la climatología del entrenamiento.
    if climate and short:
        assert train is not None
        for c in CF:
            center = train.groupby(train.fecha_mes.dt.month)[c].median()
            f[c] = f[c] - d.fecha_mes.dt.month.map(center)
    return f


def labels(d, train):
    stats = train.groupby("ubigeo").casos.agg(["mean", "std"])
    threshold = d.ubigeo.map(stats["mean"] + 2 * stats["std"])
    return (d.casos > threshold).astype(int).to_numpy()


def train_predict(train, target, family, task, candidate, variant):
    """Siempre hay predicción para las mismas filas; respaldo de casos si falta clima."""
    climate = variant != "solo_casos"
    short = variant == "huecos_cortos"
    y = labels(train, train) if task == "alerta" else train.casos.to_numpy()
    x = features(train, False)
    xt = features(target, False)
    key = (len(train), str(train.fecha_mes.max()), len(target),
           str(target.fecha_mes.min()), family, task, candidate)
    if key not in FALLBACK_CACHE:
        fallback = make_model(family, task, candidate, False)
        fit_one(fallback, x, y, task, family)
        FALLBACK_CACHE[key] = score_one(fallback, xt, task)
    pred = FALLBACK_CACHE[key].copy()
    used = np.zeros(len(target), dtype=bool)
    if climate:
        xc = features(train, True, short, train)
        xct = features(target, True, short, train)
        keep = xc.notna().all(axis=1).to_numpy()
        used = xct.notna().all(axis=1).to_numpy(copy=True)
        # No imputación oculta: todas las familias reciben filas completas.
        if keep.sum() >= 100 and used.any() and (task != "alerta" or len(np.unique(y[keep])) == 2):
            model = make_model(family, task, candidate, True)
            fit_one(model, xc.loc[keep], y[keep], task, family)
            pred[used] = score_one(model, xct.loc[used], task)
        else:
            used[:] = False
    return np.maximum(pred, 0) if task == "casos" else pred, used


def fit_one(model, x, y, task, family):
    if family == "mlp" and task == "alerta":
        weights = compute_sample_weight("balanced", y)
        model.fit(x, y, mlpclassifier__sample_weight=weights)
    else:
        model.fit(x, y)
    if family == "mlp":
        fitted = model.steps[-1][1]
        FIT_DIAGNOSTICS.append(dict(family=family, task=task, n_train=len(y),
                                    n_iter=fitted.n_iter_, max_iter=fitted.max_iter,
                                    hit_iteration_limit=fitted.n_iter_ >= fitted.max_iter))


def score_one(model, x, task):
    if isinstance(model, TemporalCNN):
        return model.predict_score(x).astype(float)
    return model.predict_proba(x)[:, 1] if task == "alerta" else model.predict(x)


def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])]) if len(thresholds) else .5


def metrics(task, y, p, threshold, scale):
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


def run_experiments(datasets):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tuning, predictions = [], [], []
    families = ["lineal", "arbol", "random_forest", "mlp", "cnn"]
    for protocol, cfg in PROTOCOLS.items():
        baseline = datasets[protocol, "sin_imputar"]
        for variant in ["solo_casos"] + VARIANTS:
            d = (baseline if variant == "solo_casos" else datasets[protocol, variant]).copy()
            d = d.sort_values(["fecha_mes", "ubigeo"]).reset_index(drop=True)
            year = d.fecha_mes.dt.year
            tr, va = d[year <= cfg["fit"]], d[(year > cfg["fit"]) & (year <= cfg["val"])]
            full, te = d[year <= cfg["val"]], d[year > cfg["val"]]
            for family in families:
                for task in ["casos", "alerta"]:
                    print(f"{protocol} {variant} {family} {task}", flush=True)
                    yv = labels(va, tr) if task == "alerta" else va.casos.to_numpy()
                    yt = labels(te, full) if task == "alerta" else te.casos.to_numpy()
                    options = []
                    for candidate in (0, 1):
                        p, used = train_predict(tr, va, family, task, candidate, variant)
                        objective = float(average_precision_score(yv, p)) if task == "alerta" else float(np.sqrt(mean_squared_error(yv, p)))
                        options.append((objective, candidate, p))
                        tuning.append(dict(protocol=protocol, variant=variant, family=family, task=task,
                                           candidate=candidate, validation_score=objective, climate_fraction=float(used.mean())))
                    best = sorted(options, key=lambda z: (-z[0] if task == "alerta" else z[0], z[1]))[0]
                    val_score, candidate, pv = best
                    threshold = choose_threshold(yv, pv) if task == "alerta" else np.nan
                    p, used = train_predict(full, te, family, task, candidate, variant)
                    scale = max(float((full.casos - full.lag12).abs().mean()), 1e-12)
                    row = dict(protocol=protocol, variant=variant, family=family, task=task,
                               candidate=candidate, validation_score=val_score, threshold=threshold,
                               n_train=len(full), n_test=len(te), districts=d.ubigeo.nunique(),
                               climate_fraction=float(used.mean()), **metrics(task, yt, p, threshold, scale))
                    rows.append(row)
                    pr = te[["ubigeo", "fecha_mes"]].copy()
                    pr = pr.assign(protocol=protocol, variant=variant, family=family, task=task,
                                   observed=yt, predicted=p, climate_used=used)
                    predictions.append(pr)
                    pd.DataFrame(rows).to_csv(OUT / "metricas.csv", index=False)
        # Referencias idénticas sobre las filas de prueba, sin reentrenamiento.
        d = baseline.sort_values(["fecha_mes", "ubigeo"])
        full = d[d.fecha_mes.dt.year <= cfg["val"]]
        te = d[d.fecha_mes.dt.year > cfg["val"]]
        scale = max(float((full.casos-full.lag12).abs().mean()), 1e-12)
        for name, p in [("ultimo_mes", te.lag1), ("estacional", te.lag12), ("cero", te.casos * 0)]:
            rows.append(dict(protocol=protocol, variant="referencia", family=name, task="casos",
                             n_train=len(full), n_test=len(te), climate_fraction=0.,
                             **metrics("casos", te.casos, p, np.nan, scale)))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "metricas.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "validacion.csv", index=False)
    pd.DataFrame(FIT_DIAGNOSTICS).to_csv(OUT / "diagnostico_ajuste.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "predicciones.csv", index=False)
    # Selección exclusivamente por validación; test solo describe el resultado.
    winners = []
    for (protocol, task), g in result[result.variant != "referencia"].groupby(["protocol", "task"]):
        winners.append(g.sort_values("validation_score", ascending=task == "casos", kind="stable").iloc[0])
    pd.DataFrame(winners).to_csv(OUT / "seleccion_por_validacion.csv", index=False)
    metadata = dict(seed=SEED, python=platform.python_version(), sklearn=sklearn.__version__,
                    torch=torch.__version__, pandas=pd.__version__, numpy=np.__version__,
                    raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(), protocols=PROTOCOLS,
                    cnn_epochs=50, candidate_count=2, note="Sin modificar notebooks existentes")
    (OUT / "ejecucion.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-only", action="store_true")
    parser.add_argument("--reuse-datasets", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # CPU determinista y pocos hilos para no monopolizar el equipo.
    torch.set_num_threads(2)
    if args.reuse_datasets:
        datasets = {(p, v): pd.read_csv(DATA / p / f"{v}.csv", dtype={"ubigeo": str}, parse_dates=["fecha_mes"], low_memory=False)
                    for p in PROTOCOLS for v in VARIANTS}
    else:
        datasets = build_datasets()
    if not args.datasets_only:
        with threadpool_limits(limits=2):
            run_experiments(datasets)
