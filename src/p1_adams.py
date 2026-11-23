"""Método del paper del integrante 1 (Adams et al., 2025) aplicado a Cajamarca.

Adams Q. H. et al. (2025). Evaluating the contribution of weather variables to
machine learning forecasts of visceral leishmaniasis in Brazil. Environ. Res.:
Health 3 045012. PDF en papers/paper_integrante1.pdf.

Qué se replica del paper (§2.2-2.3):
  * XGBoost de regresión (casos) y de clasificación (riesgo por umbral).
  * Ventana deslizante: se entrena con 24 meses y se predice el bloque siguiente.
  * Casos rezagados 1-12, media móvil de 36 meses, clima rezagado 0-6,
    covariables estáticas estandarizadas e interacciones clima x estática.
  * Optimización bayesiana de hiperparámetros (Optuna, TPE).
  * 100 réplicas bootstrap -> mediana e intervalo de predicción del 95 %; cobertura.
  * Riesgo = media de 36 meses >= umbral (2,4 casos en el paper).
  * Importancia por SHAP (TreeSHAP nativo de XGBoost, pred_contribs=True).

Adaptaciones explícitas (no son del paper):
  * modo="operativo" (por defecto): todo lo que entra al modelo para predecir el
    mes t a horizonte h se conoce en t-h. El paper indexa los rezagos respecto del
    mes objetivo (clima del propio mes, casos t-1) porque su ejercicio es
    retrospectivo; eso se conserva como modo="paper" solo como referencia.
  * Modelo conjunto de distritos por ventana (24 meses x n distritos). Con 0,67
    casos/mes de media, un modelo por distrito con 24 filas casi no aprende; la
    réplica literal por distrito queda disponible con por_distrito=True.
  * Hiperparámetros elegidos solo con ventanas de validación, nunca de prueba.

Entradas: se leen del pipeline base del equipo (src/ y data/processed/), que ya
limpia los casos, asigna estaciones SENAMHI y resume el clima por mes. Salidas:
tablas en results/p1_adams/; los gráficos los guarda el notebook 05 en
results/plots/p1_*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT                      # datos del pipeline base del equipo
RAW = BASE / "data/raw/datos_abiertos_vigilancia_leishmaniosis_2000_2024.csv"
EXTERNAL = BASE / "data/raw/external"
PROCESSED = BASE / "data/processed"
ONI_PATH = ROOT / "data/raw/external/oni.ascii.txt"   # opcional (NOAA CPC)
OUT = ROOT / "results/p1_adams"

SEED = 42
N_JOBS = 2                       # mismo límite de hilos que el pipeline base
MESES_TRAIN = 24                 # paper: ventana de entrenamiento de 24 meses
CLIMA = ["pp_total_mm", "tmax_media", "tmin_media"]
ESTATICAS = ["altitud", "latitud", "longitud", "pct_urbano"]
INTERACCION_CON = ("altitud", "pct_urbano")
UMBRAL_MOH_BRASIL = 2.4          # paper: transmisión esporádica -> moderada


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------
def distritos_70() -> list[str]:
    """Los 70 distritos con al menos 50 casos usados por los modelos del equipo."""
    d = pd.read_csv(PROCESSED / "dataset_cajamarca.csv", dtype={"ubigeo": str}, usecols=["ubigeo"])
    return sorted(d.ubigeo.unique())


def distritos_protocolo(protocolo: str) -> list[str]:
    """Distritos del experimento de clima del equipo.

    historico: 46 distritos con >= 50 casos hasta 2009 (prueba 2012-2014, con clima observado).
    oficial: 57 distritos con >= 50 casos hasta 2015 (prueba 2019-2024).
    """
    d = pd.read_csv(PROCESSED / f"experimento_clima/{protocolo}/sin_imputar.csv",
                    dtype={"ubigeo": str}, usecols=["ubigeo"])
    return sorted(d.ubigeo.unique())


def panel_casos(ubigeos: list[str]) -> pd.DataFrame:
    """Panel distrito x mes 2000-01..2024-12 con ceros de notificación.

    Usa la limpieza del equipo (src/preprocessing.py). A diferencia del panel
    del pipeline base no descarta 2000: los rezagos largos (hasta 23 meses y la media de 36)
    necesitan esa historia; si falta, queda NaN y XGBoost lo maneja de forma nativa.
    """
    cases = pp.clean_cases(pp.load_raw(RAW))
    cases = cases[cases.departamento == "CAJAMARCA"]
    meses = pd.period_range("2000-01", "2024-12", freq="M")
    conteo = (cases[cases.ubigeo.isin(ubigeos)].groupby(["ubigeo", "periodo"]).size()
              .unstack("ubigeo").reindex(index=meses, columns=ubigeos, fill_value=0)
              .fillna(0).astype(int))
    panel = conteo.stack().rename("casos").reset_index()
    panel.columns = ["periodo", "ubigeo", "casos"]
    panel["fecha_mes"] = panel.periodo.dt.to_timestamp()
    nombres = cases.drop_duplicates("ubigeo").set_index("ubigeo")[["provincia", "distrito"]]
    panel = panel.merge(nombres, left_on="ubigeo", right_index=True, how="left")
    return panel.drop(columns="periodo").sort_values(["ubigeo", "fecha_mes"]).reset_index(drop=True)


def estaticas(ubigeos: list[str]) -> pd.DataFrame:
    """Covariables estáticas disponibles en el repositorio, estandarizadas.

    altitud/latitud/longitud: capital distrital (distritos_cajamarca.csv).
    pct_urbano: % de centros poblados INEI marcados como urbanos.
    Paper: «All static covariates were standardized using their global mean and
    standard deviation». Son constantes en el tiempo, no filtran información futura.
    """
    dist = pd.read_csv(EXTERNAL / "distritos_cajamarca.csv", dtype=str)
    dist = dist.rename(columns={"inei": "ubigeo", "altitude": "altitud",
                                "latitude": "latitud", "longitude": "longitud"})
    dist = dist[dist.ubigeo.isin(ubigeos)][["ubigeo", "altitud", "latitud", "longitud"]]
    ccpp = pd.read_csv(EXTERNAL / "centros_poblados_cajamarca.csv", dtype=str)
    urb = (ccpp.assign(u=ccpp.tipo.eq("Urban")).groupby("inei_distrito").u.mean() * 100)
    dist["pct_urbano"] = dist.ubigeo.map(urb)
    for c in ESTATICAS:
        v = dist[c].astype(float)
        dist[c] = (v - v.mean()) / v.std()
    assert len(dist) == len(ubigeos), "Falta algún distrito en las fuentes estáticas"
    return dist.reset_index(drop=True)


def clima_distrito(protocolo: str = "historico", variante: str = "sin_imputar") -> pd.DataFrame:
    """Clima mensual por distrito preparado por el pipeline base del equipo
    (estación asignada, control de calidad y relleno de faltantes).

    variante: sin_imputar (faltantes vacíos) | estacional_52 (52 % de los faltantes
    rellenados con el mismo mes de años anteriores) | huecos_cortos.
    """
    d = pd.read_csv(PROCESSED / f"experimento_clima/{protocolo}/{variante}.csv",
                    dtype={"ubigeo": str}, parse_dates=["fecha_mes"],
                    usecols=["ubigeo", "fecha_mes"] + CLIMA)
    return d


def cargar_oni(path: Path = ONI_PATH) -> pd.DataFrame | None:
    """Índice Oceánico El Niño (NOAA CPC, oni.ascii.txt: SEAS YR TOTAL ANOM).

    La estación trimestral se asigna a su mes central (DJF -> enero). Devuelve
    None si el archivo no está: el resto del flujo funciona sin ONI.
    """
    path = Path(path)
    if not path.exists():
        return None
    t = pd.read_csv(path, sep=r"\s+")
    centro = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
              "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    t["fecha_mes"] = pd.to_datetime(dict(year=t.YR, month=t.SEAS.map(centro), day=1))
    return t[["fecha_mes"]].assign(oni=t.ANOM.astype(float))


# ---------------------------------------------------------------------------
# 2. Variables y objetivos
# ---------------------------------------------------------------------------
def construir_variables(panel: pd.DataFrame, h: int = 1, modo: str = "operativo",
                        clima: pd.DataFrame | None = None, oni: pd.DataFrame | None = None,
                        est: pd.DataFrame | None = None, interacciones: bool = True,
                        lags_casos: int = 12, lags_clima: int = 7):
    """Devuelve (dataframe, lista de variables) para predecir el mes t a horizonte h.

    operativo: casos t-h..t-h-11, media36 hasta t-h, clima t-h..t-h-6.
    paper:     casos t-1..t-12, media36 hasta t-1, clima t..t-6 (indexación
               retrospectiva del paper; usa el clima del propio mes objetivo).
    Objetivos: y_casos (casos del mes), y_suav (media móvil 3 meses t-2..t, como
    el suavizado del paper) y ma36_obj (media t-35..t, define el riesgo).
    """
    if modo not in ("operativo", "paper"):
        raise ValueError(modo)
    oc = h if modo == "operativo" else 1     # primer rezago de casos
    ok = h if modo == "operativo" else 0     # primer rezago de clima
    d = panel.sort_values(["ubigeo", "fecha_mes"]).reset_index(drop=True)
    exog = []
    if clima is not None:
        d = d.merge(clima, on=["ubigeo", "fecha_mes"], how="left", validate="one_to_one")
        exog += CLIMA
    if oni is not None:
        d = d.merge(oni, on="fecha_mes", how="left")
        exog.append("oni")
    g = d.groupby("ubigeo")
    cols = {}
    for k in range(oc, oc + lags_casos):
        cols[f"casos_lag{k}"] = g.casos.shift(k)
    cols[f"media36_lag{oc}"] = g.casos.transform(lambda s: s.shift(oc).rolling(36, min_periods=36).mean())
    mes = d.fecha_mes.dt.month
    cols["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    cols["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    for v in exog:
        for k in range(ok, ok + lags_clima):
            cols[f"{v}_lag{k}"] = g[v].shift(k)
    if est is not None:
        d = d.merge(est, on="ubigeo", how="left", validate="many_to_one")
        for c in ESTATICAS:
            cols[f"est_{c}"] = d[c]
        if interacciones:
            for v in exog:
                for c in INTERACCION_CON:
                    cols[f"{v}_lag{ok}_x_{c}"] = cols[f"{v}_lag{ok}"] * d[c]
    X = pd.DataFrame(cols, index=d.index)
    base = d[["ubigeo", "provincia", "distrito", "fecha_mes", "casos"]].copy()
    base["y_casos"] = base.casos.astype(float)
    gb = base.groupby("ubigeo").casos
    base["y_suav"] = gb.transform(lambda s: s.rolling(3, min_periods=3).mean())
    base["ma36_obj"] = gb.transform(lambda s: s.rolling(36, min_periods=36).mean())
    return pd.concat([base, X], axis=1), list(X.columns)


def variables_equipo(panel: pd.DataFrame):
    """Las 7 variables de los modelos del equipo: casos de hace 1, 2, 3 y 12 meses,
    promedio de los 3 meses anteriores y mes del año (seno y coseno)."""
    d = panel.sort_values(["ubigeo", "fecha_mes"]).reset_index(drop=True).copy()
    g = d.groupby("ubigeo").casos
    for k in (1, 2, 3, 12):
        d[f"lag{k}"] = g.shift(k)
    d["media3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    mes = d.fecha_mes.dt.month
    d["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    d["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    d["y_casos"] = d.casos.astype(float)
    return d, list(pp.FEATURES)


def etiqueta_riesgo(d: pd.DataFrame, umbral: float) -> pd.Series:
    """1 si la media de 36 meses terminada en t alcanza el umbral; NaN sin historia."""
    return (d.ma36_obj >= umbral).astype(float).where(d.ma36_obj.notna())


def distritos_que_alternan(d: pd.DataFrame, umbral: float) -> list[str]:
    """Distritos con ambos niveles de riesgo en su historia observable.

    Igual que el paper (22 municipios), se decide con todo el periodo: es una
    selección retrospectiva que puede sobreestimar el desempeño real.
    """
    y = etiqueta_riesgo(d, umbral)
    n = y.groupby(d.ubigeo).nunique()
    return sorted(n[n == 2].index)


# ---------------------------------------------------------------------------
# 3. Ventana deslizante
# ---------------------------------------------------------------------------
def ventanas(inicio, fin, h: int, modo: str = "operativo", meses_train: int | None = MESES_TRAIN,
             paso: int | None = None) -> list[dict]:
    """Bloques de prueba consecutivos y su tramo de entrenamiento.

    operativo: se emite el pronóstico en L = último mes del bloque - h. Todo lo
    que usa el modelo (etiquetas de entrenamiento y variables) es <= L, y el bloque
    tiene a lo más h meses para que L sea anterior a la prueba (paso = min(3, h)).
    paper: bloque de 3 meses y entrenamiento hasta el mes anterior al bloque.
    meses_train=None usa ventana creciente (toda la historia hasta L).
    """
    inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
    if paso is None:
        paso = min(3, h) if modo == "operativo" else 3
    if modo == "operativo" and paso > h:
        raise ValueError("En modo operativo el bloque no puede ser más largo que h")
    out, s = [], inicio
    while s <= fin:
        e = min(s + pd.DateOffset(months=paso - 1), fin)
        L = e - pd.DateOffset(months=h) if modo == "operativo" else s - pd.DateOffset(months=1)
        ini = None if meses_train is None else L - pd.DateOffset(months=meses_train - 1)
        out.append(dict(train_ini=ini, train_fin=L, test_ini=s, test_fin=e))
        s = e + pd.DateOffset(months=1)
    return out


def ventana_fija(train_fin, test_ini, test_fin) -> list[dict]:
    """Un único entrenamiento, como los modelos del equipo (hasta 2018, prueba 2019-2024)."""
    return [dict(train_ini=None, train_fin=pd.Timestamp(train_fin),
                 test_ini=pd.Timestamp(test_ini), test_fin=pd.Timestamp(test_fin))]


# ---------------------------------------------------------------------------
# 4. XGBoost
# ---------------------------------------------------------------------------
PARAMS_BASE = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=5.0, reg_lambda=1.0)


def modelo_xgb(tarea: str, params: dict | None = None, scale_pos_weight: float = 1.0, seed: int = SEED):
    p = {**PARAMS_BASE, **(params or {})}
    comunes = dict(tree_method="hist", n_jobs=N_JOBS, random_state=seed, **p)
    if tarea == "casos":
        return xgb.XGBRegressor(objective="reg:squarederror", **comunes)
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             scale_pos_weight=scale_pos_weight, **comunes)


def _ajustar_predecir(Xtr, ytr, Xte, tarea, params, n_boot, seed, con_shap):
    """Ajusta un modelo (y opcionalmente n_boot réplicas) y predice Xte."""
    n = len(Xte)
    res = dict(pred=np.full(n, np.nan), lo=np.full(n, np.nan), hi=np.full(n, np.nan), shap=None)
    if len(ytr) == 0:
        return res
    if tarea != "casos" and len(np.unique(ytr)) < 2:
        # Ventana sin las dos clases: el clasificador no puede aprender; se emite
        # la frecuencia observada (0 o 1) y se registra en la columna constante.
        res["pred"][:] = float(np.mean(ytr))
        res["constante"] = True
        return res
    spw = 1.0
    if tarea != "casos":
        pos = float(np.sum(ytr))
        spw = (len(ytr) - pos) / pos
    m = modelo_xgb(tarea, params, spw, seed).fit(Xtr, ytr)
    pred = m.predict(Xte) if tarea == "casos" else m.predict_proba(Xte)[:, 1]
    if con_shap:
        res["shap"] = m.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)
    if n_boot:
        rng = np.random.default_rng(seed)
        B = np.empty((n_boot, n))
        for b in range(n_boot):
            idx = rng.integers(0, len(ytr), len(ytr))
            yb = ytr[idx]
            if tarea != "casos" and len(np.unique(yb)) < 2:
                B[b] = pred
                continue
            mb = modelo_xgb(tarea, params, spw, seed + b + 1).fit(Xtr.iloc[idx], yb)
            B[b] = mb.predict(Xte) if tarea == "casos" else mb.predict_proba(Xte)[:, 1]
        # Paper: estimación puntual = mediana; límites = percentiles 2,5 y 97,5.
        pred = np.median(B, axis=0)
        res["lo"], res["hi"] = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
    if tarea == "casos":
        pred, res["lo"], res["hi"] = (np.maximum(a, 0) for a in (pred, res["lo"], res["hi"]))
    res["pred"] = pred
    return res


def walk_forward(d: pd.DataFrame, feats: list[str], objetivo, tarea: str, wins: list[dict],
                 params: dict | None = None, n_boot: int = 0, por_distrito: bool = False,
                 con_shap: bool = False, seed: int = SEED):
    """Recorre las ventanas: ajusta con el tramo de entrenamiento y predice el bloque.

    objetivo: nombre de columna o Serie alineada con d (p. ej. etiqueta_riesgo).
    Devuelve predicciones (y SHAP del modelo puntual si con_shap=True).
    """
    y = d[objetivo] if isinstance(objetivo, str) else objetivo
    out, shaps = [], []
    for w_id, w in enumerate(wins):
        tr = (d.fecha_mes <= w["train_fin"]) & y.notna()
        if w["train_ini"] is not None:
            tr &= d.fecha_mes >= w["train_ini"]
        te = (d.fecha_mes >= w["test_ini"]) & (d.fecha_mes <= w["test_fin"])
        grupos = [(None, tr, te)] if not por_distrito else [
            (u, tr & (d.ubigeo == u), te & (d.ubigeo == u)) for u in d.loc[te, "ubigeo"].unique()]
        for _, gtr, gte in grupos:
            if not gte.any():
                continue
            r = _ajustar_predecir(d.loc[gtr, feats], y[gtr].to_numpy(), d.loc[gte, feats],
                                  tarea, params, n_boot, seed + 1000 * w_id, con_shap)
            p = d.loc[gte, ["ubigeo", "distrito", "fecha_mes"]].copy()
            p["observado"] = y[gte].to_numpy()
            p["predicho"], p["lim_inf"], p["lim_sup"] = r["pred"], r["lo"], r["hi"]
            p["ventana"], p["n_train"] = w_id, int(gtr.sum())
            p["constante"] = r.get("constante", False)
            out.append(p)
            if con_shap and r["shap"] is not None:
                s = pd.DataFrame(r["shap"][:, :-1], columns=feats, index=p.index)
                shaps.append(pd.concat([p[["ubigeo", "distrito", "fecha_mes"]], s], axis=1))
    pred = pd.concat(out).sort_values(["fecha_mes", "ubigeo"]).reset_index(drop=True)
    if con_shap:
        return pred, (pd.concat(shaps).reset_index(drop=True) if shaps else None)
    return pred


# ---------------------------------------------------------------------------
# 5. Optimización bayesiana (solo validación)
# ---------------------------------------------------------------------------
def optimizar(d, feats, objetivo, tarea, wins_validacion, n_trials: int = 20, seed: int = SEED):
    """TPE de Optuna sobre las ventanas de validación.

    Regresión minimiza RMSE; clasificación maximiza average precision, que es más
    exigente que el AUC cuando los meses de riesgo son pocos (el AUC se reporta en la prueba).
    El paper optimiza por municipio; aquí se optimiza el modelo conjunto.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True))
        p = walk_forward(d, feats, objetivo, tarea, wins_validacion, params=params)
        p = p.dropna(subset=["observado"])
        if tarea == "casos":
            return float(np.sqrt(mean_squared_error(p.observado, p.predicho)))
        if p.observado.nunique() < 2:
            return 0.0
        return float(average_precision_score(p.observado, p.predicho))

    study = optuna.create_study(direction="minimize" if tarea == "casos" else "maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


def optimizar_con_cache(clave: str, *args, recalcular: bool = False, **kwargs):
    """Guarda los hiperparámetros en results/p1_adams/hiperparametros.json para
    no repetir la búsqueda en cada ejecución del notebook."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "hiperparametros.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if clave in cache and not recalcular:
        return cache[clave]["params"], cache[clave]["valor_validacion"]
    params, valor = optimizar(*args, **kwargs)
    cache[clave] = dict(params=params, valor_validacion=valor)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return params, valor


# ---------------------------------------------------------------------------
# 6. Métricas
# ---------------------------------------------------------------------------
def umbral_f1(y, p) -> float:
    """Puntaje a partir del cual se emite alerta: el que maximiza F1 en validación."""
    precision, recall, cortes = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(cortes[np.argmax(f1[:-1])]) if len(cortes) else 0.5


def metricas_regresion(obs, pred, lo=None, hi=None) -> dict:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    m = dict(RMSE=float(np.sqrt(mean_squared_error(obs, pred))),
             MAE=float(mean_absolute_error(obs, pred)), n=len(obs))
    if lo is not None and not np.all(np.isnan(lo)):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        m["cobertura_95"] = float(np.mean((obs >= lo) & (obs <= hi)))
        m["ancho_medio"] = float(np.mean(hi - lo))
    return m


def metricas_clasificacion(obs, score, corte) -> dict:
    obs, score = np.asarray(obs, int), np.asarray(score, float)
    pred = score >= corte
    dos = len(np.unique(obs)) == 2
    return dict(AUC=float(roc_auc_score(obs, score)) if dos else np.nan,
                PR_AUC=float(average_precision_score(obs, score)) if dos else np.nan,
                precision=float(precision_score(obs, pred, zero_division=0)),
                recall=float(recall_score(obs, pred, zero_division=0)),
                F1=float(f1_score(obs, pred, zero_division=0)),
                positivos=int(obs.sum()), alertas=int(pred.sum()), n=len(obs))


def por_distrito(pred: pd.DataFrame, tarea: str, corte: float | None = None) -> pd.DataFrame:
    filas = []
    for (u, nombre), g in pred.dropna(subset=["observado"]).groupby(["ubigeo", "distrito"]):
        if tarea == "casos":
            m = metricas_regresion(g.observado, g.predicho, g.lim_inf, g.lim_sup)
        else:
            m = metricas_clasificacion(g.observado, g.predicho, corte)
        filas.append(dict(ubigeo=u, distrito=nombre, **m))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 7. SHAP
# ---------------------------------------------------------------------------
def grupo_variable(nombre: str) -> str:
    if "_x_" in nombre:
        return "clima × variable fija"
    if nombre.startswith("casos_lag"):
        return "casos de meses anteriores"
    if nombre.startswith("media36"):
        return "promedio de 36 meses"
    if nombre.startswith("mes_"):
        return "mes del año"
    if nombre.startswith("oni"):
        return "ONI (El Niño)"
    if nombre.startswith("est_"):
        return "variable fija del distrito"
    return "clima (SENAMHI)"


def importancia_shap(shap_df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Media del |SHAP| por variable y grupo (como la figura 3 del paper)."""
    imp = shap_df[feats].abs().mean().rename("media_abs_shap").reset_index()
    imp.columns = ["variable", "media_abs_shap"]
    imp["grupo"] = imp.variable.map(grupo_variable)
    return imp.sort_values("media_abs_shap", ascending=False).reset_index(drop=True)
