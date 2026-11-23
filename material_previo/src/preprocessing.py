"""Preprocesamiento de la vigilancia de leishmaniosis cutánea (CDC Perú, 2000-2024).

Decisiones tomadas a partir de notebooks/01_eda.ipynb:
  * Los valores extremos NO se recortan: son los brotes que se quiere predecir.
  * Los faltantes estructurales de los rezagos (primeros meses de cada distrito)
    se eliminan; imputarlos usaría información futura o inventaría historia.
  * Todo lo que se aprende de los datos (umbral de alerta, escaladores, SMOTE)
    se ajusta SOLO con el periodo de entrenamiento.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
UMBRAL_CASOS = 50              # casos acumulados mínimos para conservar un distrito
TRAIN_END = "2018-12-31"       # entrenamiento <= 2018, prueba 2019-2024
LAGS = (1, 2, 3, 12)
FEATURES = ["lag1", "lag2", "lag3", "lag12", "media3", "mes_sin", "mes_cos"]
CONTEOS = ["lag1", "lag2", "lag3", "lag12", "media3"]   # variables de conteo
EDAD_MAX = 110                 # años; mayor valor se considera error de captura
_FACTOR_EDAD = {"A": 1.0, "M": 1 / 12, "D": 1 / 365}   # unidad -> años


# --------------------------------------------------------------------------
# 1. Carga y limpieza a nivel de caso
# --------------------------------------------------------------------------
def load_raw(path: str | Path) -> pd.DataFrame:
    """Lee el CSV original sin modificar nada."""
    return pd.read_csv(path, low_memory=False)


def clean_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de los registros individuales.

    - ubigeo a 6 dígitos (se pierde el cero inicial en los departamentos 01-09).
    - edad a años; valores > 110 años -> nulo -> mediana (SimpleImputer).
    - diresa: moda dentro del mismo ubigeo.
    - localidad: etiqueta 'SIN DATO'.
    - se eliminan localcod (32 % nulos) y las columnas constantes.
    - se agrega una fecha aproximada (1 de enero + 7·(semana-1)).
    No se eliminan duplicados: pueden ser casos distintos con los mismos atributos.
    """
    out = df.copy()

    out["ubigeo"] = out["ubigeo"].astype(str).str.zfill(6)

    out["edad_anios"] = out["edad"] * out["tipo_edad"].map(_FACTOR_EDAD)
    out.loc[out["edad_anios"] > EDAD_MAX, "edad_anios"] = np.nan
    out[["edad_anios"]] = SimpleImputer(strategy="median").fit_transform(out[["edad_anios"]])

    moda = out.groupby("ubigeo")["diresa"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    out["diresa"] = out["diresa"].fillna(out["ubigeo"].map(moda))

    out["localidad"] = out["localidad"].fillna("SIN DATO")
    out = out.drop(columns=[c for c in ("localcod", "enfermedad", "diagnostic") if c in out.columns])

    out["fecha"] = pd.to_datetime(out["ano"].astype(str) + "-01-01") + pd.to_timedelta((out["semana"] - 1) * 7, unit="D")
    out["periodo"] = out["fecha"].dt.to_period("M")
    out["mes"] = out["fecha"].dt.month
    return out


# --------------------------------------------------------------------------
# 2. Tabla de modelado distrito x mes
# --------------------------------------------------------------------------
def build_monthly_panel(
    cases: pd.DataFrame,
    min_cases: int = UMBRAL_CASOS,
    train_end: str = TRAIN_END,
    alert_k: float = 2.0,
    exclude_week1: bool = False,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Agrega los casos a distrito x mes y construye rezagos, estacionalidad y objetivos.

    Parámetros
    ----------
    min_cases : distritos con menos casos acumulados se descartan.
    alert_k : la alerta es casos > media + k·desv. del distrito (solo entrenamiento).
    exclude_week1 : análisis de sensibilidad; descarta los casos de la semana 1,
        que es una anomalía de registro (2,6 veces las semanas vecinas).
    drop_incomplete : elimina las filas sin historia completa (rezagos nulos).

    Columnas: ubigeo, departamento, provincia, distrito, fecha_mes, es_train,
    casos, umbral, alerta + FEATURES.
    """
    # El rango de meses se fija con TODOS los casos, para que el calendario no cambie
    # al excluir la semana 1 ni al filtrar distritos.
    meses = pd.period_range(cases["periodo"].min(), cases["periodo"].max(), freq="M")

    base = cases[cases["semana"] != 1] if exclude_week1 else cases
    tot = cases.groupby("ubigeo").size()
    activos = tot[tot >= min_cases].index
    nombres = (cases.drop_duplicates("ubigeo").set_index("ubigeo")[["departamento", "provincia", "distrito"]])

    panel = (
        base[base["ubigeo"].isin(activos)]
        .groupby(["ubigeo", "periodo"]).size()
        .unstack("ubigeo")
        .reindex(index=meses, columns=sorted(activos), fill_value=0)
        .fillna(0).astype(int)
        .stack().rename("casos").reset_index()
    )
    panel.columns = ["periodo", "ubigeo", "casos"]
    panel["fecha_mes"] = panel["periodo"].dt.to_timestamp()
    panel = panel.merge(nombres.reset_index(), on="ubigeo", how="left")
    panel = panel.sort_values(["ubigeo", "fecha_mes"]).reset_index(drop=True)

    # Rezagos dentro de cada distrito: nunca se mezcla información entre distritos.
    g = panel.groupby("ubigeo")["casos"]
    for k in LAGS:
        panel[f"lag{k}"] = g.shift(k)
    panel["media3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())

    mes = panel["fecha_mes"].dt.month
    panel["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    panel["mes_cos"] = np.cos(2 * np.pi * mes / 12)

    panel["es_train"] = panel["fecha_mes"] <= pd.Timestamp(train_end)

    # Alerta: umbral por distrito calculado solo con entrenamiento.
    # Se calcula ANTES de descartar filas incompletas, igual que en el EDA.
    panel = add_alert(panel, alert_k)

    if drop_incomplete:
        panel = panel.dropna(subset=FEATURES).reset_index(drop=True)

    cols = ["ubigeo", "departamento", "provincia", "distrito", "fecha_mes", "es_train",
            "casos", "umbral", "alerta"] + FEATURES
    return panel[cols]


def add_alert(panel: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """Agrega `umbral` y `alerta` (definición provisional, no es la norma del MINSA).

    umbral = media + k·desv. de los casos del distrito en entrenamiento.
    """
    out = panel.copy()
    est = out.loc[out["es_train"]].groupby("ubigeo")["casos"].agg(["mean", "std"])
    out["umbral"] = out["ubigeo"].map(est["mean"] + k * est["std"])
    out["alerta"] = (out["casos"] > out["umbral"]).astype(int)
    return out


def split_train_test(panel: pd.DataFrame):
    """Partición temporal (nunca aleatoria). Devuelve (train, test)."""
    return panel[panel["es_train"]].copy(), panel[~panel["es_train"]].copy()


# --------------------------------------------------------------------------
# 3. Escalamiento (ajustado solo con entrenamiento)
# --------------------------------------------------------------------------
def make_scaler(kind: str = "log1p_standard"):
    """Devuelve un escalador SIN ajustar.

    kind:
      'none'            -> None (árboles de decisión no lo necesitan)
      'standard'        -> StandardScaler
      'minmax'          -> MinMaxScaler
      'log1p_standard'  -> log1p sobre los conteos y luego StandardScaler
                           (recomendado para KNN, regresión logística y MLP;
                           conserva el orden y deja los brotes identificables)
    Usar siempre scaler.fit(X_train) y luego scaler.transform(X_train / X_test).
    """
    if kind == "none":
        return None
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "log1p_standard":
        return make_pipeline(FunctionTransformer(_log1p_conteos, validate=False), StandardScaler())
    raise ValueError(f"kind desconocido: {kind}")


def _log1p_conteos(X):
    """log1p solo en las columnas de conteo; deja mes_sin y mes_cos intactos."""
    X = pd.DataFrame(X, columns=FEATURES).copy() if not isinstance(X, pd.DataFrame) else X.copy()
    X[CONTEOS] = np.log1p(X[CONTEOS])
    return X


def scale_features(train: pd.DataFrame, test: pd.DataFrame, kind: str = "log1p_standard"):
    """Ajusta el escalador con `train` y transforma ambos. Devuelve (X_train, X_test, scaler)."""
    scaler = make_scaler(kind)
    if scaler is None:
        return train[FEATURES].copy(), test[FEATURES].copy(), None
    Xtr = pd.DataFrame(scaler.fit_transform(train[FEATURES]), columns=FEATURES, index=train.index)
    Xte = pd.DataFrame(scaler.transform(test[FEATURES]), columns=FEATURES, index=test.index)
    return Xtr, Xte, scaler


# --------------------------------------------------------------------------
# 4. Desbalance de clases (solo entrenamiento)
# --------------------------------------------------------------------------
def class_weights(y) -> dict:
    """Pesos 'balanced' para usar como class_weight en el modelo (opción base)."""
    y = np.asarray(y)
    clases = np.unique(y)
    pesos = compute_class_weight("balanced", classes=clases, y=y)
    return {int(c): float(w) for c, w in zip(clases, pesos)}


def apply_smote(X_train, y_train, random_state: int = 42):
    """SMOTE sobre entrenamiento (opción de comparación).

    Salvedades: crea conteos no enteros y mezcla meses y distritos distintos,
    así que no respeta el orden temporal. Aplicar nunca sobre el conjunto de prueba
    y, en validación deslizante, dentro de cada pliegue.
    """
    from imblearn.over_sampling import SMOTE
    return SMOTE(random_state=random_state).fit_resample(X_train, y_train)


# --------------------------------------------------------------------------
# 5. Pipeline completo y ejecución desde consola
# --------------------------------------------------------------------------
def run(raw_csv: str | Path, out_dir: str | Path, **panel_kwargs) -> dict:
    """Lee el CSV crudo y guarda casos_limpios.csv y panel_mensual_distrital.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = clean_cases(load_raw(raw_csv))
    panel = build_monthly_panel(cases, **panel_kwargs)

    cases.drop(columns=["periodo"]).to_csv(out_dir / "casos_limpios.csv", index=False)
    panel.to_csv(out_dir / "panel_mensual_distrital.csv", index=False)
    return {"casos": cases, "panel": panel}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw/datos_abiertos_vigilancia_leishmaniosis_2000_2024.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--exclude-week1", action="store_true", help="sensibilidad sin la semana 1")
    a = ap.parse_args()

    res = run(a.raw, a.out, exclude_week1=a.exclude_week1)
    p = res["panel"]
    tr, te = split_train_test(p)
    print(f"casos limpios: {len(res['casos']):,}  |  panel: {len(p):,} filas, {p['ubigeo'].nunique()} distritos")
    print(f"train {len(tr):,} (alerta {tr['alerta'].mean():.2%})  |  test {len(te):,} (alerta {te['alerta'].mean():.2%})")
