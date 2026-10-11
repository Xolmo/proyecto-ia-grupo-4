"""Integración del clima SENAMHI con la vigilancia de leishmaniosis cutánea.

Genera, en data/processed/:
  1. clima_diario.csv                  todas las estaciones (Cusco y Cajamarca), una fila por día.
  2. clima_mensual_cajamarca.csv       promedios mensuales por estación, solo Cajamarca.
     match_localidades_cajamarca.csv   coordenadas de cada localidad con casos y su estación más cercana.
  3. dataset_cajamarca.csv             panel distrito x mes de preprocessing.py + clima.

Formato de los .txt (RClimDex de SENAMHI): año mes día precipitación tmax tmin; -99.9 = faltante.

Coordenadas (data/external/, ver FUENTES.md):
  estaciones   mapa de estaciones de SENAMHI; Querocotillo y Acomayo del Atlas Climático (0,1°).
  localidades  centro poblado del INEI con el mismo nombre dentro del mismo distrito; si no se
               encuentra, la capital del distrito.

Match localidad -> estación: la más cercana en línea recta (haversine). Se buscan dos:
  _pp  la más cercana entre todas (todas miden lluvia).
  _t   la más cercana entre las que tienen temperatura.
Cada distrito del panel toma la estación que concentra más casos de sus localidades
(contados solo en entrenamiento, para no usar información del periodo de prueba).
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import TRAIN_END, build_monthly_panel, clean_cases, load_raw

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
REGION = "CAJAMARCA"
EXTERNAL = Path("data/external")
MIN_DIAS_VALIDOS = 0.8         # fracción mínima de días con dato para aceptar un mes
MIN_MESES_TEMP = 0.5           # una estación "tiene temperatura" si cubre >= 50 % de los meses
CORTE_NOMBRE = 0.85            # similitud mínima para aceptar un centro poblado parecido
LAGS_CLIMA = (1, 2, 3)
VARS_PP = ["pp_total_mm", "pp_media_mm"]
VARS_T = ["tmax_media", "tmin_media", "tmed_media"]
VARS_CLIMA = VARS_PP + VARS_T
VARS_LAG = ["pp_total_mm", "tmax_media", "tmin_media", "tmed_media"]
_PREFIJOS = r"^(C\.?\s?P\.?|CASERIO|CAS\.|ANEXO|SECTOR|COMUNIDAD|CC\.?\s?NN\.?|CENTRO POBLADO)\s+"


def _norm(s) -> str:
    """Mayúsculas sin tildes. El CSV de casos trae la Ñ dañada (U+FFFD); se lee como N."""
    s = str(s).replace("�", "N")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().upper()


def _nombre_localidad(s) -> str:
    """Normaliza el nombre de una localidad para compararlo con los centros poblados."""
    s = re.split(r"\s{2,}", str(s).strip())[0]      # "EL LIMON   LIVES" -> "EL LIMON"
    s = _norm(s).replace("NVO.", "NUEVO ").replace("STA.", "SANTA ").replace("STO.", "SANTO ")
    s = re.sub(_PREFIJOS, "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", s)).strip()


def _haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en km entre cada punto 1 (filas) y cada punto 2 (columnas)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------
# 1. Clima diario de todas las regiones
# --------------------------------------------------------------------------
def leer_estacion(path: str | Path) -> pd.DataFrame:
    """Lee un .txt de SENAMHI y deja como nulo todo dato inválido."""
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=range(6),
                     names=["anio", "mes", "dia", "pp", "tmax", "tmin"])
    df["fecha"] = pd.to_datetime(
        df[["anio", "mes", "dia"]].rename(columns={"anio": "year", "mes": "month", "dia": "day"}),
        errors="coerce")
    df = df.dropna(subset=["fecha"]).drop_duplicates("fecha").drop(columns=["anio", "mes", "dia"])

    df[["pp", "tmax", "tmin"]] = df[["pp", "tmax", "tmin"]].where(df[["pp", "tmax", "tmin"]] > -99)
    df.loc[df["pp"] < 0, "pp"] = np.nan
    df.loc[df["tmax"] < df["tmin"], ["tmax", "tmin"]] = np.nan   # registro inconsistente
    return df


def clima_diario(clima_dir: str | Path, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Une todas las estaciones de data/clima/<region>/*.txt en una tabla larga."""
    meta = estaciones.set_index("codigo")
    tablas = []
    for path in sorted(Path(clima_dir).glob("*/*.txt")):
        codigo = re.search(r"qc\d+", path.name).group(0)
        df = leer_estacion(path)
        df.insert(0, "region", path.parent.name.upper())
        df.insert(1, "codigo", codigo)
        df.insert(2, "estacion", path.stem.split(" - ", 1)[-1].strip().upper())
        df.insert(3, "latitud", meta["latitud"].get(codigo, np.nan))
        df.insert(4, "longitud", meta["longitud"].get(codigo, np.nan))
        tablas.append(df)
    return pd.concat(tablas, ignore_index=True)


# --------------------------------------------------------------------------
# 2. Clima mensual por estación y match con las localidades (solo Cajamarca)
# --------------------------------------------------------------------------
def clima_mensual(diario: pd.DataFrame, region: str = REGION) -> pd.DataFrame:
    """Promedios mensuales por estación.

    Un mes se acepta solo si tiene al menos 80 % de días con dato en esa variable.
    La precipitación se da como promedio diario y como total del mes (promedio x días).
    """
    d = diario[diario["region"] == region].copy()
    d["fecha_mes"] = d["fecha"].dt.to_period("M").dt.to_timestamp()
    m = (d.groupby(["codigo", "fecha_mes"])
         .agg(pp_media_mm=("pp", "mean"), tmax_media=("tmax", "mean"), tmin_media=("tmin", "mean"),
              dias_pp=("pp", "count"), dias_tmax=("tmax", "count"), dias_tmin=("tmin", "count"))
         .reset_index())

    dias_mes = m["fecha_mes"].dt.days_in_month
    for var, dias in [("pp_media_mm", "dias_pp"), ("tmax_media", "dias_tmax"), ("tmin_media", "dias_tmin")]:
        m.loc[m[dias] < MIN_DIAS_VALIDOS * dias_mes, var] = np.nan
    m["pp_total_mm"] = m["pp_media_mm"] * dias_mes
    m["tmed_media"] = (m["tmax_media"] + m["tmin_media"]) / 2
    m[VARS_CLIMA] = m[VARS_CLIMA].round(2)

    meta = d.drop_duplicates("codigo")[["codigo", "estacion", "latitud", "longitud"]]
    cols = ["codigo", "estacion", "latitud", "longitud", "fecha_mes",
            *VARS_CLIMA, "dias_pp", "dias_tmax", "dias_tmin"]
    return m.merge(meta, on="codigo")[cols]


def resumen_estaciones(mensual: pd.DataFrame) -> pd.DataFrame:
    """Coordenadas de cada estación y si tiene temperatura en el periodo de los casos."""
    periodo = mensual[mensual["fecha_mes"] >= "2000-01-01"]
    cobertura = periodo.groupby("codigo")["tmed_media"].apply(lambda s: s.notna().mean())
    est = mensual.drop_duplicates("codigo")[["codigo", "estacion", "latitud", "longitud"]].set_index("codigo")
    est["tiene_temp"] = cobertura.reindex(est.index).fillna(0) >= MIN_MESES_TEMP
    return est.reset_index()


def coordenadas_localidades(casos: pd.DataFrame, ccpp: pd.DataFrame, distritos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por localidad con casos y sus coordenadas.

    fuente_coord: ccpp_exacto    centro poblado con el mismo nombre en el mismo distrito
                  ccpp_parecido  nombre con similitud >= 0.85 (errores de tipeo)
                  capital        no se encontró: capital del distrito
    """
    casos = casos.assign(train=casos["fecha"] <= pd.Timestamp(TRAIN_END))
    loc = (casos.groupby(["ubigeo", "provincia", "distrito", "localidad"])
           .agg(casos=("train", "size"), casos_train=("train", "sum")).reset_index())

    ccpp = ccpp.assign(nombre=ccpp["ccpp"].map(_nombre_localidad),
                       latitude=ccpp["latitude"].astype(float), longitude=ccpp["longitude"].astype(float))
    por_distrito = {u: g.groupby("nombre")[["latitude", "longitude"]].mean() for u, g in ccpp.groupby("inei_distrito")}
    capital = distritos.set_index("inei")[["latitude", "longitude"]].astype(float)

    filas = []
    for ubigeo, localidad in loc[["ubigeo", "localidad"]].itertuples(index=False):
        nombre = _nombre_localidad(localidad)
        tabla = por_distrito.get(ubigeo)
        fuente, match = "capital", None
        if tabla is not None and nombre and nombre != "SIN DATO":
            if nombre in tabla.index:
                fuente, match = "ccpp_exacto", nombre
            else:
                parecido = difflib.get_close_matches(nombre, tabla.index, n=1, cutoff=CORTE_NOMBRE)
                if parecido:
                    fuente, match = "ccpp_parecido", parecido[0]
        lat, lon = tabla.loc[match] if match else capital.loc[ubigeo] if ubigeo in capital.index else (np.nan, np.nan)
        filas.append({"ccpp_match": match, "fuente_coord": fuente, "latitud": lat, "longitud": lon})
    return pd.concat([loc, pd.DataFrame(filas, index=loc.index)], axis=1)


def estacion_mas_cercana(loc: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Agrega la estación más cercana para lluvia (_pp) y para temperatura (_t)."""
    out = loc.copy()
    for sufijo, est in [("pp", estaciones), ("t", estaciones[estaciones["tiene_temp"]])]:
        dist = _haversine_km(out["latitud"].to_numpy(), out["longitud"].to_numpy(),
                             est["latitud"].to_numpy(), est["longitud"].to_numpy())
        i = np.nanargmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
        out[f"codigo_{sufijo}"] = est["codigo"].to_numpy()[i]
        out[f"estacion_{sufijo}"] = est["estacion"].to_numpy()[i]
        out[f"dist_{sufijo}_km"] = dist[np.arange(len(out)), i].round(1)
    return out


def asignar_distritos(localidades: pd.DataFrame) -> pd.DataFrame:
    """Estación de cada distrito: la que concentra más casos de entrenamiento de sus localidades."""
    filas = []
    for ubigeo, g in localidades.groupby("ubigeo"):
        peso = g["casos_train"] if g["casos_train"].sum() > 0 else g["casos"]
        fila = {"ubigeo": ubigeo}
        for sufijo in ("pp", "t"):
            por_est = peso.groupby(g[f"codigo_{sufijo}"]).sum()
            codigo = por_est.idxmax()
            elegidas = g[f"codigo_{sufijo}"] == codigo
            fila[f"codigo_{sufijo}"] = codigo
            fila[f"estacion_{sufijo}"] = g.loc[elegidas, f"estacion_{sufijo}"].iloc[0]
            fila[f"pct_casos_{sufijo}"] = round(100 * por_est.max() / peso.sum(), 1)
            fila[f"dist_{sufijo}_km"] = round(np.average(g.loc[elegidas, f"dist_{sufijo}_km"],
                                                         weights=peso[elegidas] + 1e-9), 1)
        filas.append(fila)
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 3. Dataset final: panel distrito x mes + clima
# --------------------------------------------------------------------------
def clima_por_distrito(mensual: pd.DataFrame, asignacion: pd.DataFrame) -> pd.DataFrame:
    """Clima mensual de cada distrito (lluvia y temperatura de su estación) y sus rezagos."""
    pp = asignacion[["ubigeo", "codigo_pp"]].merge(
        mensual[["codigo", "fecha_mes", *VARS_PP]], left_on="codigo_pp", right_on="codigo")
    t = asignacion[["ubigeo", "codigo_t"]].merge(
        mensual[["codigo", "fecha_mes", *VARS_T]], left_on="codigo_t", right_on="codigo")
    clima = (pp.set_index(["ubigeo", "fecha_mes"])[VARS_PP]
             .join(t.set_index(["ubigeo", "fecha_mes"])[VARS_T], how="outer"))

    # Calendario completo para que shift(k) sea exactamente k meses atrás.
    meses = pd.date_range(mensual["fecha_mes"].min(), mensual["fecha_mes"].max(), freq="MS")
    idx = pd.MultiIndex.from_product([asignacion["ubigeo"].unique(), meses], names=["ubigeo", "fecha_mes"])
    clima = clima.reindex(idx).reset_index().sort_values(["ubigeo", "fecha_mes"])

    g = clima.groupby("ubigeo")
    for var in VARS_LAG:
        for k in LAGS_CLIMA:
            clima[f"{var}_lag{k}"] = g[var].shift(k)
    return clima


def dataset_final(casos: pd.DataFrame, mensual: pd.DataFrame, asignacion: pd.DataFrame, **panel_kwargs) -> pd.DataFrame:
    """Panel de preprocessing.py (distritos con >= 50 casos) unido al clima de su estación."""
    panel = build_monthly_panel(casos, **panel_kwargs)
    clima = clima_por_distrito(mensual, asignacion)
    out = (panel.merge(asignacion, on="ubigeo", how="left")
           .merge(clima, on=["ubigeo", "fecha_mes"], how="left"))
    out["tiene_clima"] = out["pp_total_mm"].notna()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    return out


# --------------------------------------------------------------------------
# Ejecución
# --------------------------------------------------------------------------
def run(raw_csv: str | Path, clima_dir: str | Path, out_dir: str | Path, external: str | Path = EXTERNAL) -> dict:
    out_dir, external = Path(out_dir), Path(external)
    out_dir.mkdir(parents=True, exist_ok=True)

    estaciones = pd.read_csv(external / "estaciones_senamhi.csv")
    ccpp = pd.read_csv(external / "centros_poblados_cajamarca.csv", dtype=str)
    distritos = pd.read_csv(external / "distritos_cajamarca.csv", dtype=str)

    diario = clima_diario(clima_dir, estaciones)
    casos = clean_cases(load_raw(raw_csv))
    casos = casos[casos["departamento"] == REGION]

    mensual = clima_mensual(diario)
    localidades = estacion_mas_cercana(coordenadas_localidades(casos, ccpp, distritos),
                                       resumen_estaciones(mensual))
    asignacion = asignar_distritos(localidades)
    final = dataset_final(casos, mensual, asignacion)

    diario.to_csv(out_dir / "clima_diario.csv", index=False)
    mensual.to_csv(out_dir / "clima_mensual_cajamarca.csv", index=False)
    localidades.to_csv(out_dir / "match_localidades_cajamarca.csv", index=False)
    final.to_csv(out_dir / "dataset_cajamarca.csv", index=False)
    return {"diario": diario, "mensual": mensual, "localidades": localidades, "final": final}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw/datos_abiertos_vigilancia_leishmaniosis_2000_2024.csv")
    ap.add_argument("--clima", default="data/clima")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()

    r = run(a.raw, a.clima, a.out)
    d, m, loc, f = r["diario"], r["mensual"], r["localidades"], r["final"]
    print(f"clima diario: {len(d):,} filas, {d['codigo'].nunique()} estaciones, regiones {sorted(d['region'].unique())}")
    print(f"clima mensual {REGION}: {len(m):,} filas, {m['codigo'].nunique()} estaciones")
    print("localidades por fuente de coordenadas:")
    print(loc.groupby("fuente_coord")["casos"].agg(["count", "sum"]).rename(columns={"count": "localidades"}))
    print("distancia a la estación más cercana (km, ponderada por casos): "
          f"lluvia {np.average(loc['dist_pp_km'], weights=loc['casos']):.1f}, "
          f"temperatura {np.average(loc['dist_t_km'], weights=loc['casos']):.1f}")
    print(f"dataset final: {len(f):,} filas, {f['ubigeo'].nunique()} distritos, "
          f"{f['tiene_clima'].mean():.1%} de filas con clima")
