"""Réplica del modelo del paper 2 (Donizette, Rocco y de Queiroz, 2025).

El paper compara redes feedforward y un SVR para predecir los casos mensuales
de leishmaniasis. Aquí se implementan sus redes, en PyTorch (sin agregar
TensorFlow al proyecto):

- ``SFNN1`` / ``SFNN2``: 1 capa oculta densa, Dropout 0,2 y salida lineal de
  1 neurona. Se diferencian solo en la cuadrícula de neuronas (4-8 vs 12-28).
- ``DFNN``: 2 capas ocultas densas, cada una seguida de Dropout 0,2.

Se conserva del paper: MinMaxScaler (0-1), pérdida MSE, cuadrícula de
neuronas, activación (ReLU/Tanh), optimizador (Adam/RMSprop) y épocas (100/300),
y RMSE como criterio de elección.

Diferencias con el paper (por la naturaleza de este proyecto):

- La elección de hiperparámetros usa la validación temporal 2016-2018 del
  proyecto, no ``GridSearchCV`` de 5 folds, que mezclaría futuro y pasado.
- Cuadrícula de neuronas reducida (SFNN2: 12/20/28, DFNN: 4/12/28) y lotes de
  256 para que corra en minutos y no en horas.
- Variables (SENAMHI no tiene insolación ni humedad; no se prueban las 127
  combinaciones del paper, solo estos cuatro conjuntos):
  ``casos3`` = ``lag1``-``lag3`` (como el paper), ``casos7`` = las 7 variables de
  casos del equipo (mismas entradas que los demás modelos, comparación justa) y
  sus versiones ``_clima`` con los 9 rezagos de lluvia y temperaturas.
- Predice el mes siguiente con clima rezagado, no el clima del mismo mes.

Los hiperparámetros se eligen con la semilla 42; con la configuración elegida
se reentrena con 5 semillas (``SEEDS``) para medir cuánto varía el RMSE de prueba.

El paper solo hace regresión; la alerta se evalúa usando los casos predichos
como puntaje (umbral elegido en validación), para compararla con los demás.

Uso (desde la raíz del repositorio)::

    python src/modelo_paper2.py                        # protocolo oficial -> results/paper2/
    python src/modelo_paper2.py --protocolo historico  # clima observado -> results/paper2_historico/
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

import preprocessing as pp
from evaluation import choose_threshold, labels, metrics
from experimento_clima import CF, DATA, PROTOCOLS
from models import SEED

ROOT = Path(__file__).resolve().parents[1]
# oficial: ajuste 2001-2015, validación 2016-2018, prueba 2019-2024 (clima imputado).
# historico: ajuste hasta 2009, validación 2010-2011, prueba 2012-2014 (clima observado).
CASOS3 = ["lag1", "lag2", "lag3"]
FEATURE_SETS = {"casos3": CASOS3, "casos3_clima": CASOS3 + CF,
                "casos7": list(pp.FEATURES), "casos7_clima": list(pp.FEATURES) + CF}
SEEDS = (42, 43, 44, 45, 46)
DROPOUT = 0.2
BATCH = 256

# Neuronas por modelo (recortadas del paper); el resto de la cuadrícula es la misma.
NEURONS = {"SFNN1": (4, 8), "SFNN2": (12, 20, 28), "DFNN": (4, 12, 28)}
ACTIVATIONS = ("relu", "tanh")
OPTIMIZERS = ("adam", "rmsprop")
EPOCHS = (100, 300)


class FeedForward:
    """Red densa del paper: ``n_hidden`` capas (1 = SFNN, 2 = DFNN), Dropout 0,2, salida lineal."""

    def __init__(self, n_hidden: int, neurons: int, activation: str, optimizer: str, epochs: int,
                 seed: int = SEED):
        self.n_hidden, self.neurons, self.seed = n_hidden, neurons, seed
        self.activation, self.optimizer, self.epochs = activation, optimizer, epochs

    def _build(self, n_in: int):
        act = torch.nn.ReLU if self.activation == "relu" else torch.nn.Tanh
        layers, size = [], n_in
        for _ in range(self.n_hidden):
            layers += [torch.nn.Linear(size, self.neurons), act(), torch.nn.Dropout(DROPOUT)]
            size = self.neurons
        layers.append(torch.nn.Linear(size, 1))       # salida lineal (regresión)
        return torch.nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        # MinMax a [0, 1] en entradas y objetivo, ajustado solo con el entrenamiento.
        self.sx, self.sy = MinMaxScaler().fit(X), MinMaxScaler().fit(np.asarray(y, float).reshape(-1, 1))
        tx = torch.tensor(self.sx.transform(X), dtype=torch.float32)
        ty = torch.tensor(self.sy.transform(np.asarray(y, float).reshape(-1, 1)), dtype=torch.float32)
        self.net = self._build(tx.shape[1])
        opt_cls = torch.optim.Adam if self.optimizer == "adam" else torch.optim.RMSprop
        opt = opt_cls(self.net.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        generator = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            for ids in torch.randperm(len(ty), generator=generator).split(BATCH):
                opt.zero_grad()
                loss_fn(self.net(tx[ids]), ty[ids]).backward()
                opt.step()
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.tensor(self.sx.transform(X), dtype=torch.float32)).numpy()
        return np.maximum(self.sy.inverse_transform(z).ravel(), 0)


def grid(model: str):
    n_hidden = 2 if model == "DFNN" else 1
    for neurons, act, opt, epochs in itertools.product(NEURONS[model], ACTIVATIONS, OPTIMIZERS, EPOCHS):
        yield dict(n_hidden=n_hidden, neurons=neurons, activation=act, optimizer=opt, epochs=epochs)


def predict_rows(train, target, cols, params, y_train, seed=SEED):
    """Entrena con las filas completas de ``train`` y predice ``target``.

    Con clima, las filas sin clima completo usan la misma red entrenada solo con
    casos (mismo criterio de respaldo que el resto del proyecto).
    """
    base = [c for c in cols if c not in CF]
    keep = train[cols].notna().all(axis=1).to_numpy()
    used = target[cols].notna().all(axis=1).to_numpy()
    pred = np.zeros(len(target))
    if len(base) < len(cols):
        pred = predict_rows(train, target, base, params, y_train, seed)[0]
        if keep.sum() < 100 or not used.any():
            return pred, np.zeros(len(target), dtype=bool)
    model = FeedForward(**params, seed=seed).fit(train.loc[keep, cols], y_train[keep])
    pred[used] = model.predict(target.loc[used, cols])
    return pred, used


def main(protocol: str, out: Path):
    cfg = PROTOCOLS[protocol]
    d = pd.read_csv(DATA / protocol / "sin_imputar.csv", dtype={"ubigeo": str},
                    parse_dates=["fecha_mes"], low_memory=False)
    d = d.sort_values(["fecha_mes", "ubigeo"]).reset_index(drop=True)
    year = d.fecha_mes.dt.year
    tr, va = d[year <= cfg["fit"]], d[(year > cfg["fit"]) & (year <= cfg["val"])]
    full, te = d[year <= cfg["val"]], d[year > cfg["val"]]
    scale = max(float((full.casos - full.lag12).abs().mean()), 1e-12)
    yv_alert, yt_alert = labels(va, tr), labels(te, full)
    out.mkdir(parents=True, exist_ok=True)
    rows, tuning, predictions = [], [], []
    for fname, cols in FEATURE_SETS.items():
        for model in NEURONS:
            t0, best = time.time(), None
            for params in grid(model):
                p, _ = predict_rows(tr, va, cols, params, tr.casos.to_numpy())
                rmse = float(np.sqrt(mean_squared_error(va.casos, p)))
                tuning.append(dict(modelo=model, entradas=fname, val_RMSE=rmse, **params))
                if best is None or rmse < best[0]:
                    best = (rmse, params, p)
            val_rmse, params, pv = best
            p, used = predict_rows(full, te, cols, params, full.casos.to_numpy())
            threshold = choose_threshold(yv_alert, pv)
            # Variación por semilla con la configuración elegida (la semilla 42 es la principal).
            seed_rmse = [float(np.sqrt(mean_squared_error(te.casos, predict_rows(
                full, te, cols, params, full.casos.to_numpy(), seed)[0]))) for seed in SEEDS[1:]]
            seed_rmse = [float(np.sqrt(mean_squared_error(te.casos, p)))] + seed_rmse
            rows.append(dict(modelo=model, entradas=fname, RMSE_validacion=val_rmse, **params,
                             RMSE_5semillas_media=float(np.mean(seed_rmse)),
                             RMSE_5semillas_sd=float(np.std(seed_rmse, ddof=1)),
                             uso_clima_prueba=float(used.mean()) if "clima" in fname else 0.0,
                             **metrics("casos", te.casos.to_numpy(), p, np.nan, scale),
                             **{f"alerta_{k}": v for k, v in
                                metrics("alerta", yt_alert, p, threshold, scale).items()}))
            predictions.append(te[["ubigeo", "fecha_mes"]].assign(
                modelo=model, entradas=fname, casos_observados=te.casos.to_numpy(), casos_predichos=p,
                alerta_observada=yt_alert, umbral_alerta=threshold))
            print(f"{fname} {model}: val RMSE {val_rmse:.3f} test RMSE {rows[-1]['RMSE']:.3f} "
                  f"{params} ({time.time() - t0:.0f} s)", flush=True)
            pd.DataFrame(rows).to_csv(out / "metricas.csv", index=False)
            pd.DataFrame(tuning).to_csv(out / "validacion.csv", index=False)
            pd.concat(predictions, ignore_index=True).to_csv(out / "predicciones.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocolo", choices=list(PROTOCOLS), default="oficial")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main(args.protocolo, ROOT / ("results/paper2" if args.protocolo == "oficial" else f"results/paper2_{args.protocolo}"))
