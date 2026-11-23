"""Modelos comparados en el proyecto (Grupo 4, leishmaniosis cutánea en Cajamarca).

Se implementan cinco modelos. Cada uno resuelve dos tareas:

- ``casos`` (regresión): número de casos del distrito en el mes siguiente.
- ``alerta`` (clasificación): si ese mes supera media + 2 desviaciones estándar
  del distrito en el entrenamiento.

| Modelo          | Regresión                 | Clasificación              |
|-----------------|---------------------------|----------------------------|
| lineal          | Ridge                     | Regresión logística        |
| arbol           | DecisionTreeRegressor     | DecisionTreeClassifier     |
| random_forest   | RandomForestRegressor     | RandomForestClassifier     |
| mlp             | MLPRegressor              | MLPClassifier              |
| cnn             | TemporalCNN (PyTorch)     | TemporalCNN (PyTorch)      |

Cada modelo tiene dos configuraciones candidatas (``candidate`` = 0 o 1); la
elección entre ellas se hace con el periodo de validación en
``experimento_clima.run_experiments``, nunca con la prueba.

Uso::

    from models import make_model
    modelo = make_model("random_forest", "casos", candidate=1, climate=False)
    modelo.fit(X, y)
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42
MODELS = ["lineal", "arbol", "random_forest", "mlp", "cnn"]

# MLP: con el límite original de 150 iteraciones los 41 ajustes terminaban sin
# converger. Se usa early stopping: se reserva el 10 % del tramo de ajuste y el
# entrenamiento se detiene tras 20 épocas sin mejora (no usa validación ni prueba).
MLP_MAX_ITER = 1000


class TemporalCNN:
    """Red convolucional 1D sobre los tres meses anteriores.

    Entradas (en este orden, ver ``preprocessing.FEATURES``):
    ``[lag1, lag2, lag3, lag12, media3, mes_sin, mes_cos]`` y, si ``climate``,
    los 9 rezagos de clima (lluvia, tmax, tmin × meses 1-3).

    - Rama convolucional: Conv1d de kernel 2 sobre la secuencia de 3 meses
      (1 canal de casos, o 4 canales con clima), ReLU y aplanado.
    - Rama de contexto: lag12, media3 y estacionalidad (seno/coseno).
    - Cabeza densa de 16 neuronas y una salida (conteo o logit de alerta).

    Interfaz mínima compatible con el resto del pipeline: ``fit`` y
    ``predict_score``.
    """

    def __init__(self, climate: bool, task: str, width: int):
        self.climate, self.task, self.width = climate, task, width

    def fit(self, X, y):
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        self.scaler = StandardScaler().fit(X)
        a = self.scaler.transform(X).astype("float32")
        channels = 4 if self.climate else 1
        # Columnas que forman la secuencia: lags de casos y, con clima, sus 9 rezagos.
        self.seq_cols = [0, 1, 2] + (list(range(7, 16)) if self.climate else [])
        width, context = self.width, 4
        self.conv = torch.nn.Sequential(torch.nn.Conv1d(channels, width, 2), torch.nn.ReLU(), torch.nn.Flatten())
        self.head = torch.nn.Sequential(torch.nn.Linear(width * 2 + context, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        self.model = torch.nn.ModuleList([self.conv, self.head])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.005, weight_decay=.001)
        tx = torch.from_numpy(a)
        ty = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        if self.task == "alerta":
            # Desbalance: las alertas pesan (negativos / positivos) veces más.
            pos = max(float(ty.sum()), 1.)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos))
        else:
            loss_fn = torch.nn.MSELoss()
        generator = torch.Generator().manual_seed(SEED)
        self.model.train()
        for _ in range(50):  # 50 épocas, lotes de 512
            for ids in torch.randperm(len(y), generator=generator).split(512):
                optimizer.zero_grad()
                # (lote, canales, 3 meses) en orden pasado -> presente.
                seq = tx[ids][:, self.seq_cols].reshape(-1, channels, 3).flip(-1)
                z = self.head(torch.cat([self.conv(seq), tx[ids][:, 3:7]], dim=1))
                loss = loss_fn(z, ty[ids])
                loss.backward()
                optimizer.step()
        return self

    def predict_score(self, X):
        """Conteo estimado (regresión) o probabilidad de alerta (clasificación)."""
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            seq = x[:, self.seq_cols].reshape(-1, 4 if self.climate else 1, 3).flip(-1)
            z = self.head(torch.cat([self.conv(seq), x[:, 3:7]], dim=1)).flatten()
            return (torch.sigmoid(z) if self.task == "alerta" else z).numpy()


def make_model(family: str, task: str, candidate: int, climate: bool):
    """Crea un modelo sin entrenar.

    Args:
        family: uno de ``MODELS``.
        task: ``"casos"`` (regresión) o ``"alerta"`` (clasificación).
        candidate: 0 o 1, la configuración candidata.
        climate: si las entradas incluyen clima (solo cambia la CNN).

    Los modelos lineales y la MLP se envuelven con ``StandardScaler`` ajustado
    solo con el entrenamiento; los árboles no necesitan escalado. En
    clasificación se usan pesos de clase balanceados.
    """
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
        raise ValueError(f"Modelo desconocido: {family}")
    return make_pipeline(StandardScaler(), m) if family in ("lineal", "mlp") else m
