"""Pruebas de las redes del paper 2 y de las métricas de la comparación final."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparacion_final import alert_metrics, eer
from experimento_clima import CF
from modelo_paper2 import DROPOUT, FeedForward, predict_rows

PARAMS = dict(n_hidden=1, neurons=4, activation="relu", optimizer="adam", epochs=2)


def toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.poisson(1.0, (n, 3)), columns=["lag1", "lag2", "lag3"]).astype(float)
    d["casos"] = (d.lag1 + d.lag2) / 2
    return d


class ArchitectureTests(unittest.TestCase):
    def test_sfnn_has_one_hidden_layer_dropout_and_linear_output(self):
        net = FeedForward(**PARAMS)._build(3)
        kinds = [type(m) for m in net]
        self.assertEqual(kinds, [torch.nn.Linear, torch.nn.ReLU, torch.nn.Dropout, torch.nn.Linear])
        self.assertEqual(net[2].p, DROPOUT)
        self.assertEqual(net[-1].out_features, 1)          # una sola neurona, sin activación

    def test_dfnn_has_two_hidden_layers(self):
        net = FeedForward(**{**PARAMS, "n_hidden": 2})._build(3)
        self.assertEqual(sum(isinstance(m, torch.nn.Dropout) for m in net), 2)
        self.assertEqual(sum(isinstance(m, torch.nn.Linear) for m in net), 3)

    def test_same_seed_same_prediction_and_nonnegative(self):
        d = toy()
        a = FeedForward(**PARAMS, seed=1).fit(d[["lag1", "lag2", "lag3"]], d.casos).predict(d[["lag1", "lag2", "lag3"]])
        b = FeedForward(**PARAMS, seed=1).fit(d[["lag1", "lag2", "lag3"]], d.casos).predict(d[["lag1", "lag2", "lag3"]])
        np.testing.assert_array_equal(a, b)
        self.assertTrue((a >= 0).all())


class FallbackTests(unittest.TestCase):
    def test_without_climate_in_target_prediction_equals_cases_only(self):
        tr, te = toy(300, 1), toy(100, 2)
        for c in CF:
            tr[c], te[c] = 1.0, np.nan                     # el destino no tiene clima
        cols = ["lag1", "lag2", "lag3"]
        base, _ = predict_rows(tr, te, cols, PARAMS, tr.casos.to_numpy())
        withc, used = predict_rows(tr, te, cols + CF, PARAMS, tr.casos.to_numpy())
        np.testing.assert_array_equal(base, withc)
        self.assertFalse(used.any())


class MetricTests(unittest.TestCase):
    def test_eer_perfect_and_random(self):
        y = np.array([0] * 50 + [1] * 50)
        self.assertEqual(eer(y, np.r_[np.zeros(50), np.ones(50)]), 0.0)
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(eer(np.tile([0, 1], 2000), rng.random(4000)), 0.5, delta=0.05)

    def test_alert_metrics_counts_and_accuracy(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        score = np.array([.1, .2, .6, .1, .9, .3])
        m = alert_metrics(y, score, .5)
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["TN"]), (1, 1, 1, 3))
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertAlmostEqual(m["recall"], .5)


if __name__ == "__main__":
    unittest.main()
