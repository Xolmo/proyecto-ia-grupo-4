"""Pruebas del SARIMA del paper 4: orden del paper, causalidad y no negatividad."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modelo_paper4 as p4


def serie(n=96, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2001-01-01", periods=n, freq="MS")
    estacional = 3 + 2 * np.sin(2 * np.pi * np.arange(n) / 12)
    return pd.Series(rng.poisson(estacional).astype(float), index=idx)


class SarimaTests(unittest.TestCase):
    def test_orden_del_paper(self):
        self.assertEqual(p4.PAPER_ORDER, ((0, 1, 0), (1, 1, 0, 12)))
        self.assertIn(p4.PAPER_ORDER, p4.GRID)

    def test_pronostico_a_un_paso_no_usa_el_futuro(self):
        y = serie()
        y_fit = y.iloc[:72]
        base = p4.one_step(y_fit, y, p4.PAPER_ORDER)
        cambiada = y.copy()
        cambiada.iloc[-1] += 50  # cambiar el último mes no debe alterar las predicciones anteriores
        otra = p4.one_step(y_fit, cambiada, p4.PAPER_ORDER)
        self.assertEqual(len(base), len(y) - len(y_fit))
        np.testing.assert_allclose(base[:-1], otra[:-1])
        self.assertTrue((base >= 0).all())

    def test_seleccion_por_aic_devuelve_un_orden_de_la_cuadricula(self):
        order, aic, lb = p4.select_order(serie().iloc[:72])
        self.assertIn(order, p4.GRID)
        self.assertTrue(np.isfinite(aic))


if __name__ == "__main__":
    unittest.main()
