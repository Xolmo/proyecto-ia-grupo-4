"""Invariantes relevantes: presupuesto, observaciones y causalidad temporal."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimento_clima import CLIMATE, CF, train_predict, seasonal_fill


class SeasonalFillTests(unittest.TestCase):
    def fixture(self):
        d = pd.DataFrame({"ubigeo": "060101", "fecha_mes": pd.date_range("2001-01-01", periods=72, freq="MS")})
        for v, offset in zip(CLIMATE, [0, 200, 100]):
            d[v] = np.arange(len(d), dtype=float) + offset
        return d

    def test_exact_budget_preserves_observations_and_sources_are_past(self):
        d = self.fixture()
        d.loc[24:59, CLIMATE] = np.nan
        filled, audit, stats = seasonal_fill(d, "estacional_52")
        self.assertEqual(len(audit), round(36 * 3 * .52))
        for v in CLIMATE:
            observed = d[v].notna()
            np.testing.assert_array_equal(d.loc[observed, v], filled.loc[observed, v])
        self.assertTrue((audit.fuente_fecha < audit.fecha_mes).all())
        self.assertEqual(stats["filled_cells"], stats["target_cells"])

    def test_short_gaps_do_not_extend_long_absence(self):
        d = self.fixture()
        d.loc[24:30, CLIMATE] = np.nan
        filled, audit, _ = seasonal_fill(d, "huecos_cortos")
        self.assertTrue(filled.loc[24:25, CLIMATE].notna().all().all())
        self.assertTrue(filled.loc[26:30, CLIMATE].isna().all().all())
        self.assertFalse(audit.fuente_imputada.any())

    def test_future_values_do_not_change_past_imputations(self):
        d = self.fixture()
        d.loc[24:30, CLIMATE] = np.nan
        first, _, _ = seasonal_fill(d, "huecos_cortos")
        d.loc[40:, CLIMATE] = 999999.
        second, _, _ = seasonal_fill(d, "huecos_cortos")
        pd.testing.assert_frame_equal(first.loc[:39, CLIMATE], second.loc[:39, CLIMATE])

    def test_no_imputation_preserves_missing_cells(self):
        d = self.fixture()
        d.loc[24:30, CLIMATE] = np.nan
        filled, audit, stats = seasonal_fill(d, "sin_imputar")
        pd.testing.assert_frame_equal(d[CLIMATE], filled[CLIMATE])
        self.assertTrue(audit.empty)
        self.assertEqual(stats["initial_missing_cells"], stats["remaining_missing_cells"])

    def test_absent_climate_uses_identical_case_fallback(self):
        d = self.fixture()
        d["casos"] = np.arange(len(d)) % 5
        for k in (1, 2, 3, 12):
            d[f"lag{k}"] = d.casos.shift(k)
        d["media3"] = d.casos.shift(1).rolling(3).mean()
        d["mes_sin"] = np.sin(d.fecha_mes.dt.month * np.pi / 6)
        d["mes_cos"] = np.cos(d.fecha_mes.dt.month * np.pi / 6)
        for c in CF:
            d[c] = np.nan
        d = d.dropna(subset=["lag12"]).reset_index(drop=True)
        tr, te = d.iloc[:40], d.iloc[40:]
        base, _ = train_predict(tr, te, "lineal", "casos", 0, "solo_casos")
        pred, used = train_predict(tr, te, "lineal", "casos", 0, "sin_imputar")
        np.testing.assert_array_equal(base, pred)
        self.assertFalse(used.any())

    def test_rejects_imputed_minimum_above_observed_maximum(self):
        d = self.fixture()
        d.loc[24, "tmin_media"] = np.nan
        d.loc[24, "tmax_media"] = 10.
        filled, _, _ = seasonal_fill(d, "huecos_cortos")
        self.assertTrue(pd.isna(filled.loc[24, "tmin_media"]))
        self.assertFalse((filled.tmax_media < filled.tmin_media).any())


if __name__ == "__main__":
    unittest.main()
