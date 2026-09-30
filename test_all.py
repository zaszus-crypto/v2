"""
=============================================================================
TEST SUITE v35.0 — Full coverage
Run: python test_all.py   (or: pytest test_all.py -v)
=============================================================================
"""
import os
import sys
import json
import shutil
import tempfile
import datetime
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# FIXTURES
# =============================================================================
def make_df(n=300, seed=42, drift=0.0, vol=5.0, start=2000.0, tz=True):
    rng = np.random.default_rng(seed)
    c = start + np.cumsum(rng.normal(drift, vol, n))
    h = c + np.abs(rng.normal(0, vol, n))
    l = c - np.abs(rng.normal(0, vol, n))
    o = c + rng.normal(0, vol * 0.3, n)
    v = np.abs(rng.normal(1000, 300, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min",
                        tz="UTC" if tz else None)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v},
                        index=idx)


def make_uptrend_df(n=300, seed=1):
    return make_df(n, seed, drift=0.5, vol=3.0)


def make_downtrend_df(n=300, seed=2):
    return make_df(n, seed, drift=-0.5, vol=3.0)


# =============================================================================
# CONFIG
# =============================================================================
class TestConfig(unittest.TestCase):
    def test_grade_thresholds(self):
        from config import grade_from_score
        self.assertEqual(grade_from_score(95), "A Super")
        self.assertEqual(grade_from_score(92), "A Super")
        self.assertEqual(grade_from_score(85), "A++")
        self.assertEqual(grade_from_score(80), "A++")
        self.assertEqual(grade_from_score(70), "A")
        self.assertEqual(grade_from_score(65), "A")
        self.assertEqual(grade_from_score(50), "REJECT")

    def test_grade_at_least(self):
        from config import grade_at_least
        self.assertTrue(grade_at_least("A Super", "A"))
        self.assertTrue(grade_at_least("A++", "A"))
        self.assertTrue(grade_at_least("A", "A"))
        self.assertFalse(grade_at_least("A", "A++"))
        self.assertFalse(grade_at_least("A++", "A Super"))
        self.assertFalse(grade_at_least("REJECT", "A"))

    def test_resolve_grade_risk(self):
        from config import resolve_grade_to_risk
        pct, usd, lot = resolve_grade_to_risk("A Super", equity=10000)
        self.assertEqual(pct, 3.0)
        self.assertEqual(usd, 300.0)
        self.assertEqual(lot, 3.0)
        pct, _, _ = resolve_grade_to_risk("A", override_pct=1.0, equity=10000)
        self.assertEqual(pct, 1.0)

    def test_news_filter_nfp(self):
        from config import is_news_blocked, UTC
        dt = datetime.datetime(2024, 1, 5, 12, 30, tzinfo=UTC)  # Friday
        blocked, label = is_news_blocked(dt, block_min=45)
        self.assertTrue(blocked)
        self.assertIn("NFP", label)

    def test_news_filter_clear(self):
        from config import is_news_blocked, UTC
        dt = datetime.datetime(2024, 1, 5, 3, 0, tzinfo=UTC)
        blocked, _ = is_news_blocked(dt, block_min=45)
        self.assertFalse(blocked)

    def test_utils_resample(self):
        from config import resample_ohlcv
        df = make_df(200)
        r4 = resample_ohlcv(df, "4h")
        self.assertIsNotNone(r4)
        self.assertLess(len(r4), len(df))
        self.assertTrue((r4["high"] >= r4["low"]).all())

    def test_safe_float(self):
        from config import safe_float
        self.assertEqual(safe_float("3.14"), 3.14)
        self.assertEqual(safe_float(None, default=-1), -1)
        self.assertEqual(safe_float(float("nan"), default=0), 0)


# =============================================================================
# CORE
# =============================================================================
class TestRegime(unittest.TestCase):
    def test_atr_positive(self):
        from core import wilder_atr_value
        self.assertGreater(wilder_atr_value(make_df(100)), 0)

    def test_atr_uptrend_lower_than_volatile(self):
        from core import wilder_atr_value
        up = make_uptrend_df(200)
        vol = make_df(200, vol=15.0)
        self.assertLess(wilder_atr_value(up), wilder_atr_value(vol))

    def test_detect_regime_returns(self):
        from core import detect_regime, RegimeState
        r = detect_regime(make_df(200))
        self.assertIsInstance(r, RegimeState)
        self.assertGreaterEqual(r.confidence, 0.0)
        self.assertLessEqual(r.confidence, 1.0)

    def test_detect_regime_insufficient(self):
        from core import detect_regime, Regime
        r = detect_regime(make_df(10))
        self.assertEqual(r.regime, Regime.TRANSITION)


class TestAnomaly(unittest.TestCase):
    def test_anomaly_bounds(self):
        from core import detect_anomaly
        a = detect_anomaly(make_df(100))
        self.assertGreaterEqual(a.score, 0.0)
        self.assertLessEqual(a.score, 1.0)

    def test_anomaly_insufficient(self):
        from core import detect_anomaly
        a = detect_anomaly(make_df(10))
        self.assertEqual(a.score, 0.0)
        self.assertFalse(a.is_anomaly)

    def test_anomaly_spike(self):
        from core import detect_anomaly
        df = make_df(100)
        df.iloc[-1, df.columns.get_loc("close")] *= 1.10
        df.iloc[-1, df.columns.get_loc("high")] *= 1.12
        a = detect_anomaly(df)
        self.assertGreater(a.score, 0.2)


class TestEmbedding(unittest.TestCase):
    def test_encode_shape(self):
        from core import encode, EMB_DIM
        v = encode(make_df(100))
        self.assertEqual(v.shape, (EMB_DIM,))
        self.assertEqual(v.dtype, np.float32)

    def test_encode_normalized(self):
        from core import encode
        v = encode(make_df(100))
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=3)

    def test_encode_short_returns_zeros(self):
        from core import encode, EMB_DIM
        v = encode(make_df(5))
        self.assertEqual(v.shape, (EMB_DIM,))
        self.assertTrue(np.allclose(v, 0))


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_store_and_query(self):
        from core import Memory, detect_regime
        m = Memory(path=self.db)
        df = make_df(200)
        reg = detect_regime(df)
        eid = m.store(df, reg, "BUY", 2050.0, sl=2040.0, tp=2070.0)
        self.assertGreater(eid, 0)

    def test_query_empty(self):
        from core import Memory
        m = Memory(path=self.db)
        _, stats = m.query(make_df(200))
        self.assertEqual(stats["n"], 0)
        self.assertEqual(stats["winrate"], 50.0)

    def test_resolve_open_trades(self):
        from core import Memory, detect_regime
        m = Memory(path=self.db)
        df = make_df(200)
        reg = detect_regime(df)
        entry = float(df["close"].iloc[-1])
        m.store(df, reg, "BUY", entry, sl=entry - 10, tp=entry + 5)
        df2 = df.copy()
        df2.iloc[-1, df2.columns.get_loc("high")] = entry + 8
        n = m.resolve_open_trades(df2)
        self.assertGreaterEqual(n, 0)


class TestEngines(unittest.TestCase):
    def test_all_engines_signature(self):
        from core import ENGINES
        df = make_df(300)
        for name, fn in ENGINES:
            sc, w = fn(df)
            self.assertIn(sc, (-1, 0, 1), f"{name} bad signal {sc}")
            self.assertGreaterEqual(w, 0.0, f"{name} negative weight")
            self.assertLessEqual(w, 5.0, f"{name} weight too high {w}")

    def test_engines_short_df(self):
        from core import ENGINES
        df = make_df(10)
        for name, fn in ENGINES:
            sc, w = fn(df)
            self.assertEqual(sc, 0, f"{name} should return 0 on short df")

    def test_fvg_detection(self):
        from core import detect_fvg
        zones = detect_fvg(make_df(200))
        for z in zones:
            self.assertIn(z.kind, ("FVG_BULL", "FVG_BEAR"))
            self.assertGreater(z.top, z.bottom)

    def test_order_blocks(self):
        from core import detect_order_blocks
        obs = detect_order_blocks(make_df(200))
        for z in obs:
            self.assertIn(z.kind, ("OB_BULL", "OB_BEAR"))

    def test_swings_detection(self):
        from core import find_swings
        s = find_swings(make_df(200), 4, 20)
        self.assertLessEqual(len(s), 20)


class TestPrecision(unittest.TestCase):
    def test_buy_entry_valid(self):
        from core import calculate_precise_entry
        df = make_uptrend_df(300)
        p = float(df["close"].iloc[-1])
        e = calculate_precise_entry(df, "BUY", p, "BULLISH", "BULLISH", 75.0, 5.0)
        self.assertLessEqual(e.entry_low, e.entry_high)
        self.assertLess(e.sl, e.entry_ideal)
        self.assertGreater(e.tp1, e.entry_ideal)
        self.assertGreater(e.risk_points, 0)

    def test_sell_entry_valid(self):
        from core import calculate_precise_entry
        df = make_downtrend_df(300)
        p = float(df["close"].iloc[-1])
        e = calculate_precise_entry(df, "SELL", p, "BEARISH", "BEARISH", 75.0, 5.0)
        self.assertGreater(e.sl, e.entry_ideal)
        self.assertLess(e.tp1, e.entry_ideal)

    def test_tps_ordered_buy(self):
        from core import calculate_precise_entry
        df = make_uptrend_df(300)
        p = float(df["close"].iloc[-1])
        e = calculate_precise_entry(df, "BUY", p, "BULLISH", "BULLISH", 75.0, 5.0)
        self.assertLess(e.tp1, e.tp2)
        self.assertLess(e.tp2, e.tp3)

    def test_tps_ordered_sell(self):
        from core import calculate_precise_entry
        df = make_downtrend_df(300)
        p = float(df["close"].iloc[-1])
        e = calculate_precise_entry(df, "SELL", p, "BEARISH", "BEARISH", 75.0, 5.0)
        self.assertGreater(e.tp1, e.tp2)
        self.assertGreater(e.tp2, e.tp3)

    def test_grade_boundaries(self):
        from core import calculate_precise_entry
        df = make_df(300)
        p = float(df["close"].iloc[-1])
        e = calculate_precise_entry(df, "BUY", p, "NEUTRAL", "NEUTRAL", 50.0, 5.0)
        self.assertIn(e.precision_grade, ("A", "A++", "A Super", "REJECT"))
        self.assertGreaterEqual(e.precision_score, 0)
        self.assertLessEqual(e.precision_score, 100)


class TestGrader(unittest.TestCase):
    def test_grade_signal_bounds(self):
        from core import grade_signal
        states = {"e1": {"sc": 1, "weight": 2.0},
                  "e2": {"sc": 1, "weight": 1.5}}
        g = grade_signal(75, "BUY", states, "BULLISH", "BULLISH",
                         10.0, {"n": 5, "winrate": 60}, 0.1, 0.8)
        self.assertIn("score", g)
        self.assertIn("grade", g)
        self.assertGreaterEqual(g["score"], 0)
        self.assertLessEqual(g["score"], 100)

    def test_grade_signal_reject(self):
        from core import grade_signal
        states = {"e1": {"sc": 1, "weight": 1.0}}
        g = grade_signal(30, "BUY", states, "BEARISH", "BEARISH",
                         10.0, {"n": 0, "winrate": 50}, 0.0, 0.3)
        self.assertEqual(g["grade"], "REJECT")

    def test_anomaly_deduction(self):
        from core import grade_signal
        states = {"e1": {"sc": 1, "weight": 2.0}}
        g_clean = grade_signal(80, "BUY", states, "BULLISH", "BULLISH",
                               10.0, {"n": 10, "winrate": 60}, 0.0, 0.9)
        g_anom = grade_signal(80, "BUY", states, "BULLISH", "BULLISH",
                              10.0, {"n": 10, "winrate": 60}, 0.7, 0.9)
        self.assertLess(g_anom["score"], g_clean["score"])


class TestEngineTracker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "tracker.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_weight(self):
        from core import EngineTracker
        tr = EngineTracker(path=self.path)
        self.assertEqual(tr.get_weight_mult("TRENDING_UP", "TREND_STACK"), 1.0)

    def test_update_and_quarantine(self):
        from core import EngineTracker
        tr = EngineTracker(path=self.path, min_samples=10, disable_threshold=0.4)
        for _ in range(15):
            tr.update("RANGING", "TEST", 1, "BUY", "LOSS")
        self.assertTrue(tr.is_disabled("RANGING", "TEST"))
        self.assertEqual(tr.get_weight_mult("RANGING", "TEST"), 0.0)

    def test_recovery(self):
        from core import EngineTracker
        tr = EngineTracker(path=self.path, min_samples=10,
                            disable_threshold=0.4, reenable_threshold=0.5)
        for _ in range(15):
            tr.update("RANGING", "E", 1, "BUY", "LOSS")
        self.assertTrue(tr.is_disabled("RANGING", "E"))
        for _ in range(40):
            tr.update("RANGING", "E", 1, "BUY", "WIN")
        self.assertFalse(tr.is_disabled("RANGING", "E"))


class TestAdaptive(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_defaults(self):
        from core import AdaptiveThreshold
        a = AdaptiveThreshold(path=os.path.join(self.tmpdir, "a.json"))
        c, p = a.adjusted_thresholds()
        self.assertEqual(c, a.base_conf)
        self.assertEqual(p, a.base_prec)

    def test_tightens_on_low_wr(self):
        from core import AdaptiveThreshold
        a = AdaptiveThreshold(path=os.path.join(self.tmpdir, "b.json"),
                              base_confluence=65.0, base_precision=70.0)
        for _ in range(30):
            a.record_outcome(False)
        c, p = a.adjusted_thresholds()
        self.assertGreater(c, 65.0)
        self.assertGreater(p, 70.0)


class TestCalibrator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_unfitted_identity(self):
        from core import Calibrator
        c = Calibrator(path=os.path.join(self.tmpdir, "cal.json"))
        self.assertAlmostEqual(c.calibrate(0.7), 0.7, places=3)

    def test_fit_from_samples(self):
        from core import Calibrator
        import core as core_mod
        samples_path = os.path.join(self.tmpdir, "samples.json")
        data = [{"conf": i / 100.0, "y": 1 if i > 50 else 0} for i in range(100)]
        with open(samples_path, "w") as f:
            json.dump(data, f)
        old = core_mod.CAL_SAMPLES
        core_mod.CAL_SAMPLES = samples_path
        try:
            c = Calibrator(path=os.path.join(self.tmpdir, "cal.json"))
            c.fit_from_samples()
            self.assertTrue(c.fitted)
            self.assertGreater(c.calibrate(0.7), 0.5)
        finally:
            core_mod.CAL_SAMPLES = old


class TestMetaLearner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_penalty(self):
        from core import MetaLearner
        m = MetaLearner(path=os.path.join(self.tmpdir, "m.json"))
        self.assertEqual(m.penalty("TRENDING_UP"), 1.0)

    def test_penalty_decrease_on_losses(self):
        from core import MetaLearner
        m = MetaLearner(path=os.path.join(self.tmpdir, "m.json"))
        for _ in range(50):
            m.update("RANGING", win=False, pnl_r=-1.0)
        self.assertLess(m.penalty("RANGING"), 1.0)

    def test_penalty_increase_on_wins(self):
        from core import MetaLearner
        m = MetaLearner(path=os.path.join(self.tmpdir, "m.json"))
        for _ in range(50):
            m.update("TRENDING_UP", win=True, pnl_r=1.5)
        self.assertGreater(m.penalty("TRENDING_UP"), 1.0)


class TestRegimePrior(unittest.TestCase):
    def test_prior(self):
        from core import regime_engine_prior
        self.assertGreater(regime_engine_prior("RANGING", "VWAP_REVERSION"), 1.0)
        self.assertLess(regime_engine_prior("RANGING", "TREND_STACK"), 1.0)
        self.assertEqual(regime_engine_prior("UNKNOWN", "X"), 1.0)


# =============================================================================
# INTEGRATIONS
# =============================================================================
class TestIntegrations(unittest.TestCase):
    def test_escape_html(self):
        from integrations import esc
        self.assertEqual(esc("<b>x</b>"), "&lt;b&gt;x&lt;/b&gt;")
        self.assertEqual(esc("a & b"), "a &amp; b")

    def test_position_manager_open(self):
        from integrations import PositionManager
        tmpdir = tempfile.mkdtemp()
        try:
            pm = PositionManager(path=os.path.join(tmpdir, "pos.json"))
            p = pm.open_position("BUY", 2000, 1990, 2020, 2040, 2060,
                                 5.0, "A++", 1.5)
            self.assertIn(p.id, pm.positions)
            self.assertEqual(len(pm.active()), 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_position_manager_be(self):
        from integrations import PositionManager
        tmpdir = tempfile.mkdtemp()
        try:
            pm = PositionManager(path=os.path.join(tmpdir, "pos.json"))
            p = pm.open_position("BUY", 2000, 1990, 2020, 2040, 2060,
                                 5.0, "A++", 1.5)
            events = pm.update_all(2010.0)
            actions = [e["action"] for e in events]
            self.assertIn("MOVE_BE", actions)
            self.assertEqual(p.sl, 2000.0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_position_manager_close_sl(self):
        from integrations import PositionManager
        tmpdir = tempfile.mkdtemp()
        try:
            pm = PositionManager(path=os.path.join(tmpdir, "pos.json"))
            p = pm.open_position("BUY", 2000, 1990, 2020, 2040, 2060,
                                 5.0, "A++", 1.5)
            pm.update_all(1985.0)
            self.assertTrue(p.closed)
            self.assertLess(p.pnl_r, 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# BACKTEST
# =============================================================================
class TestBacktest(unittest.TestCase):
    def test_backtest_runs(self):
        from main import run_backtest
        df = make_df(400)
        r = run_backtest(df, df, df, min_conf=55.0, min_prec=55.0, min_grade="A")
        self.assertGreaterEqual(r.filled_trades, 0)
        self.assertGreaterEqual(r.total_signals, 0)

    def test_backtest_empty_df(self):
        from main import run_backtest
        r = run_backtest(pd.DataFrame(), None, None)
        self.assertEqual(r.filled_trades, 0)

    def test_metrics_consistency(self):
        from main import run_backtest
        df = make_df(400, seed=99, drift=0.3)
        r = run_backtest(df, df, df, min_conf=55.0, min_prec=55.0, min_grade="A")
        if r.filled_trades > 0:
            self.assertEqual(r.wins + r.losses, r.filled_trades)
            self.assertGreaterEqual(r.winrate, 0)
            self.assertLessEqual(r.winrate, 100)
            self.assertGreaterEqual(r.max_dd_r, 0)


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("XAUUSD AGI v35.0 — TEST SUITE")
    print("=" * 70)
    unittest.main(verbosity=2)
