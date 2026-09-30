"""
=============================================================================
CORE — regime, anomaly, embedding, memory, calibrator, meta,
       11 engines, precision entry, grader, engine tracker, adaptive
=============================================================================
"""
import os
import json
import math
import sqlite3
import datetime
import threading
from contextlib import closing
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

from config import (
    STATE_DIR, GRADE_LOT, GRADE_RISK_PCT, grade_from_score,
    load_json, save_json, clip,
)


# =============================================================================
# REGIME DETECTION
# =============================================================================
class Regime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"
    TRANSITION = "TRANSITION"


@dataclass
class RegimeState:
    regime: Regime
    confidence: float
    adx: float
    atr_pct: float
    bb_pct: float
    slope: float
    description: str


def wilder_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or len(df) < 2:
        return pd.Series(dtype=float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / max(1, period), min_periods=period, adjust=False).mean()


def wilder_atr_value(df: pd.DataFrame, period: int = 14) -> float:
    try:
        s = wilder_atr_series(df, period).dropna()
        if len(s) == 0:
            return 8.0
        v = float(s.iloc[-1])
        return v if v > 0 and not math.isnan(v) else 8.0
    except Exception:
        return 8.0


def _rma(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(alpha=1.0 / max(1, p), min_periods=p, adjust=False).mean()


def _adx(df: pd.DataFrame, p: int = 14) -> float:
    try:
        if df is None or len(df) < p * 2:
            return 20.0
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        up = h.diff()
        dn = -l.diff()
        pdm = np.where((up > dn) & (up > 0), up, 0.0)
        mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr = pd.concat(
            [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
        ).max(axis=1)
        atr = _rma(tr, p)
        pdm_s = _rma(pd.Series(pdm, index=df.index), p)
        mdm_s = _rma(pd.Series(mdm, index=df.index), p)
        denom = atr.replace(0, np.nan) + 1e-9
        pdi = 100 * (pdm_s / denom).fillna(0.0)
        mdi = 100 * (mdm_s / denom).fillna(0.0)
        dx = (100 * (pdi - mdi).abs() / ((pdi + mdi).replace(0, np.nan) + 1e-9)).fillna(0.0)
        v = _rma(dx, p).iloc[-1]
        return float(np.clip(v if not pd.isna(v) else 20.0, 0.0, 100.0))
    except Exception:
        return 20.0


def _atr_pct(df: pd.DataFrame, p: int = 14, lb: int = 100) -> float:
    try:
        atr = wilder_atr_series(df, p).dropna()
        if len(atr) < 10:
            return 0.5
        t = atr.tail(min(len(atr), lb))
        return float((t < atr.iloc[-1]).mean())
    except Exception:
        return 0.5


def _bb_pct(df: pd.DataFrame, p: int = 20, lb: int = 100) -> float:
    try:
        c = df["close"].astype(float)
        ma = c.rolling(p).mean()
        sd = c.rolling(p).std().fillna(0.0)
        w = ((4 * sd) / (ma.replace(0, np.nan) + 1e-9)).dropna()
        if len(w) < 10:
            return 0.5
        t = w.tail(min(len(w), lb))
        return float((t < w.iloc[-1]).mean())
    except Exception:
        return 0.5


def _slope(df: pd.DataFrame, span: int = 50, lb: int = 10) -> float:
    try:
        c = df["close"].astype(float)
        e = c.ewm(span=span, adjust=False).mean()
        return float((e.iloc[-1] - e.iloc[-lb]) / (wilder_atr_value(df) + 1e-9))
    except Exception:
        return 0.0


def detect_regime(df: pd.DataFrame) -> RegimeState:
    if df is None or len(df) < 60:
        return RegimeState(
            Regime.TRANSITION, 0.0, 20.0, 0.5, 0.5, 0.0, "insufficient_data"
        )
    adx = _adx(df)
    ap = _atr_pct(df)
    bp = _bb_pct(df)
    sl = _slope(df)

    regime, conf = Regime.TRANSITION, 0.40
    if ap >= 0.80 and bp >= 0.80:
        regime = Regime.VOLATILE
        conf = min(0.98, 0.65 + (ap - 0.80) * 1.5)
    elif adx >= 24.0 and abs(sl) > 0.80:
        regime = Regime.TRENDING_UP if sl > 0 else Regime.TRENDING_DOWN
        conf = min(0.98, 0.55 + (adx - 24.0) / 40.0)
    elif ap <= 0.28 and bp <= 0.32:
        regime = Regime.QUIET
        conf = min(0.92, 0.55 + (0.28 - ap) * 1.8)
    elif adx < 20.0 and bp <= 0.55:
        regime = Regime.RANGING
        conf = min(0.90, 0.50 + (20.0 - adx) / 35.0)

    return RegimeState(
        regime=regime,
        confidence=round(conf, 3),
        adx=round(adx, 2),
        atr_pct=round(ap, 3),
        bb_pct=round(bp, 3),
        slope=round(sl, 4),
        description=f"{regime.value} ADX={adx:.1f} ATR%={ap:.2f} BB%={bp:.2f} Sl={sl:.2f}",
    )


# =============================================================================
# ANOMALY DETECTION (MAD robust)
# =============================================================================
@dataclass
class AnomalyReport:
    score: float
    is_anomaly: bool
    reason: str


def _robust_z(x: np.ndarray) -> float:
    if len(x) < 5:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    return float(abs(x[-1] - med) / (1.4826 * mad))


def detect_anomaly(df: pd.DataFrame, z_th: float = 3.5) -> AnomalyReport:
    if df is None or len(df) < 30:
        return AnomalyReport(0.0, False, "insufficient_bars")
    try:
        c = df["close"].astype(float).values
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        has_vol = "volume" in df and df["volume"].sum() > 0 and df["volume"].std() > 1e-6
        v = df["volume"].astype(float).values if has_vol else None

        ret = np.diff(c) / (c[:-1] + 1e-9)
        rng = (h - l) / (c + 1e-9)

        rz = _robust_z(ret[-50:] if len(ret) >= 50 else ret)
        rgz = _robust_z(rng[-50:] if len(rng) >= 50 else rng)
        vz = _robust_z(v[-50:] if (has_vol and len(v) >= 50) else (v if has_vol else np.zeros(1)))

        wr, wrg, wv = (0.40, 0.35, 0.25) if has_vol else (0.55, 0.45, 0.0)
        score = wr * min(1.0, rz / z_th) + wrg * min(1.0, rgz / z_th) + wv * min(1.0, vz / z_th)
        score = float(np.clip(score, 0.0, 1.0))

        reasons = []
        if rz >= z_th:
            reasons.append(f"Ret Z={rz:.1f}")
        if rgz >= z_th:
            reasons.append(f"Range Z={rgz:.1f}")
        if has_vol and vz >= z_th:
            reasons.append(f"Vol Z={vz:.1f}")

        return AnomalyReport(round(score, 3), score >= 0.65, "; ".join(reasons) or "normal")
    except Exception as e:
        return AnomalyReport(0.0, False, f"err:{str(e)[:40]}")


# =============================================================================
# 32-DIM EMBEDDING
# =============================================================================
EMB_DIM = 32


def _moments(x: np.ndarray, k: int = 4) -> List[float]:
    if len(x) < 4:
        return [0.0] * k
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-9
    z = (x - m) / s
    return [
        m, s,
        float(np.clip(np.mean(z ** 3), -10, 10)),
        float(np.clip(np.mean(z ** 4) - 3, -10, 20)),
    ]


def encode(df: pd.DataFrame) -> np.ndarray:
    if df is None or len(df) < 20:
        return np.zeros(EMB_DIM, dtype=np.float32)
    try:
        c = df["close"].astype(float).values
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        v = df["volume"].astype(float).values if "volume" in df else np.ones(len(c))

        ret = np.diff(c) / (c[:-1] + 1e-9)
        rng = (h - l) / (c + 1e-9)

        feats: List[float] = []
        feats += _moments(ret)
        feats += _moments(rng)
        feats += _moments(np.log1p(np.abs(v)))

        if len(ret) >= 20:
            x = ret - ret.mean()
            d = np.dot(x, x) + 1e-9
            for lag in (1, 2, 3, 5, 8, 13):
                if lag < len(x):
                    feats.append(float(np.clip(np.dot(x[:-lag], x[lag:]) / d, -1, 1)))
                else:
                    feats.append(0.0)
        else:
            feats += [0.0] * 6

        if len(c) >= 16:
            fft = np.abs(np.fft.rfft(c - c.mean()))
            top = np.sort(fft)[::-1][:8]
            tot = fft.sum() + 1e-9
            feats += [float(t / tot) for t in top]
        else:
            feats += [0.0] * 8

        if len(c) >= 10:
            qs = np.percentile(c, [10, 25, 50, 75, 90])
            norm = (qs - qs[0]) / (qs[-1] - qs[0] + 1e-9)
            feats += [float(q) for q in norm]
        else:
            feats += [0.0] * 5

        vec = np.array(feats[:EMB_DIM], dtype=np.float32)
        if len(vec) < EMB_DIM:
            vec = np.pad(vec, (0, EMB_DIM - len(vec)))
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        n = np.linalg.norm(vec) + 1e-9
        return (vec / n).astype(np.float32)
    except Exception:
        return np.zeros(EMB_DIM, dtype=np.float32)


# =============================================================================
# EPISODIC MEMORY (SQLite WAL)
# =============================================================================
DB_FILE = os.path.join(STATE_DIR, "agi_memory.db")


class Memory:
    def __init__(self, path: str = DB_FILE, max_entries: int = 2500):
        self.path = path
        self.max_entries = max_entries
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=15.0)
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        return c

    def _init_db(self):
        with closing(self._conn()) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS mem (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL, embedding BLOB, signal TEXT,
                    entry REAL, sl REAL, tp REAL,
                    outcome TEXT, pnl_r REAL, regime TEXT,
                    extra TEXT, resolved_ts REAL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_mem_outcome ON mem(outcome);")
            c.commit()

    def store(self, df: pd.DataFrame, regime_state: RegimeState,
              signal: str, entry: float, sl: float = 0.0, tp: float = 0.0,
              outcome: str = "OPEN", pnl_r: float = 0.0,
              extra: Optional[dict] = None) -> int:
        emb = encode(df)
        with closing(self._conn()) as c:
            cur = c.execute(
                "INSERT INTO mem (ts, embedding, signal, entry, sl, tp, outcome, pnl_r, regime, extra) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (datetime.datetime.now(datetime.timezone.utc).timestamp(),
                 emb.tobytes(), signal, float(entry), float(sl), float(tp),
                 outcome, float(pnl_r), regime_state.regime.value,
                 json.dumps(extra or {})),
            )
            c.commit()
            eid = cur.lastrowid
        self._prune()
        return eid or 0

    def resolve_open_trades(self, df: pd.DataFrame) -> int:
        if df is None or df.empty or len(df) < 2:
            return 0
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT id, ts, signal, entry, sl, tp FROM mem WHERE outcome='OPEN'"
            ).fetchall()
        if not rows:
            return 0

        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        idx_ts = (df.index.view("int64") / 1e9) if isinstance(df.index, pd.DatetimeIndex) else None

        resolved = 0
        for eid, ts, sig, entry, sl, tp in rows:
            if sl <= 0 or tp <= 0:
                continue
            start = int(np.searchsorted(idx_ts, ts)) if idx_ts is not None else 0
            risk = abs(entry - sl)
            if risk < 0.01:
                continue
            outcome = None
            pnl_r = 0.0
            for i in range(start, len(df)):
                if sig == "BUY":
                    if lows[i] <= sl:
                        outcome, pnl_r = "LOSS", -1.0
                        break
                    if highs[i] >= tp:
                        outcome, pnl_r = "WIN", abs(tp - entry) / risk
                        break
                else:
                    if highs[i] >= sl:
                        outcome, pnl_r = "LOSS", -1.0
                        break
                    if lows[i] <= tp:
                        outcome, pnl_r = "WIN", abs(entry - tp) / risk
                        break
            if outcome:
                with closing(self._conn()) as c:
                    c.execute(
                        "UPDATE mem SET outcome=?, pnl_r=?, resolved_ts=? WHERE id=?",
                        (outcome, pnl_r,
                         datetime.datetime.now(datetime.timezone.utc).timestamp(), eid),
                    )
                    c.commit()
                resolved += 1
        return resolved

    def _prune(self):
        with closing(self._conn()) as c:
            n = c.execute("SELECT COUNT(*) FROM mem").fetchone()[0]
            if n > self.max_entries:
                diff = n - self.max_entries
                c.execute(
                    f"DELETE FROM mem WHERE id IN "
                    f"(SELECT id FROM mem ORDER BY id ASC LIMIT {diff})"
                )
                c.commit()

    def query(self, df: pd.DataFrame, k: int = 20) -> Tuple[List[dict], dict]:
        q = encode(df)
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT id, embedding, signal, entry, outcome, pnl_r, regime "
                "FROM mem WHERE outcome IN ('WIN','LOSS','DRAW') "
                "ORDER BY id DESC LIMIT 500"
            ).fetchall()
        if not rows:
            return [], {"n": 0, "winrate": 50.0, "avg_r": 0.0, "similarity": 0.0}

        valid, embs = [], []
        for r in rows:
            e = np.frombuffer(r[1], dtype=np.float32)
            if len(e) == EMB_DIM:
                valid.append(r)
                embs.append(e)
        if not embs:
            return [], {"n": 0, "winrate": 50.0, "avg_r": 0.0, "similarity": 0.0}

        mat = np.vstack(embs)
        sims = np.dot(mat, q)
        order = np.argsort(-sims)[:k]

        wins = losses = 0
        rs: List[float] = []
        for i in order:
            r = valid[i]
            if r[4] == "WIN":
                wins += 1
                rs.append(r[5])
            elif r[4] == "LOSS":
                losses += 1
                rs.append(r[5])

        tot = wins + losses
        return [], {
            "n": tot,
            "winrate": round((wins / tot) * 100.0, 1) if tot > 0 else 50.0,
            "avg_r": round(float(np.mean(rs)), 3) if rs else 0.0,
            "similarity": round(float(np.mean([sims[i] for i in order])), 3) if len(order) else 0.0,
        }


# =============================================================================
# CALIBRATOR (Platt scaling)
# =============================================================================
CAL_FILE = os.path.join(STATE_DIR, "calibration.json")
CAL_SAMPLES = os.path.join(STATE_DIR, "calib_samples.json")


class Calibrator:
    def __init__(self, path: str = CAL_FILE):
        self.path = path
        self.a, self.b = 1.0, 0.0
        self.fitted = False
        self.n_samples = 0
        d = load_json(path, {})
        if d:
            self.a = float(d.get("a", 1.0))
            self.b = float(d.get("b", 0.0))
            self.fitted = bool(d.get("fitted", False))
            self.n_samples = int(d.get("n_samples", 0))

    def save(self):
        save_json(self.path, {
            "a": self.a, "b": self.b,
            "fitted": self.fitted, "n_samples": self.n_samples,
        })

    def calibrate(self, p: float) -> float:
        if not self.fitted:
            return float(np.clip(p, 0.0, 1.0))
        z = np.clip(self.a * float(p) + self.b, -30.0, 30.0)
        return float(1.0 / (1.0 + np.exp(-z)))

    def accumulate(self, conf: float, y: int):
        samples = load_json(CAL_SAMPLES, [])
        samples.append({"conf": round(float(conf), 4), "y": int(y)})
        samples = samples[-800:]
        save_json(CAL_SAMPLES, samples)
        if len(samples) >= 20 and len(samples) % 5 == 0:
            self.fit_from_samples()

    def fit_from_samples(self):
        samples = load_json(CAL_SAMPLES, [])
        if len(samples) < 20:
            return
        X = np.array([float(s["conf"]) for s in samples])
        Y = np.array([float(s["y"]) for s in samples])
        if len(np.unique(Y)) < 2:
            return

        a, b = 1.0, 0.0
        for _ in range(50):
            z = np.clip(a * X + b, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - Y
            w = np.clip(p * (1.0 - p), 1e-4, 0.25)
            haa = np.mean(w * X * X) + 1e-4
            hbb = np.mean(w) + 1e-4
            a -= 0.3 * float(np.clip(np.mean(err * X) / haa, -0.5, 0.5))
            b -= 0.3 * float(np.clip(np.mean(err) / hbb, -0.5, 0.5))

        self.a, self.b = float(a), float(b)
        self.fitted = True
        self.n_samples = len(X)
        self.save()


# =============================================================================
# META LEARNER
# =============================================================================
META_FILE = os.path.join(STATE_DIR, "meta.json")


class MetaLearner:
    def __init__(self, path: str = META_FILE, alpha: float = 0.12):
        self.path = path
        self.alpha = alpha
        self.state: Dict[str, dict] = load_json(path, {})

    def update(self, regime: str, win: bool, pnl_r: float):
        s = self.state.setdefault(regime, {
            "n": 0, "wins": 0, "ema_wr": 0.50, "ema_r": 0.0, "penalty": 1.00,
        })
        s["n"] += 1
        if win:
            s["wins"] += 1
        s["ema_wr"] = (1.0 - self.alpha) * s["ema_wr"] + self.alpha * (1.0 if win else 0.0)
        s["ema_r"] = (1.0 - self.alpha) * s["ema_r"] + self.alpha * float(pnl_r)
        if s["ema_wr"] < 0.40 or s["ema_r"] < -0.15:
            s["penalty"] = max(0.60, s["penalty"] * 0.98)
        elif s["ema_wr"] > 0.55 and s["ema_r"] > 0.10:
            s["penalty"] = min(1.20, s["penalty"] * 1.02)
        save_json(self.path, self.state)

    def penalty(self, regime: str) -> float:
        return float(self.state.get(regime, {}).get("penalty", 1.0))


# =============================================================================
# ENGINES (11)
# =============================================================================
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr_s(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()


def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()
    rs = g / (l.replace(0, np.nan) + 1e-9)
    return (100 - 100 / (1 + rs)).fillna(50.0)


@dataclass
class SwingPoint:
    idx: int
    price: float
    kind: str


def find_swings(df: pd.DataFrame, lookback: int = 4, limit: int = 25) -> List[SwingPoint]:
    out: List[SwingPoint] = []
    if df is None or len(df) < lookback * 2 + 1:
        return out
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    n = len(df)
    for i in range(lookback, n - lookback):
        wh = h[i - lookback: i + lookback + 1]
        wl = l[i - lookback: i + lookback + 1]
        if h[i] == np.max(wh) and h[i] > h[i - 1]:
            out.append(SwingPoint(i, float(h[i]), "high"))
        elif l[i] == np.min(wl) and l[i] < l[i - 1]:
            out.append(SwingPoint(i, float(l[i]), "low"))
    return out[-limit:]


def _bos_choch(df: pd.DataFrame, swings: List[SwingPoint]) -> Dict:
    r = {"bos": None, "choch": None, "trend": "NEUTRAL", "level": None}
    if len(swings) < 4 or len(df) < 10:
        return r
    hs = [s for s in swings if s.kind == "high"]
    ls = [s for s in swings if s.kind == "low"]
    if len(hs) < 2 or len(ls) < 2:
        return r
    lc = float(df["close"].iloc[-1])
    hh = hs[-1].price > hs[-2].price
    hl = ls[-1].price > ls[-2].price
    lh = hs[-1].price < hs[-2].price
    ll = ls[-1].price < ls[-2].price
    if hh and hl:
        r["trend"] = "BULLISH"
    elif lh and ll:
        r["trend"] = "BEARISH"
    if r["trend"] == "BULLISH" and lc > hs[-1].price:
        r["bos"] = "BULLISH"
        r["level"] = hs[-1].price
    elif r["trend"] == "BEARISH" and lc < ls[-1].price:
        r["bos"] = "BEARISH"
        r["level"] = ls[-1].price
    if r["trend"] == "BULLISH" and lc < ls[-1].price:
        r["choch"] = "BEARISH"
        r["level"] = ls[-1].price
    elif r["trend"] == "BEARISH" and lc > hs[-1].price:
        r["choch"] = "BULLISH"
        r["level"] = hs[-1].price
    return r


@dataclass
class Zone:
    kind: str
    top: float
    bottom: float
    idx: int
    strength: float

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0


def detect_fvg(df: pd.DataFrame, max_zones: int = 10) -> List[Zone]:
    out: List[Zone] = []
    if df is None or len(df) < 5:
        return out
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    n = len(df)
    atr = (float(np.mean(h[-20:] - l[-20:])) if n >= 20 else 5.0) + 1e-9

    for i in range(2, n):
        if l[i] > h[i - 2]:
            gap = l[i] - h[i - 2]
            if gap >= atr * 0.15:
                mitigated = any(l[j] <= h[i - 2] for j in range(i + 1, n))
                if not mitigated:
                    out.append(Zone("FVG_BULL", float(l[i]), float(h[i - 2]),
                                    i, min(1.0, gap / atr)))
        elif h[i] < l[i - 2]:
            gap = l[i - 2] - h[i]
            if gap >= atr * 0.15:
                mitigated = any(h[j] >= l[i - 2] for j in range(i + 1, n))
                if not mitigated:
                    out.append(Zone("FVG_BEAR", float(l[i - 2]), float(h[i]),
                                    i, min(1.0, gap / atr)))
    return out[-max_zones:]


def detect_order_blocks(df: pd.DataFrame, max_obs: int = 10) -> List[Zone]:
    out: List[Zone] = []
    if df is None or len(df) < 10:
        return out
    o = df["open"].astype(float).values
    c = df["close"].astype(float).values
    n = len(df)
    body = np.abs(c - o)
    avg = pd.Series(body).rolling(15, min_periods=5).mean().values

    for i in range(2, n - 2):
        if np.isnan(avg[i]):
            continue
        nb = abs(c[i + 1] - o[i + 1])
        if nb <= avg[i] * 1.6:
            continue
        if c[i] < o[i] and c[i + 1] > o[i + 1]:
            out.append(Zone("OB_BULL", float(max(o[i], c[i])),
                            float(min(o[i], c[i])), i, 0.75))
        elif c[i] > o[i] and c[i + 1] < o[i + 1]:
            out.append(Zone("OB_BEAR", float(max(o[i], c[i])),
                            float(min(o[i], c[i])), i, 0.75))
    return out[-max_obs:]


def find_nearest_zone(zones: List[Zone], price: float,
                      prefix: str, below: bool) -> Optional[Zone]:
    c = [z for z in zones if z.kind.startswith(prefix)]
    if below:
        c = [z for z in c if z.top <= price]
        c.sort(key=lambda z: price - z.top)
    else:
        c = [z for z in c if z.bottom >= price]
        c.sort(key=lambda z: z.bottom - price)
    return c[0] if c else None


def e_trend_stack(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        c = df["close"].astype(float)
        if len(c) < 30:
            return (0, 1.0)
        e9 = _ema(c, 9).iloc[-1]
        e21 = _ema(c, 21).iloc[-1]
        e50 = _ema(c, 50).iloc[-1]
        e200 = _ema(c, min(200, len(c) - 1)).iloc[-1]
        p = c.iloc[-1]
        if p > e9 > e21 > e50 > e200:
            return (1, 2.2)
        if p < e9 < e21 < e50 < e200:
            return (-1, 2.2)
        if p > e21 and e9 > e21:
            return (1, 1.2)
        if p < e21 and e9 < e21:
            return (-1, 1.2)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def e_market_structure(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        s = _bos_choch(df, find_swings(df, 4, 20))
        if s["bos"] == "BULLISH":
            return (1, 2.0)
        if s["bos"] == "BEARISH":
            return (-1, 2.0)
        if s["choch"] == "BULLISH":
            return (1, 1.7)
        if s["choch"] == "BEARISH":
            return (-1, 1.7)
        if s["trend"] == "BULLISH":
            return (1, 1.0)
        if s["trend"] == "BEARISH":
            return (-1, 1.0)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def e_liquidity_sweep(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        if len(df) < 25:
            return (0, 1.0)
        swings = find_swings(df.iloc[:-2], 4, 15)
        hs = [x.price for x in swings if x.kind == "high"]
        ls = [x.price for x in swings if x.kind == "low"]
        if not hs or not ls:
            return (0, 1.0)
        prev = df.iloc[-2]
        last = df.iloc[-1]
        th = max(hs[-3:])
        tl = min(ls[-3:])
        if prev["low"] < tl and last["close"] > tl:
            return (1, 1.9)
        if prev["high"] > th and last["close"] < th:
            return (-1, 1.9)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def e_fvg(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        z = detect_fvg(df, 6)
        if not z:
            return (0, 1.0)
        p = float(df["close"].iloc[-1])
        last = z[-1]
        if last.kind == "FVG_BULL" and last.bottom <= p <= (last.top + 0.5):
            return (1, 1.6)
        if last.kind == "FVG_BEAR" and (last.bottom - 0.5) <= p <= last.top:
            return (-1, 1.6)
        if last.idx >= len(df) - 3:
            return (1 if last.kind == "FVG_BULL" else -1, 1.3)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def e_order_block(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        obs = detect_order_blocks(df, 6)
        if not obs:
            return (0, 1.0)
        p = float(df["close"].iloc[-1])
        last = obs[-1]
        if not (last.bottom <= p <= last.top):
            return (0, 1.0)
        c = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        rng = (c["high"] - c["low"]) + 1e-9
        if body / rng < 0.45:
            if last.kind == "OB_BULL":
                return (1, 1.8)
            if last.kind == "OB_BEAR":
                return (-1, 1.8)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def e_rsi_div(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        if len(df) < 35:
            return (0, 1.0)
        r = _rsi(df["close"].astype(float), 14)
        swings = find_swings(df, 3, 8)
        hs = [x for x in swings if x.kind == "high"][-2:]
        ls = [x for x in swings if x.kind == "low"][-2:]
        if len(hs) == 2 and hs[-1].price > hs[-2].price:
            rn = float(r.iloc[hs[-1].idx])
            rp = float(r.iloc[hs[-2].idx])
            if rn < rp and rn > 58.0:
                return (-1, 1.7)
        if len(ls) == 2 and ls[-1].price < ls[-2].price:
            rn = float(r.iloc[ls[-1].idx])
            rp = float(r.iloc[ls[-2].idx])
            if rn > rp and rn < 42.0:
                return (1, 1.7)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def e_volume_climax(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        if "volume" not in df or len(df) < 25:
            return (0, 1.0)
        v = df["volume"].astype(float)
        if v.std() < 1e-6 or v.iloc[-1] <= 0:
            return (0, 1.0)
        vma = v.rolling(20).mean().iloc[-1] + 1e-9
        if v.iloc[-1] < vma * 2.1:
            return (0, 1.0)
        c = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        uw = c["high"] - max(c["close"], c["open"])
        lw = min(c["close"], c["open"]) - c["low"]
        if lw > body * 2.0 and lw > uw:
            return (1, 1.5)
        if uw > body * 2.0 and uw > lw:
            return (-1, 1.5)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def e_vol_regime(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        a = _atr_s(df, 14)
        if len(a) < 30:
            return (0, 1.0)
        ca = a.iloc[-1]
        pa = a.iloc[-5]
        ma = a.rolling(30).mean().iloc[-1]
        if not (ca > pa * 1.15 and ca > ma):
            return (0, 1.0)
        c = df.iloc[-1]
        return (1, 1.4) if c["close"] > c["open"] else (-1, 1.4)
    except Exception:
        return (0, 1.0)


def e_session_momentum(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 10:
            return (0, 1.0)
        last = df.index[-1]
        utc = last.tz_convert("UTC") if last.tzinfo else last.tz_localize("UTC")
        mins = utc.hour * 60 + utc.minute
        if not ((420 <= mins <= 600) or (750 <= mins <= 930)):
            return (0, 1.0)
        rec = float(df["close"].iloc[-4:].mean())
        prev = float(df["close"].iloc[-8:-4].mean())
        if rec > prev * 1.0006:
            return (1, 1.5)
        if rec < prev * 0.9994:
            return (-1, 1.5)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def e_momentum_roc(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        c = df["close"].astype(float)
        if len(c) < 25:
            return (0, 1.0)
        roc = (c.iloc[-1] - c.iloc[-10]) / (c.iloc[-10] + 1e-9)
        rp = (c.iloc[-5] - c.iloc[-15]) / (c.iloc[-15] + 1e-9)
        if roc > 0.0012 and roc > rp:
            return (1, 1.3)
        if roc < -0.0012 and roc < rp:
            return (-1, 1.3)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def e_vwap_reversion(df: pd.DataFrame) -> Tuple[int, float]:
    try:
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 40:
            return (0, 1.0)
        day = df.index[-1].date()
        today = df[df.index.date == day]
        if len(today) < 10:
            return (0, 1.0)
        tp = (today["high"] + today["low"] + today["close"]) / 3.0
        vsum = today["volume"].sum() + 1e-9
        vwap = float((tp * today["volume"]).sum() / vsum)
        p = float(today["close"].iloc[-1])
        if abs(p - vwap) / (vwap + 1e-9) < 0.0008:
            return (0, 1.0)
        return (-1, 1.2) if p > vwap else (1, 1.2)
    except Exception:
        return (0, 1.0)


ENGINES = [
    ("TREND_STACK", e_trend_stack),
    ("MARKET_STRUCTURE", e_market_structure),
    ("LIQUIDITY_SWEEP", e_liquidity_sweep),
    ("FVG_DETECT", e_fvg),
    ("ORDER_BLOCK", e_order_block),
    ("RSI_DIVERGENCE", e_rsi_div),
    ("VOLUME_CLIMAX", e_volume_climax),
    ("VOLATILITY_REGIME", e_vol_regime),
    ("SESSION_MOMENTUM", e_session_momentum),
    ("MOMENTUM_ROC", e_momentum_roc),
    ("VWAP_REVERSION", e_vwap_reversion),
]


# =============================================================================
# PRECISION ENTRY
# =============================================================================
@dataclass
class PrecisionEntry:
    entry_low: float
    entry_high: float
    entry_ideal: float
    entry_type: str
    sl: float
    sl_reason: str
    tp1: float
    tp2: float
    tp3: float
    tp4: float
    tp_reasons: List[str]
    risk_points: float
    rr_tp1: float
    rr_tp2: float
    rr_tp3: float
    precision_score: float
    precision_grade: str
    lot_tier: str
    lot_multiplier: float
    risk_pct_recommended: float
    notes: List[str] = field(default_factory=list)


def _rounds(price: float, count: int = 3, step: float = 10.0) -> List[float]:
    base = round(price / step) * step
    return [base + i * step for i in range(-count, count + 1)]


def _swing_above(price: float, swings: List[SwingPoint]) -> Optional[float]:
    h = sorted([s.price for s in swings if s.kind == "high" and s.price > price + 0.5])
    return h[0] if h else None


def _swing_below(price: float, swings: List[SwingPoint]) -> Optional[float]:
    l = sorted([s.price for s in swings if s.kind == "low" and s.price < price - 0.5],
               reverse=True)
    return l[0] if l else None


def calculate_precise_entry(df: pd.DataFrame, signal: str, base_price: float,
                            h4_trend: str, h1_trend: str, consensus: float,
                            atr_value: float) -> PrecisionEntry:
    notes: List[str] = []
    score = 0.0
    swings = find_swings(df, 4, 25)
    fvgs = detect_fvg(df, 10)
    obs = detect_order_blocks(df, 10)

    entry_type = "STRUCTURE_PULLBACK"
    entry_ideal = entry_low = entry_high = base_price

    if signal == "BUY":
        fvg = find_nearest_zone(fvgs, base_price, "FVG_BULL", True)
        ob = find_nearest_zone(obs, base_price, "OB_BULL", True)
        cands = []
        if fvg and (base_price - fvg.top) <= atr_value * 2.5:
            cands.append(("FVG_FILL", fvg, fvg.strength + 0.25))
        if ob and (base_price - ob.top) <= atr_value * 2.5:
            cands.append(("OB_RETEST", ob, ob.strength + 0.20))
        if cands:
            cands.sort(key=lambda x: -x[2])
            entry_type = cands[0][0]
            z = cands[0][1]
            entry_high = min(z.top, base_price)
            entry_low = z.bottom
            entry_ideal = (entry_high + entry_low) / 2.0
            score += 22.0 if entry_type == "FVG_FILL" else 17.0
            notes.append(f"{entry_type} {entry_low:.2f}-{entry_high:.2f}")
        else:
            e21 = float(_ema(df["close"].astype(float), 21).iloc[-1])
            if base_price >= e21:
                entry_ideal = e21
                entry_low = entry_ideal - atr_value * 0.15
                entry_high = min(base_price, entry_ideal + atr_value * 0.15)
                score += 8.0
    else:  # SELL
        fvg = find_nearest_zone(fvgs, base_price, "FVG_BEAR", False)
        ob = find_nearest_zone(obs, base_price, "OB_BEAR", False)
        cands = []
        if fvg and (fvg.bottom - base_price) <= atr_value * 2.5:
            cands.append(("FVG_FILL", fvg, fvg.strength + 0.25))
        if ob and (ob.bottom - base_price) <= atr_value * 2.5:
            cands.append(("OB_RETEST", ob, ob.strength + 0.20))
        if cands:
            cands.sort(key=lambda x: -x[2])
            entry_type = cands[0][0]
            z = cands[0][1]
            entry_low = max(z.bottom, base_price)
            entry_high = z.top
            entry_ideal = (entry_high + entry_low) / 2.0
            score += 22.0 if entry_type == "FVG_FILL" else 17.0
            notes.append(f"{entry_type} {entry_low:.2f}-{entry_high:.2f}")
        else:
            e21 = float(_ema(df["close"].astype(float), 21).iloc[-1])
            if base_price <= e21:
                entry_ideal = e21
                entry_low = max(base_price, entry_ideal - atr_value * 0.15)
                entry_high = entry_ideal + atr_value * 0.15
                score += 8.0

    buffer = atr_value * 0.30
    if signal == "BUY":
        sw = _swing_below(entry_low, swings)
        if sw:
            sl = sw - buffer
            sl_reason = f"Below swing {sw:.2f}-{buffer:.2f}"
            score += 15.0
        else:
            sl = entry_low - atr_value * 1.3
            sl_reason = f"Vol stop {atr_value*1.3:.2f}pts"
            score += 5.0
        if (entry_ideal - sl) < atr_value * 0.7:
            sl = entry_ideal - atr_value * 0.9
            sl_reason += " (ATR floor)"
    else:
        sw = _swing_above(entry_high, swings)
        if sw:
            sl = sw + buffer
            sl_reason = f"Above swing {sw:.2f}+{buffer:.2f}"
            score += 15.0
        else:
            sl = entry_high + atr_value * 1.3
            sl_reason = f"Vol stop {atr_value*1.3:.2f}pts"
            score += 5.0
        if (sl - entry_ideal) < atr_value * 0.7:
            sl = entry_ideal + atr_value * 0.9
            sl_reason += " (ATR floor)"

    risk = max(abs(entry_ideal - sl), 0.1)
    tp_reasons: List[str] = []

    if signal == "BUY":
        sw = _swing_above(entry_high, swings)
        tp1 = sw if sw and (sw - entry_ideal) >= risk * 1.2 else entry_ideal + risk * 1.5
        tp_reasons.append(f"TP1 liquidity @ {tp1:.2f}")
        fb = find_nearest_zone(fvgs, entry_high, "FVG_BEAR", False)
        tp2 = fb.bottom if fb and fb.bottom > tp1 else entry_ideal + risk * 2.5
        tp_reasons.append(f"TP2 imbalance @ {tp2:.2f}")
        rnd = sorted([r for r in _rounds(entry_ideal, 4, 10.0) if r > tp2])
        tp3 = rnd[0] if rnd else entry_ideal + risk * 4.0
        tp_reasons.append(f"TP3 round @ {tp3:.2f}")
        tp4 = entry_ideal + risk * 6.0
    else:
        sw = _swing_below(entry_low, swings)
        tp1 = sw if sw and (entry_ideal - sw) >= risk * 1.2 else entry_ideal - risk * 1.5
        tp_reasons.append(f"TP1 liquidity @ {tp1:.2f}")
        fb = find_nearest_zone(fvgs, entry_low, "FVG_BULL", True)
        tp2 = fb.top if fb and fb.top < tp1 else entry_ideal - risk * 2.5
        tp_reasons.append(f"TP2 imbalance @ {tp2:.2f}")
        rnd = sorted([r for r in _rounds(entry_ideal, 4, 10.0) if r < tp2], reverse=True)
        tp3 = rnd[0] if rnd else entry_ideal - risk * 4.0
        tp_reasons.append(f"TP3 round @ {tp3:.2f}")
        tp4 = entry_ideal - risk * 6.0

    rr1 = abs(tp1 - entry_ideal) / risk
    rr2 = abs(tp2 - entry_ideal) / risk
    rr3 = abs(tp3 - entry_ideal) / risk

    if signal == "BUY":
        if "BULLISH" in h4_trend and "BULLISH" in h1_trend:
            score += 25.0
            notes.append("H1+H4 bull sync")
        elif "BULLISH" in h1_trend:
            score += 15.0
    else:
        if "BEARISH" in h4_trend and "BEARISH" in h1_trend:
            score += 25.0
            notes.append("H1+H4 bear sync")
        elif "BEARISH" in h1_trend:
            score += 15.0

    score += min(20.0, max(0.0, (consensus - 55.0) / 45.0 * 20.0))
    if rr1 >= 1.5:
        score += 10.0
    if rr2 >= 2.5:
        score += 10.0

    final = float(np.clip(score, 0.0, 100.0))
    grade = grade_from_score(final)
    tier = {"A Super": "🚀 A Super", "A++": "🔴 A++",
            "A": "🟢 A", "REJECT": "⚪ REJECT"}[grade]

    return PrecisionEntry(
        entry_low=round(entry_low, 2),
        entry_high=round(entry_high, 2),
        entry_ideal=round(entry_ideal, 2),
        entry_type=entry_type,
        sl=round(sl, 2),
        sl_reason=sl_reason,
        tp1=round(tp1, 2),
        tp2=round(tp2, 2),
        tp3=round(tp3, 2),
        tp4=round(tp4, 2),
        tp_reasons=tp_reasons,
        risk_points=round(risk, 2),
        rr_tp1=round(rr1, 2),
        rr_tp2=round(rr2, 2),
        rr_tp3=round(rr3, 2),
        precision_score=round(final, 1),
        precision_grade=grade,
        lot_tier=tier,
        lot_multiplier=GRADE_LOT[grade],
        risk_pct_recommended=GRADE_RISK_PCT[grade],
        notes=notes,
    )


# =============================================================================
# SIGNAL GRADER
# =============================================================================
def grade_signal(consensus: float, signal: str, engine_states: dict,
                 h1_trend: str, h4_trend: str, atr: float,
                 memory_stats: dict, anomaly_score: float,
                 regime_conf: float = 0.5) -> dict:
    s = 0.0
    s += max(0.0, min(30.0, (consensus - 55.0) / 45.0 * 30.0))

    want_bull = (signal == "BUY")
    h1_ok = (want_bull and "BULLISH" in h1_trend) or (not want_bull and "BEARISH" in h1_trend)
    h4_ok = (want_bull and "BULLISH" in h4_trend) or (not want_bull and "BEARISH" in h4_trend)
    if h1_ok and h4_ok:
        s += 25.0
    elif h1_ok:
        s += 15.0
    elif h4_ok:
        s += 8.0

    aw, tw = 0.0, 0.0
    for st in engine_states.values():
        sc = st.get("sc", 0)
        w = st.get("weight", 1.0)
        if sc != 0 and w > 0.05:
            tw += w
            if (sc > 0) == want_bull:
                aw += w
    if tw > 0:
        s += (aw / tw) * 20.0

    if 5.0 <= atr <= 16.0:
        s += 15.0
    elif atr < 5.0:
        s += max(0.0, (atr / 5.0) * 15.0)
    else:
        s += max(0.0, 15.0 - (atr - 16.0) * 1.5)

    if memory_stats.get("n", 0) >= 4:
        s += min(10.0, max(0.0, (memory_stats.get("winrate", 50.0) - 45.0) / 30.0 * 10.0))

    s += regime_conf * 5.0

    if anomaly_score >= 0.65:
        s -= 18.0
    elif anomaly_score >= 0.50:
        s -= 8.0

    final = float(np.clip(s, 0.0, 100.0))
    return {"score": round(final, 2), "grade": grade_from_score(final)}


# =============================================================================
# ENGINE TRACKER
# =============================================================================
TRACKER_FILE = os.path.join(STATE_DIR, "engine_tracker.json")


@dataclass
class EngineStat:
    n: int = 0
    correct: int = 0
    recent_correct: List[int] = field(default_factory=list)
    disabled: bool = False
    disabled_at: Optional[str] = None
    disabled_reason: str = ""

    def acc(self) -> float:
        return (self.correct / self.n) if self.n > 0 else 0.50

    def recent_acc(self, w: int = 30) -> float:
        if not self.recent_correct:
            return self.acc()
        window = self.recent_correct[-w:]
        return sum(window) / max(1, len(window))


class EngineTracker:
    def __init__(self, path: str = TRACKER_FILE, min_samples: int = 15,
                 disable_threshold: float = 0.38,
                 reenable_threshold: float = 0.52):
        self.path = path
        self.min_samples = min_samples
        self.disable_threshold = disable_threshold
        self.reenable_threshold = reenable_threshold
        self.lock = threading.RLock()
        self.data: Dict[str, Dict[str, EngineStat]] = {}

        raw = load_json(path, {})
        for regime, engines in raw.items():
            self.data[regime] = {}
            for name, s in engines.items():
                try:
                    self.data[regime][name] = EngineStat(**s)
                except Exception:
                    self.data[regime][name] = EngineStat()

    def _save(self):
        with self.lock:
            out = {r: {n: asdict(s) for n, s in e.items()}
                   for r, e in self.data.items()}
            save_json(self.path, out)

    def update(self, regime: str, engine_name: str,
               engine_signal: int, trade_signal: str, trade_result: str):
        if engine_signal == 0 or trade_result not in ("WIN", "LOSS"):
            return
        with self.lock:
            r = self.data.setdefault(regime, {})
            stat = r.setdefault(engine_name, EngineStat())
            was_bull = engine_signal > 0
            correct = (was_bull == (trade_signal == "BUY"))
            success = (correct and trade_result == "WIN") or \
                      (not correct and trade_result == "LOSS")

            stat.n += 1
            if success:
                stat.correct += 1
            stat.recent_correct.append(1 if success else 0)
            if len(stat.recent_correct) > 120:
                stat.recent_correct = stat.recent_correct[-120:]

            if stat.n >= self.min_samples:
                ra = stat.recent_acc(30)
                if ra < self.disable_threshold and not stat.disabled:
                    stat.disabled = True
                    stat.disabled_at = datetime.datetime.now(
                        datetime.timezone.utc).isoformat()
                    stat.disabled_reason = f"Recent {ra:.1%} < {self.disable_threshold:.1%}"
                elif ra >= self.reenable_threshold and stat.disabled:
                    stat.disabled = False
                    stat.disabled_at = None
                    stat.disabled_reason = "Recovered"
            self._save()

    def is_disabled(self, regime: str, engine_name: str) -> bool:
        with self.lock:
            s = self.data.get(regime, {}).get(engine_name)
            return s.disabled if s else False

    def get_weight_mult(self, regime: str, engine_name: str) -> float:
        with self.lock:
            s = self.data.get(regime, {}).get(engine_name)
            if s is None:
                return 1.0
            if s.disabled:
                return 0.0
            if s.n < self.min_samples:
                return 1.0
            return float(max(0.30, min(1.60, 0.50 + (s.recent_acc(30) - 0.30) * 1.6)))


# =============================================================================
# REGIME-AWARE ADAPTIVE WEIGHTS
# =============================================================================
REGIME_ENGINE_PRIOR: Dict[str, Dict[str, float]] = {
    "TRENDING_UP": {
        "TREND_STACK": 1.5, "MARKET_STRUCTURE": 1.4, "MOMENTUM_ROC": 1.3,
        "LIQUIDITY_SWEEP": 1.1, "VWAP_REVERSION": 0.6, "VOLATILITY_REGIME": 0.9,
    },
    "TRENDING_DOWN": {
        "TREND_STACK": 1.5, "MARKET_STRUCTURE": 1.4, "MOMENTUM_ROC": 1.3,
        "LIQUIDITY_SWEEP": 1.1, "VWAP_REVERSION": 0.6, "VOLATILITY_REGIME": 0.9,
    },
    "RANGING": {
        "VWAP_REVERSION": 1.5, "ORDER_BLOCK": 1.3, "FVG_DETECT": 1.2,
        "RSI_DIVERGENCE": 1.3, "TREND_STACK": 0.7, "MOMENTUM_ROC": 0.7,
        "MARKET_STRUCTURE": 0.8,
    },
    "VOLATILE": {
        "VOLUME_CLIMAX": 1.4, "LIQUIDITY_SWEEP": 1.3, "VOLATILITY_REGIME": 1.3,
        "TREND_STACK": 0.9, "SESSION_MOMENTUM": 1.1,
    },
    "QUIET": {
        "FVG_DETECT": 1.3, "ORDER_BLOCK": 1.3, "VWAP_REVERSION": 1.1,
        "VOLUME_CLIMAX": 0.6, "VOLATILITY_REGIME": 0.7,
    },
    "TRANSITION": {},
}


def regime_engine_prior(regime: str, engine_name: str) -> float:
    return REGIME_ENGINE_PRIOR.get(regime, {}).get(engine_name, 1.0)


# =============================================================================
# ADAPTIVE THRESHOLD (adjust by rolling WR)
# =============================================================================
ADAPTIVE_FILE = os.path.join(STATE_DIR, "adaptive.json")


class AdaptiveThreshold:
    def __init__(self, path: str = ADAPTIVE_FILE,
                 base_confluence: float = 62.0, base_precision: float = 68.0,
                 min_confluence: float = 55.0, max_confluence: float = 75.0,
                 min_precision: float = 60.0, max_precision: float = 85.0):
        self.path = path
        self.base_conf = base_confluence
        self.base_prec = base_precision
        self.min_conf, self.max_conf = min_confluence, max_confluence
        self.min_prec, self.max_prec = min_precision, max_precision
        d = load_json(path, {})
        self.recent_results: List[int] = d.get("recent", [])[-100:]

    def record_outcome(self, win: bool):
        self.recent_results.append(1 if win else 0)
        self.recent_results = self.recent_results[-100:]
        save_json(self.path, {"recent": self.recent_results})

    def _recent_wr(self, window: int = 30) -> float:
        if not self.recent_results:
            return 0.50
        w = self.recent_results[-window:]
        return sum(w) / len(w)

    def adjusted_thresholds(self) -> Tuple[float, float]:
        wr = self._recent_wr(30)
        n = len(self.recent_results)
        if n < 15:
            return self.base_conf, self.base_prec
        shift = (0.50 - wr) * 16.0
        c = float(np.clip(self.base_conf + shift, self.min_conf, self.max_conf))
        p = float(np.clip(self.base_prec + shift, self.min_prec, self.max_prec))
        return round(c, 1), round(p, 1)
