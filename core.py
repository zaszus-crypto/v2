import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import os

@dataclass
class RegimeResult:
    regime: str
    confidence: float

@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float

@dataclass
class PreciseEntry:
    entry_type: str
    entry_low: float
    entry_high: float
    entry_ideal: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    rr_tp1: float
    rr_tp2: float
    rr_tp3: float
    risk_points: float
    precision_score: float
    precision_grade: str
    lot_multiplier: float

class Calibrator:
    def __init__(self):
        pass
    def calibrate(self, score: float) -> float:
        return score

def wilder_atr_value(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 2:
        return 5.0
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-1])

def detect_regime(df: pd.DataFrame) -> RegimeResult:
    if len(df) < 50:
        return RegimeResult("TRENDING", 0.7)
    close = df['close'].astype(float)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    std = close.rolling(20).std()
    bb_width = (std * 4) / sma20
    is_volatile = bb_width.iloc[-1] > bb_width.rolling(50).mean().iloc[-1] * 1.5
    if is_volatile:
        return RegimeResult("VOLATILE", 0.85)
    if abs(sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1] < 0.002:
        return RegimeResult("RANGING", 0.75)
    return RegimeResult("TRENDING", 0.8)

def detect_anomaly(df: pd.DataFrame) -> AnomalyResult:
    if len(df) < 20:
        return AnomalyResult(False, 0.0)
    returns = df['close'].astype(float).pct_change().dropna()
    z = abs(returns.iloc[-1] - returns.mean()) / (returns.std() + 1e-9)
    if z > 3.0:
        return AnomalyResult(True, float(z))
    return AnomalyResult(False, float(z))

def engine_ma(df: pd.DataFrame) -> Tuple[float, float]:
    c = df['close'].astype(float)
    fast = c.rolling(9).mean()
    slow = c.rolling(21).mean()
    diff = fast.iloc[-1] - slow.iloc[-1]
    score = np.clip(diff * 10.0, -100.0, 100.0)
    return float(score), 1.2

def engine_macd(df: pd.DataFrame) -> Tuple[float, float]:
    c = df['close'].astype(float)
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd.iloc[-1] - signal.iloc[-1]
    score = np.clip(hist * 20.0, -100.0, 100.0)
    return float(score), 1.5

def engine_rsi(df: pd.DataFrame, period: int = 14) -> Tuple[float, float]:
    c = df['close'].astype(float)
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    score = 0.0
    if val < 30:
        score = 80.0
    elif val > 70:
        score = -80.0
    else:
        score = (50 - val) * 1.5
    return float(score), 1.0

ENGINES = [
    ("MA", engine_ma),
    ("MACD", engine_macd),
    ("RSI", engine_rsi),
]

def regime_engine_prior(regime: str, engine_name: str) -> float:
    return 1.0

class MetaLearner:
    def penalty(self, regime: str) -> float:
        if regime == "VOLATILE":
            return 0.85
        return 1.0

class EngineTracker:
    def get_weight_mult(self, regime: str, name: str) -> float:
        return 1.0

class AdaptiveThreshold:
    def __init__(self, base_confluence: float, base_precision: float):
        self.base_confluence = base_confluence
        self.base_precision = base_precision
        self.recent_results: List[bool] = []

    def adjusted_thresholds(self) -> Tuple[float, float]:
        return self.base_confluence, self.base_precision

class Memory:
    def __init__(self, path: str = ".state_cache/memory.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def query(self, df: pd.DataFrame, k: int = 20) -> Tuple[dict, dict]:
        return {}, {"n": 10, "winrate": 65.0}

    def resolve_open_trades(self, df: pd.DataFrame) -> int:
        return 0

    def store(self, df, regime, signal, entry, sl, tp, extra):
        pass

def grade_signal(confluence: float, signal: str, states: dict, h1: str, h4: str, atr: float, mem: dict, anomaly: float, reg_conf: float) -> dict:
    score = confluence
    if score >= 80.0:
        grade = "A Super"
    elif score >= 70.0:
        grade = "A++"
    else:
        grade = "A"
    return {"grade": grade, "score": score}

def calculate_precise_entry(df: pd.DataFrame, signal: str, price: float, h4_trend: str, h1_trend: str, confluence: float, atr: float) -> PreciseEntry:
    if signal == "BUY":
        ideal = price - (atr * 0.2)
        low = ideal - (atr * 0.3)
        high = ideal + (atr * 0.1)
        sl = ideal - (atr * 1.5)
        tp1 = ideal + (atr * 1.5)
        tp2 = ideal + (atr * 3.0)
        tp3 = ideal + (atr * 4.5)
    else:
        ideal = price + (atr * 0.2)
        low = ideal - (atr * 0.1)
        high = ideal + (atr * 0.3)
        sl = ideal + (atr * 1.5)
        tp1 = ideal - (atr * 1.5)
        tp2 = ideal - (atr * 3.0)
        tp3 = ideal - (atr * 4.5)

    risk_pts = abs(ideal - sl)
    return PreciseEntry(
        entry_type="LIMIT / ZONE",
        entry_low=round(low, 2),
        entry_high=round(high, 2),
        entry_ideal=round(ideal, 2),
        sl=round(sl, 2),
        tp1=round(tp1, 2),
        tp2=round(tp2, 2),
        tp3=round(tp3, 2),
        rr_tp1=1.5,
        rr_tp2=3.0,
        rr_tp3=4.5,
        risk_points=round(risk_pts, 2),
        precision_score=85.0,
        precision_grade="OPTIMAL",
        lot_multiplier=1.0
    )
