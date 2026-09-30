"""
=============================================================================
CONFIG — env, logger, utils, grade constants, news filter
=============================================================================
"""
import os
import json
import math
import logging
import datetime
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
import pytz

STATE_DIR = ".state_cache"
os.makedirs(STATE_DIR, exist_ok=True)
UTC = pytz.UTC

# =============================================================================
# GRADE SYSTEM (UNIFIED 3-TIER)
# =============================================================================
GRADE_THRESHOLDS = {"A": 65.0, "A++": 80.0, "A Super": 92.0}
GRADE_ORDER = ["REJECT", "A", "A++", "A Super"]
GRADE_LOT = {"A Super": 3.0, "A++": 1.5, "A": 0.5, "REJECT": 0.0}
GRADE_RISK_PCT = {"A Super": 3.0, "A++": 1.5, "A": 0.5, "REJECT": 0.0}


def grade_from_score(score: float) -> str:
    if score >= GRADE_THRESHOLDS["A Super"]:
        return "A Super"
    if score >= GRADE_THRESHOLDS["A++"]:
        return "A++"
    if score >= GRADE_THRESHOLDS["A"]:
        return "A"
    return "REJECT"


def grade_at_least(grade: str, minimum: str) -> bool:
    try:
        return GRADE_ORDER.index(grade) >= GRADE_ORDER.index(minimum)
    except ValueError:
        return False


def resolve_grade_to_risk(grade: str, override_pct: float = 0.0, equity: float = 10000.0):
    pct = override_pct if override_pct > 0 else GRADE_RISK_PCT.get(grade, 0.0)
    return pct, equity * (pct / 100.0), GRADE_LOT.get(grade, 0.0)


# =============================================================================
# ENV HELPERS
# =============================================================================
def _b(k: str, d: str = "false") -> bool:
    return os.getenv(k, d).strip().lower() in ("1", "true", "yes", "on")


def _f(k: str, d: float) -> float:
    try:
        return float(os.getenv(k, str(d)))
    except (TypeError, ValueError):
        return float(d)


def _i(k: str, d: int) -> int:
    try:
        return int(os.getenv(k, str(d)))
    except (TypeError, ValueError):
        return int(d)


def _s(k: str, d: str = "") -> str:
    return os.getenv(k, d).strip()


def _l(k: str, d: str = "") -> List[str]:
    return [x.strip() for x in os.getenv(k, d).split(",") if x.strip()]


# =============================================================================
# CONFIG DATACLASS
# =============================================================================
@dataclass
class Config:
    # AI
    gemini_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Telegram
    tg_token: str = ""
    tg_chats: List[str] = field(default_factory=list)
    tg_users: List[str] = field(default_factory=list)

    # Market feed
    symbol_mt5: str = "XAUUSD"
    symbol_deriv: str = "frxXAUUSD"
    yahoo_ticker: str = "GC=F"
    yahoo_auto: bool = True
    yahoo_fallback: float = -35.0
    feed_timeout: int = 8

    # Signal filters
    min_confluence: float = 62.0
    min_grade: str = "A"
    min_precision: float = 68.0
    cooldown_min: int = 45
    require_mtf: bool = True
    block_news_min: int = 45
    force_run: bool = False

    # Risk
    account_equity: float = 10000.0
    risk_override: float = 0.0

    # Ops
    log_level: str = "INFO"
    log_dir: str = ".state_cache/logs"
    healthcheck: str = ""
    interval: int = 300


def load_config() -> Config:
    c = Config(
        gemini_key=_s("GEMINI_API_KEY"),
        gemini_model=_s("GEMINI_MODEL", "gemini-2.0-flash"),
        tg_token=_s("TELEGRAM_BOT_TOKEN"),
        tg_chats=_l("TELEGRAM_CHAT_ID"),
        tg_users=_l("TELEGRAM_ALLOWED_USERS"),
        symbol_mt5=_s("SYMBOL_MT5", "XAUUSD"),
        symbol_deriv=_s("SYMBOL_DERIV", "frxXAUUSD"),
        yahoo_ticker=_s("YAHOO_TICKER", "GC=F"),
        yahoo_auto=_b("YAHOO_OFFSET_AUTO", "true"),
        yahoo_fallback=_f("YAHOO_OFFSET_FALLBACK", -35.0),
        feed_timeout=_i("FEED_TIMEOUT_SEC", 8),
        min_confluence=_f("MIN_CONFLUENCE_SCORE", 62.0),
        min_grade=_s("MIN_SIGNAL_GRADE", "A"),
        min_precision=_f("MIN_PRECISION_SCORE", 68.0),
        cooldown_min=_i("SIGNAL_COOLDOWN_MINUTES", 45),
        require_mtf=_b("REQUIRE_MTF_ALIGN", "true"),
        block_news_min=_i("BLOCK_NEWS_WINDOW_MIN", 45),
        force_run=_b("FORCE_RUN", "false"),
        account_equity=_f("ACCOUNT_EQUITY", 10000.0),
        risk_override=_f("RISK_PER_TRADE_OVERRIDE", 0.0),
        log_level=_s("LOG_LEVEL", "INFO").upper(),
        log_dir=_s("LOG_DIR", ".state_cache/logs"),
        healthcheck=_s("HEALTHCHECK_URL"),
        interval=_i("RUN_INTERVAL_SEC", 300),
    )
    if c.min_grade not in ("A", "A++", "A Super"):
        c.min_grade = "A"
    return c


# =============================================================================
# LOGGER
# =============================================================================
def get_logger(name: str = "agi", level: str = "INFO",
               log_dir: str = ".state_cache/logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(getattr(logging, level, logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    lg.addHandler(ch)
    fh = RotatingFileHandler(
        os.path.join(log_dir, "engine.log"),
        maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg


# =============================================================================
# UTILS
# =============================================================================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        out = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return out
    except Exception:
        return df


def load_json(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path: str, data: Any) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        pass


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df


# =============================================================================
# NEWS FILTER (static recurring high-impact windows)
# =============================================================================
STATIC_NEWS_WINDOWS = [
    (4, 12 * 60, 13 * 60 + 30, "NFP"),
    (2, 17 * 60 + 45, 18 * 60 + 45, "FOMC"),
]


def is_news_blocked(dt: datetime.datetime = None, block_min: int = 45) -> Tuple[bool, str]:
    if dt is None:
        dt = datetime.datetime.now(UTC)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)

    wd = dt.weekday()
    mins = dt.hour * 60 + dt.minute

    for w, s, e, label in STATIC_NEWS_WINDOWS:
        if wd == w and (s - block_min) <= mins <= (e + block_min):
            return True, label

    if 8 <= dt.day <= 20 and wd in (1, 2, 3) \
            and (12 * 60 - block_min) <= mins <= (13 * 60 + 30 + block_min):
        return True, "CPI/PPI (approx)"

    return False, ""
