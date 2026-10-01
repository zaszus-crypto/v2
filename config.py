import os
import logging
from dataclasses import dataclass

@dataclass
class Config:
    symbol_mt5: str = "XAUUSD"
    symbol_deriv: str = "GOLD"
    yahoo_ticker: str = "GC=F"
    yahoo_auto: bool = True
    yahoo_fallback: bool = True
    feed_timeout: int = 10
    interval: int = 1800  # 30 menit
    min_confluence: float = 65.0
    min_precision: float = 70.0
    min_grade: str = "A"
    cooldown_min: int = 30
    block_news_min: int = 30
    require_mtf: bool = True
    force_run: bool = False
    log_level: str = "INFO"
    log_dir: str = "logs"
    state_dir: str = ".state_cache"
    tg_token: str = os.getenv("TG_TOKEN", "")
    tg_chats: list = field_parse_chats(os.getenv("TG_CHATS", ""))
    tg_users: list = field_parse_users(os.getenv("TG_USERS", ""))
    gemini_key: str = os.getenv("GEMINI_KEY", "")
    gemini_model: str = "gemini-2.5-flash"
    healthcheck: str = os.getenv("HEALTHCHECK_URL", "")
    account_equity: float = float(os.getenv("ACCOUNT_EQUITY", "10000.0"))
    risk_override: float = float(os.getenv("RISK_OVERRIDE", "0.0"))

def field_parse_chats(val: str) -> list:
    if not val:
        return []
    return [c.strip() for c in val.split(",") if c.strip()]

def field_parse_users(val: str) -> list:
    if not val:
        return []
    return [int(u.strip()) for u in val.split(",") if u.strip().isdigit()]

STATE_DIR = ".state_cache"

def load_config() -> Config:
    os.makedirs(STATE_DIR, exist_ok=True)
    return Config()

def get_logger(name: str, level: str = "INFO", log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def is_news_blocked(block_min: int = 30) -> tuple:
    # Integrasi kalender ekonomi placeholder aman dari error
    return False, "Normal Market Conditions"

def resolve_grade_to_risk(grade: str, override: float, equity: float) -> tuple:
    if override > 0:
        pct = override
    else:
        mapping = {"A Super": 3.0, "A++": 1.5, "A": 0.5}
        pct = mapping.get(grade, 0.5)
    usd = equity * (pct / 100.0)
    mult = {"A Super": 3.0, "A++": 1.5, "A": 0.5}.get(grade, 1.0)
    return pct, usd, mult

def grade_at_least(actual: str, minimum: str) -> bool:
    hierarchy = {"A": 1, "A++": 2, "A Super": 3}
    return hierarchy.get(actual, 1) >= hierarchy.get(minimum, 1)
