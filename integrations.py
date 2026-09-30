"""
=============================================================================
INTEGRATIONS — price feed (MT5/Deriv/Yahoo/Stooq), Gemini, Telegram,
               position manager, chart builders
=============================================================================
"""
import os
import io
import json
import time
import html
import datetime
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, asdict, field
import requests
import websocket
import pandas as pd
import numpy as np

from config import (
    STATE_DIR, load_json, save_json, resample_ohlcv, get_logger,
)
from core import PrecisionEntry, RegimeState

log = get_logger("integrations")

OFFSET_FILE = os.path.join(STATE_DIR, "feed_offset.json")
POS_FILE = os.path.join(STATE_DIR, "positions.json")
BOT_STATE = os.path.join(STATE_DIR, "bot_state.json")
TG_OFFSET = os.path.join(STATE_DIR, "tg_offset.json")
TG_API = "https://api.telegram.org/bot{token}/{method}"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# =============================================================================
# PRICE FEED
# =============================================================================
def _load_offsets() -> Dict[str, float]:
    return load_json(OFFSET_FILE, {})


def _update_offset(key: str, value: float, alpha: float = 0.2):
    o = _load_offsets()
    prev = o.get(key, value)
    o[key] = round((1 - alpha) * prev + alpha * value, 4)
    save_json(OFFSET_FILE, o)


def _get_offset(key: str, default: float = 0.0) -> float:
    return _load_offsets().get(key, default)


def fetch_mt5(symbol: str, timeframe: str = "M15", count: int = 300):
    """Exact MT5 broker feed. Requires MetaTrader5 package + running terminal."""
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception:
        return None, None
    tf_map = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 16385, "H4": 16388, "D1": 16408,
    }
    try:
        if not mt5.initialize():
            return None, None
        rates = mt5.copy_rates_from_pos(symbol, tf_map.get(timeframe.upper(), 15), 0, count)
        if rates is None or len(rates) < 30:
            mt5.shutdown()
            return None, None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"}).set_index("time")
        df = df[["open", "high", "low", "close", "volume"]]
        tick = mt5.symbol_info_tick(symbol)
        price = float(tick.bid) if tick else float(df["close"].iloc[-1])
        mt5.shutdown()
        log.info(f"MT5 OK: {symbol} {timeframe} n={len(df)} p={price:.2f}")
        return df, price
    except Exception as e:
        log.debug(f"MT5 err: {e}")
        try:
            import MetaTrader5 as mt5  # type: ignore
            mt5.shutdown()
        except Exception:
            pass
        return None, None


def fetch_deriv(symbol: str = "frxXAUUSD", gran_sec: int = 900,
                count: int = 300, timeout: int = 8):
    url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    for _ in range(2):
        ws = None
        try:
            ws = websocket.create_connection(url, timeout=timeout)
            ws.send(json.dumps({
                "ticks_history": symbol, "count": count, "end": "latest",
                "granularity": gran_sec, "style": "candles",
            }))
            t0 = time.time()
            while time.time() - t0 < timeout:
                res = json.loads(ws.recv())
                if res.get("candles"):
                    df = pd.DataFrame(res["candles"])
                    for c in ("open", "high", "low", "close"):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df["volume"] = 100.0
                    df.index = pd.to_datetime(df["epoch"], unit="s", utc=True)
                    df = df[["open", "high", "low", "close", "volume"]].dropna()
                    return df, float(df["close"].iloc[-1])
        except Exception as e:
            log.debug(f"Deriv err: {e}")
        finally:
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass
        time.sleep(0.8)
    return None, None


def fetch_yahoo(ticker: str = "GC=F", period: str = "5d",
                interval: str = "15m", min_len: int = 40):
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).history(period=period, interval=interval,
                                        auto_adjust=False)
        if raw is None or len(raw) < min_len:
            return None, None
        price = float(raw["Close"].iloc[-1])
        raw = raw.reset_index()
        dc = next((c for c in raw.columns
                   if "date" in c.lower() or "datetime" in c.lower()),
                  raw.columns[0])
        raw = raw.rename(columns={dc: "time", "Close": "close",
                                   "High": "high", "Low": "low", "Open": "open"})
        vc = next((c for c in raw.columns if c.lower() == "volume"), None)
        raw["volume"] = (pd.to_numeric(raw[vc], errors="coerce").fillna(100.0)
                         if vc else 100.0)
        raw["time"] = pd.to_datetime(raw["time"], utc=True)
        raw = raw.dropna(subset=["close", "high", "low", "open"]).set_index("time")
        return raw[["open", "high", "low", "close", "volume"]].tail(300), price
    except Exception as e:
        log.debug(f"Yahoo err: {e}")
        return None, None


def fetch_stooq(symbol: str = "xauusd"):
    try:
        r = requests.get(f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=8)
        if r.status_code != 200 or len(r.text) < 100:
            return None, None
        df = pd.read_csv(io.StringIO(r.text))
        if "Date" not in df.columns or len(df) < 30:
            return None, None
        df = df.rename(columns={"Date": "time", "Open": "open", "High": "high",
                                 "Low": "low", "Close": "close",
                                 "Volume": "volume"})
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        return df[["open", "high", "low", "close", "volume"]].tail(300), \
               float(df["close"].iloc[-1])
    except Exception as e:
        log.debug(f"Stooq err: {e}")
        return None, None


def get_data(symbol_mt5: str = "XAUUSD", symbol_deriv: str = "frxXAUUSD",
             yahoo_ticker: str = "GC=F", yahoo_auto: bool = True,
             yahoo_fallback: float = -35.0, timeout: int = 8):
    # 1. MT5
    df_m15, p = fetch_mt5(symbol_mt5, "M15", 300)
    if df_m15 is not None and p is not None:
        df_h1, _ = fetch_mt5(symbol_mt5, "H1", 250)
        df_h4, _ = fetch_mt5(symbol_mt5, "H4", 200)
        if df_h1 is None:
            df_h1 = resample_ohlcv(df_m15, "1h")
        if df_h4 is None:
            df_h4 = resample_ohlcv(df_h1, "4h")
        return "MT5", 0.0, p, df_m15, df_h1, df_h4

    # 2. Deriv
    df_m15_d, p_d = fetch_deriv(symbol_deriv, 900, 300, timeout)
    df_h1_d, _ = (fetch_deriv(symbol_deriv, 3600, 250, timeout)
                  if df_m15_d is not None else (None, None))
    df_h4_d, _ = (fetch_deriv(symbol_deriv, 14400, 200, timeout)
                  if df_m15_d is not None else (None, None))

    # 3. Yahoo
    df_m15_y, p_y = fetch_yahoo(yahoo_ticker, "5d", "15m", 40)
    df_h1_y, _ = fetch_yahoo(yahoo_ticker, "60d", "1h", 50)
    df_h4_y, _ = fetch_yahoo(yahoo_ticker, "1y", "1d", 40)

    if df_m15_d is not None and p_d is not None:
        df_h1 = df_h1_d if df_h1_d is not None else resample_ohlcv(df_m15_d, "1h")
        df_h4 = df_h4_d if df_h4_d is not None else resample_ohlcv(df_h1, "4h")
        if yahoo_auto and df_m15_y is not None and p_y is not None:
            _update_offset("yahoo_offset", p_d - p_y)
        return "Deriv", 0.0, p_d, df_m15_d, df_h1, df_h4

    if df_m15_y is not None and p_y is not None:
        offset = _get_offset("yahoo_offset", yahoo_fallback) if yahoo_auto else yahoo_fallback
        price = p_y + offset
        df_m15 = df_m15_y.copy()
        for c in ("open", "high", "low", "close"):
            df_m15[c] = df_m15[c] + offset
        if df_h1_y is not None:
            df_h1 = df_h1_y.copy()
            for c in ("open", "high", "low", "close"):
                df_h1[c] = df_h1[c] + offset
        else:
            df_h1 = resample_ohlcv(df_m15, "1h")
        if df_h4_y is not None:
            df_h4 = df_h4_y.copy()
            for c in ("open", "high", "low", "close"):
                df_h4[c] = df_h4[c] + offset
        else:
            df_h4 = resample_ohlcv(df_h1, "4h")
        log.warning(f"Yahoo fallback offset={offset:+.2f}")
        return f"Yahoo({yahoo_ticker})", offset, price, df_m15, df_h1, df_h4

    df_d, p_s = fetch_stooq()
    if df_d is not None and p_s is not None:
        return "Stooq", 0.0, p_s, df_d, resample_ohlcv(df_d, "1h"), df_d

    raise RuntimeError("All price sources failed")


# =============================================================================
# GEMINI
# =============================================================================
def _gemini_call(prompt: str, api_key: str, model: str = "gemini-2.0-flash",
                 timeout: int = 20, json_mode: bool = False) -> Optional[str]:
    if not api_key:
        return None
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.15, "topP": 0.8, "maxOutputTokens": 500,
                **({"responseMimeType": "application/json"} if json_mode else {}),
            },
        }
        r = requests.post(
            GEMINI_URL.format(model=model),
            json=payload,
            headers={"Content-Type": "application/json",
                     "X-goog-api-key": api_key},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        cands = r.json().get("candidates") or []
        if not cands:
            return None
        parts = (cands[0].get("content") or {}).get("parts") or []
        return parts[0].get("text", "").strip() if parts else None
    except Exception:
        return None


def debate(signal: str, price: float, consensus: float, regime: str,
           h1_trend: str, h4_trend: str, rsi: float, atr: float,
           memory_stats: dict, anomaly_score: float,
           api_key: str, model: str = "gemini-2.0-flash") -> dict:
    default = {"verdict": "ABSTAIN", "confidence_mult": 1.0, "notes": "unavailable"}
    if not api_key:
        return {**default, "notes": "no_api_key"}

    prompt = f"""Institutional Gold Council. Return ONLY JSON:
{{"verdict":"AGREE"|"DISAGREE"|"ABSTAIN","confidence_mult":number 0.6-1.25,"bull_case":string,"bear_case":string,"notes":string}}

Data: {signal} @ {price:.2f} | Conf {consensus:.1f}% | Regime {regime}
H1={h1_trend} H4={h4_trend} RSI={rsi:.1f} ATR={atr:.2f}
Memory n={memory_stats.get('n', 0)} WR={memory_stats.get('winrate', 50)}%
Anomaly={anomaly_score:.2f}"""

    raw = _gemini_call(prompt, api_key, model, json_mode=True)
    if not raw:
        return default
    try:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0) if m else raw)
        v = str(obj.get("verdict", "ABSTAIN")).upper()
        if v not in ("AGREE", "DISAGREE", "ABSTAIN"):
            v = "ABSTAIN"
        cm = float(obj.get("confidence_mult", 1.0))
        return {
            "verdict": v,
            "confidence_mult": float(max(0.6, min(1.25, cm))),
            "bull_case": str(obj.get("bull_case", "")),
            "bear_case": str(obj.get("bear_case", "")),
            "notes": str(obj.get("notes", "")),
            "raw": raw,
        }
    except Exception:
        return {**default, "raw": raw}


# =============================================================================
# TELEGRAM
# =============================================================================
def esc(s: str) -> str:
    return html.escape(str(s), quote=False)


def send_text(token: str, text: str, chat_ids: List[str],
              silent: bool = False) -> bool:
    if not token or not chat_ids:
        return False
    if len(text) > 4000:
        text = text[:3990] + "..."
    ok = True
    for cid in chat_ids:
        s = silent or cid.startswith("silent:")
        actual = cid[7:] if cid.startswith("silent:") else cid
        sent = False
        for _ in range(3):
            try:
                r = requests.post(
                    TG_API.format(token=token, method="sendMessage"),
                    json={
                        "chat_id": actual, "text": text, "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                        "disable_notification": s,
                    },
                    timeout=12,
                )
                if r.status_code == 200:
                    sent = True
                    break
                if r.status_code == 429:
                    time.sleep(int(r.headers.get("Retry-After", 2)))
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(1.0)
        if not sent:
            ok = False
        time.sleep(0.05)
    return ok


def send_photo(token: str, caption: str, photo_path: str,
               chat_ids: List[str]) -> bool:
    if not token or not chat_ids or not os.path.exists(photo_path):
        return False
    if len(caption) > 1024:
        caption = caption[:1020] + "..."
    ok = True
    for cid in chat_ids:
        actual = cid.replace("silent:", "")
        try:
            with open(photo_path, "rb") as f:
                r = requests.post(
                    TG_API.format(token=token, method="sendPhoto"),
                    data={"chat_id": actual, "caption": caption,
                          "parse_mode": "HTML"},
                    files={"photo": f}, timeout=25,
                )
                if r.status_code != 200:
                    ok = False
        except Exception:
            ok = False
        time.sleep(0.1)
    return ok


# =============================================================================
# BOT STATE
# =============================================================================
def is_paused() -> bool:
    d = load_json(BOT_STATE, {"paused": False, "paused_until": None})
    if not d.get("paused"):
        return False
    u = d.get("paused_until")
    if u and datetime.datetime.now(datetime.timezone.utc).timestamp() > u:
        save_json(BOT_STATE, {"paused": False, "paused_until": None})
        return False
    return True


def bot_pause(hours: Optional[float] = None):
    until = None
    if hours:
        until = (datetime.datetime.now(datetime.timezone.utc)
                 + datetime.timedelta(hours=hours)).timestamp()
    save_json(BOT_STATE, {"paused": True, "paused_until": until})


def bot_resume():
    save_json(BOT_STATE, {"paused": False, "paused_until": None})


def tg_poll(token: str, allowed_users: List[str], chat_ids: List[str],
            force_scan_cb=None, report_cb=None, positions_cb=None):
    """Poll Telegram bot updates and respond to commands."""
    if not token:
        return
    try:
        offset = int(load_json(TG_OFFSET, {}).get("offset", 0))
        r = requests.get(
            TG_API.format(token=token, method="getUpdates"),
            params={"offset": offset + 1, "timeout": 3}, timeout=10,
        )
        if r.status_code != 200:
            return
        for upd in r.json().get("result", []):
            save_json(TG_OFFSET, {"offset": upd["update_id"]})
            msg = upd.get("message") or {}
            txt = (msg.get("text") or "").strip().lower()
            cid = str(msg.get("chat", {}).get("id", ""))
            uid = str(msg.get("from", {}).get("id", ""))
            if allowed_users and uid not in allowed_users:
                send_text(token, "🚫 Not authorized.", [cid])
                continue
            if txt.startswith(("/start", "/help")):
                send_text(token,
                          "🤖 <b>XAUUSD AGI Bot</b>\n"
                          "/status /pause [h] /resume /signal /report /equity /positions",
                          [cid])
            elif txt.startswith("/status"):
                st = "⏸ PAUSED" if is_paused() else "▶️ ACTIVE"
                send_text(token, f"Status: <b>{st}</b>", [cid])
            elif txt.startswith("/pause"):
                parts = txt.split()
                hrs = float(parts[1]) if len(parts) > 1 else None
                bot_pause(hrs)
                send_text(token, f"⏸ Paused{' ' + str(hrs) + 'h' if hrs else ''}.", [cid])
            elif txt.startswith("/resume"):
                bot_resume()
                send_text(token, "▶️ Resumed.", [cid])
            elif txt.startswith("/signal"):
                send_text(token, "🔄 Forcing scan...", [cid])
                if force_scan_cb:
                    force_scan_cb()
            elif txt.startswith("/report"):
                send_text(token, report_cb() if report_cb else "n/a", [cid])
            elif txt.startswith("/equity"):
                if report_cb:
                    report_cb()
                send_text(token, "📈 Chart sent (if available).", [cid])
            elif txt.startswith("/positions"):
                send_text(token, positions_cb() if positions_cb else "n/a", [cid])
    except Exception:
        pass


# =============================================================================
# POSITION MANAGER
# =============================================================================
@dataclass
class Position:
    id: str
    opened_ts: float
    signal: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    atr: float
    grade: str
    lot_mult: float
    be_moved: bool = False
    trail_active: bool = False
    partial_done: bool = False
    closed: bool = False
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_r: float = 0.0
    notes: List[str] = field(default_factory=list)


class PositionManager:
    def __init__(self, path: str = POS_FILE):
        self.path = path
        self.positions: Dict[str, Position] = {}
        raw = load_json(path, {})
        for k, v in raw.items():
            v.setdefault("notes", [])
            try:
                self.positions[k] = Position(**v)
            except Exception:
                pass

    def _save(self):
        save_json(self.path, {k: asdict(v) for k, v in self.positions.items()})

    def open_position(self, sig: str, entry: float, sl: float, tp1: float,
                      tp2: float, tp3: float, atr: float,
                      grade: str, lot_mult: float) -> Position:
        pid = f"{int(time.time())}-{sig}"
        p = Position(pid, time.time(), sig, entry, sl, tp1, tp2, tp3,
                     atr, grade, lot_mult)
        self.positions[pid] = p
        self._save()
        return p

    def update_all(self, price: float) -> List[Dict]:
        events: List[Dict] = []
        for pid, p in self.positions.items():
            if p.closed:
                continue
            risk = abs(p.entry - p.sl)
            if risk < 0.01:
                continue
            r_now = ((price - p.entry) / risk) if p.signal == "BUY" \
                else ((p.entry - price) / risk)

            if not p.partial_done and r_now >= 1.0:
                p.partial_done = True
                events.append({"id": pid, "action": "PARTIAL_TP", "r": round(r_now, 2)})
            if not p.be_moved and r_now >= 1.0:
                p.sl = p.entry
                p.be_moved = True
                events.append({"id": pid, "action": "MOVE_BE"})
            if not p.trail_active and r_now >= 1.5:
                p.trail_active = True
                events.append({"id": pid, "action": "TRAIL_ON"})

            if p.trail_active:
                td = p.atr * 1.2
                ns = price - td if p.signal == "BUY" else price + td
                if p.signal == "BUY" and ns > p.sl:
                    p.sl = ns
                elif p.signal == "SELL" and ns < p.sl:
                    p.sl = ns

            hit_sl = ((p.signal == "BUY" and price <= p.sl)
                      or (p.signal == "SELL" and price >= p.sl))
            hit_tp = ((p.signal == "BUY" and price >= p.tp3)
                      or (p.signal == "SELL" and price <= p.tp3))

            if hit_sl:
                p.closed = True
                p.exit_price = p.sl
                p.exit_reason = "SL/BE/TRAIL"
                p.pnl_r = round(
                    ((p.sl - p.entry) / risk) if p.signal == "BUY"
                    else ((p.entry - p.sl) / risk), 3)
                events.append({"id": pid, "action": "CLOSED",
                               "r": p.pnl_r, "reason": p.exit_reason})
            elif hit_tp:
                p.closed = True
                p.exit_price = p.tp3
                p.exit_reason = "TP3"
                p.pnl_r = round(
                    ((p.tp3 - p.entry) / risk) if p.signal == "BUY"
                    else ((p.entry - p.tp3) / risk), 3)
                events.append({"id": pid, "action": "CLOSED",
                               "r": p.pnl_r, "reason": p.exit_reason})
        self._save()
        return events

    def active(self) -> List[Position]:
        return [p for p in self.positions.values() if not p.closed]

    def report(self) -> dict:
        closed = [p for p in self.positions.values() if p.closed]
        wins = [p for p in closed if p.pnl_r > 0]
        return {
            "total": len(self.positions),
            "active": len(self.active()),
            "closed": len(closed),
            "wins": len(wins),
            "losses": len(closed) - len(wins),
            "winrate": round((len(wins) / len(closed) * 100), 1) if closed else 0.0,
            "total_r": round(sum(p.pnl_r for p in closed), 3),
        }


# =============================================================================
# CHART BUILDERS
# =============================================================================
def build_snapshot(df: pd.DataFrame, signal: str, entry: PrecisionEntry,
                   regime: RegimeState, source: str, price: float) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        snap = df.tail(80).copy()
        if len(snap) < 10:
            return None

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")

        o = snap["open"].values
        h = snap["high"].values
        l = snap["low"].values
        c = snap["close"].values

        for i in range(len(snap)):
            color = "#22c55e" if c[i] >= o[i] else "#ef4444"
            ax.plot([i, i], [l[i], h[i]], color=color, linewidth=0.8, alpha=0.85)
            ax.plot([i, i], [o[i], c[i]], color=color, linewidth=2.0)

        ax.axhline(entry.entry_ideal, color="#3b82f6", linestyle="--",
                   linewidth=1.2, label=f"Entry {entry.entry_ideal:.2f}")
        ax.axhline(entry.sl, color="#ef4444", linewidth=1.2,
                   label=f"SL {entry.sl:.2f}")
        ax.axhline(entry.tp1, color="#22c55e", linewidth=1.0,
                   label=f"TP1 {entry.tp1:.2f}")
        ax.axhline(entry.tp2, color="#22c55e", linestyle=":", linewidth=1.0,
                   alpha=0.7, label=f"TP2 {entry.tp2:.2f}")
        ax.axhline(entry.tp3, color="#22c55e", linestyle=":", linewidth=1.0,
                   alpha=0.5, label=f"TP3 {entry.tp3:.2f}")

        ax.set_title(
            f"XAUUSD {signal} | {regime.regime.value} | "
            f"Grade {entry.precision_grade} | {source}",
            color="#e2e8f0", fontsize=11, fontweight="bold",
        )
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_color("#334155")
        ax.grid(alpha=0.15, color="#475569")
        ax.legend(loc="best", fontsize=8, facecolor="#1e293b",
                  edgecolor="#334155", labelcolor="#e2e8f0")

        plt.tight_layout()
        out = os.path.join(STATE_DIR, f"snapshot_{int(time.time())}.png")
        plt.savefig(out, facecolor="#0f172a")
        plt.close(fig)
        return out
    except Exception:
        return None


def build_equity_chart(pm: PositionManager) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        closed = [p for p in pm.positions.values() if p.closed]
        if len(closed) < 2:
            return None
        closed.sort(key=lambda p: p.opened_ts)
        rs = np.array([p.pnl_r for p in closed])
        eq = np.cumsum(rs)

        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        ax.plot(eq, color="#3b82f6", linewidth=2)
        ax.fill_between(range(len(eq)), 0, eq, color="#3b82f6", alpha=0.15)
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.set_title(f"Equity — {len(closed)} closed | Total {eq[-1]:.2f}R",
                     color="#e2e8f0", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_color("#334155")
        ax.grid(alpha=0.15, color="#475569")

        plt.tight_layout()
        out = os.path.join(STATE_DIR, f"equity_{int(time.time())}.png")
        plt.savefig(out, facecolor="#0f172a")
        plt.close(fig)
        return out
    except Exception:
        return None
