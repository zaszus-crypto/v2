"""
=============================================================================
XAUUSD AGI v35.1 - PRODUCTION GRADE QUANT BOT
— scan, alert, report, backtest, tuner, CLI, zip builder
=============================================================================
"""
import os
import sys
import time
import json
import zipfile
import sqlite3
import datetime
import argparse
import itertools
from contextlib import closing
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd
import pytz

from config import (
    load_config, get_logger, STATE_DIR,
    is_news_blocked, resolve_grade_to_risk, grade_at_least,
)
from core import (
    detect_regime, detect_anomaly,
    wilder_atr_value, Memory, Calibrator,
    MetaLearner, EngineTracker, ENGINES,
    grade_signal, calculate_precise_entry,
    regime_engine_prior, AdaptiveThreshold,
)
from integrations import (
    get_data, debate, send_text, send_photo,
    esc, is_paused, tg_poll,
    PositionManager, build_snapshot,
    build_equity_chart,
)

WIB = pytz.timezone("Asia/Jakarta")
DB_FILE = os.path.join(STATE_DIR, "state.db")
TUNER_FILE = os.path.join(STATE_DIR, "tuned_params.json")

CFG = load_config()
log = get_logger("main", CFG.log_level, CFG.log_dir)


# =============================================================================
# DB & FAILOVER STATE
# =============================================================================
def init_db():
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_FILE, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, direction TEXT, price REAL, zone REAL,
                grade TEXT, entry REAL, sl REAL, tp1 REAL,
                source TEXT, status TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        log.error(f"Database initialization error: {e}")


def is_duplicate(direction: str, price: float, cooldown_min: int, tol: float = 2.5) -> bool:
    zone = round(price / 5.0) * 5.0
    now = time.time()
    try:
        with closing(sqlite3.connect(DB_FILE, timeout=15.0)) as c:
            rows = c.execute(
                "SELECT ts, direction, price, zone FROM signals ORDER BY id DESC LIMIT 10"
            ).fetchall()
        for ts, ld, lp, lz in rows:
            if (now - ts) / 60.0 < cooldown_min:
                if direction == ld and abs(zone - lz) < 1e-4:
                    return True
                if direction != ld and abs(price - lp) < tol:
                    return True
    except Exception as e:
        log.warning(f"Duplicate check warning: {e}")
    return False


def save_signal(direction: str, price: float, grade: str, entry: float, sl: float, tp1: float, source: str):
    try:
        with closing(sqlite3.connect(DB_FILE, timeout=15.0)) as c:
            c.execute(
                "INSERT INTO signals (ts, direction, price, zone, grade, entry, sl, tp1, source, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (time.time(), direction, float(price), round(price / 5.0) * 5.0,
                 grade, float(entry), float(sl), float(tp1), source, "ACTIVE"),
            )
            c.commit()
    except Exception as e:
        log.error(f"Failed to save signal: {e}")


# =============================================================================
# ALERT FORMATTING
# =============================================================================
def format_alert(signal, e, atr, consensus, grade, score, regime, anomaly, mem, dv, source, price, offset, risk_info) -> str:
    banner = {
        "A Super": "🚀 <b>GRADE A SUPER (PRIME)</b>\n💰 Risk 3.0% | Lot 3.0x",
        "A++":     "🔴 <b>GRADE A++ (HIGH CONVICTION)</b>\n💰 Risk 1.5% | Lot 1.5x",
        "A":       "🟢 <b>GRADE A (STANDARD)</b>\n💰 Risk 0.5% | Lot 0.5x",
    }.get(grade, "⚪ <b>UNGRADED</b>")

    emoji = "🟢" if signal == "BUY" else "🔴"
    bar = "█" * int(consensus / 10) + "░" * (10 - int(consensus / 10))
    off = f" off {offset:+.2f}" if abs(offset) > 0.01 else ""
    rp, ru, lm = risk_info

    return f"""{emoji} <b>XAUUSD {signal} EXECUTION</b>
━━━━━━━━━━━━━━━━━━━━━
{banner}
━━━━━━━━━━━━━━━━━━━━━
📊 Confluence: [{bar}] {consensus:.1f}% (Score {score:.1f})
🌊 Regime: {esc(regime.regime)} ({regime.confidence:.2f})
🎯 Precision: {e.precision_score:.1f}/100 ({esc(e.precision_grade)})
📡 Feed: {esc(source)}{off} | <code>{price:.2f}</code>
━━━━━━━━━━━━━━━━━━━━━
<b>Entry</b> ({esc(e.entry_type)}): <code>{e.entry_low:.2f}-{e.entry_high:.2f}</code>
<b>Ideal</b>: <code>{e.entry_ideal:.2f}</code>
<b>SL</b>: <code>{e.sl:.2f}</code>
<b>Risk</b>: <code>{e.risk_points:.2f}pts</code> = <b>{rp:.1f}%</b> (${ru:.2f})
━━━━━━━━━━━━━━━━━━━━━
🎯 TP1: <code>{e.tp1:.2f}</code> ({e.rr_tp1}R)
🎯 TP2: <code>{e.tp2:.2f}</code> ({e.rr_tp2}R)
🎯 TP3: <code>{e.tp3:.2f}</code> ({e.rr_tp3}R)
━━━━━━━━━━━━━━━━━━━━━
🧠 Memory: n={mem.get('n', 0)} WR={mem.get('winrate', 50):.0f}%
⚡ Anomaly: {anomaly.score:.2f}
🏛️ Council: {esc(dv.get('verdict', 'ABSTAIN'))} ({dv.get('confidence_mult', 1.0)}x)
⏰ <i>{datetime.datetime.now(WIB).strftime('%d %b %Y | %H:%M WIB')}</i>"""

# =============================================================================
# SCAN ENGINE WITH FAILOVER
# =============================================================================
def run_scan(force: bool = False) -> int:
    try:
        source, offset, price, df_m30, df_h1, df_h4 = get_data(
            CFG.symbol_mt5, CFG.symbol_deriv, CFG.yahoo_ticker,
            CFG.yahoo_auto, CFG.yahoo_fallback, CFG.feed_timeout,
        )
        log.info(f"Feed={source} off={offset:+.2f} p={price:.2f} M30={len(df_m30)} H1={len(df_h1)} H4={len(df_h4)}")
    except Exception as e:
        log.error(f"Feed failover triggered due to exception: {e}")
        return 1

    memory = Memory()
    calibrator = Calibrator()
    meta = MetaLearner()
    tracker = EngineTracker()
    pm = PositionManager()
    adaptive = AdaptiveThreshold(
        base_confluence=CFG.min_confluence,
        base_precision=CFG.min_precision,
    )

    for ev in pm.update_all(price):
        log.info(f"POS: {ev}")

    resolved = memory.resolve_open_trades(df_m30)
    if resolved:
        log.info(f"Memory trades resolved: {resolved}")

    if not (force or CFG.force_run):
        blocked, label = is_news_blocked(block_min=CFG.block_news_min)
        if blocked:
            log.info(f"News window active: {label}")
            return 0

    regime = detect_regime(df_m30)
    anomaly = detect_anomaly(df_m30)
    _, mem_stats = memory.query(df_m30, k=20)

    h1_trend = "BULLISH" if df_h1["close"].iloc[-1] > df_h1["close"].rolling(50, min_periods=10).mean().iloc[-1] else "BEARISH"
    h4_trend = "BULLISH" if df_h4["close"].iloc[-1] > df_h4["close"].rolling(50, min_periods=10).mean().iloc[-1] else "BEARISH"

    buy_w = sell_w = 0.0
    states: dict = {}
    for name, fn in ENGINES:
        try:
            sc, w = fn(df_m30)
            t_mult = tracker.get_weight_mult(regime.regime, name)
            r_prior = regime_engine_prior(regime.regime, name)
            eff = w * t_mult * r_prior
            states[name] = {"sc": sc, "weight": eff}
            if sc > 0:
                buy_w += eff * abs(sc)
            elif sc < 0:
                sell_w += eff * abs(sc)
        except Exception as e:
            log.debug(f"Engine {name}: {e}")
            states[name] = {"sc": 0, "weight": 1.0}

    if h1_trend == "BULLISH":
        buy_w += 2.5
    else:
        sell_w += 2.5
    if h4_trend == "BULLISH":
        buy_w += 1.2
    else:
        sell_w += 1.2

    if (buy_w + sell_w) < 0.01 or abs(buy_w - sell_w) < 0.01:
        log.info("Market equilibrium reached. Holding.")
        return 0

    consensus = (max(buy_w, sell_w) / (buy_w + sell_w)) * 100.0
    signal = "BUY" if buy_w > sell_w else "SELL"
    adj = consensus * meta.penalty(regime.regime)

    a_conf, a_prec = adaptive.adjusted_thresholds()
    if adj < a_conf and not (force or CFG.force_run):
        log.info(f"Confluence {adj:.1f}% < required {a_conf}%")
        return 0

    if CFG.require_mtf and not (force or CFG.force_run):
        ok = ((signal == "BUY" and h1_trend == "BULLISH") or (signal == "SELL" and h1_trend == "BEARISH"))
        if not ok:
            log.info(f"MTF alignment rejected {signal}/{h1_trend}")
            return 0

    if anomaly.is_anomaly and not (force or CFG.force_run):
        log.info(f"Anomaly detected score: {anomaly.score:.2f}")
        return 0

    atr_val = wilder_atr_value(df_m30, 14)
    g = grade_signal(adj, signal, states, h1_trend, h4_trend, atr_val, mem_stats, anomaly.score, regime.confidence)

    if not grade_at_least(g["grade"], CFG.min_grade) and not (force or CFG.force_run):
        log.info(f"Grade {g['grade']} below minimum {CFG.min_grade}")
        return 0

    e = calculate_precise_entry(df_m30, signal, price, h4_trend, h1_trend, adj, atr_val)

    if e.precision_score < a_prec and not (force or CFG.force_run):
        log.info(f"Precision score {e.precision_score:.1f} < {a_prec}")
        return 0

    if is_duplicate(signal, e.entry_ideal, CFG.cooldown_min) and not (force or CFG.force_run):
        log.info("Duplicate signal within cooldown window.")
        return 0

    dv = {"verdict": "AGREE", "confidence_mult": 1.0, "notes": "local"}
    if CFG.gemini_key:
        dv = debate(signal, price, adj, regime.regime, h1_trend, h4_trend, 50.0, atr_val, mem_stats, anomaly.score, CFG.gemini_key, CFG.gemini_model)
        if dv.get("verdict") == "DISAGREE" and not (force or CFG.force_run):
            log.info(f"Council consensus rejected: {dv.get('notes')}")
            return 0

    risk_info = resolve_grade_to_risk(g["grade"], CFG.risk_override, CFG.account_equity)

    memory.store(df_m30, regime, signal, e.entry_ideal, sl=e.sl, tp=e.tp1, extra={"grade": g["grade"], "engines": {k: v["sc"] for k, v in states.items()}})
    save_signal(signal, e.entry_ideal, g["grade"], e.entry_ideal, e.sl, e.tp1, source)
    pm.open_position(signal, e.entry_ideal, e.sl, e.tp1, e.tp2, e.tp3, atr_val, g["grade"], e.lot_multiplier)

    msg = format_alert(signal, e, atr_val, adj, g["grade"], g["score"], regime, anomaly, mem_stats, dv, source, price, offset, risk_info)

    if CFG.tg_token and CFG.tg_chats:
        chart = build_snapshot(df_m30, signal, e, regime, source, price)
        if chart:
            send_photo(CFG.tg_token, msg[:1000], chart, CFG.tg_chats)
        send_text(CFG.tg_token, msg, CFG.tg_chats)
        log.info("Telegram signal successfully dispatched.")
    else:
        log.info("No Telegram credentials found. Printing to stdout.")
        print(msg)

    return 0


# =============================================================================
# REPORT & POSITIONS
# =============================================================================
def build_report(with_chart: bool = False) -> str:
    pm = PositionManager()
    rep = pm.report()
    try:
        with closing(sqlite3.connect(DB_FILE)) as c:
            sig_n = c.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    except Exception:
        sig_n = 0

    if with_chart and CFG.tg_token and CFG.tg_chats:
        try:
            chart = build_equity_chart(pm)
            if chart:
                send_photo(CFG.tg_token, "📈 Equity Curve", chart, CFG.tg_chats)
        except Exception:
            pass

    return (f"📊 <b>PERFORMANCE REPORT</b>\n"
            f"Signals: <b>{sig_n}</b>\n"
            f"Positions total: <b>{rep['total']}</b> (active {rep['active']})\n"
            f"Winrate: <b>{rep['winrate']}%</b> ({rep['wins']}W/{rep['losses']}L)\n"
            f"Total R: <b>{rep['total_r']}</b>")


def positions_report() -> str:
    pm = PositionManager()
    act = pm.active()
    if not act:
        return "No active positions."
    return "\n".join([f"• {p.signal} @ {p.entry:.2f} SL {p.sl:.2f} | {p.grade}" for p in act])


# =============================================================================
# BACKTESTING & TUNER MODULES
# =============================================================================
@dataclass
class BacktestResult:
    total_signals: int = 0
    filled_trades: int = 0
    unfilled_orders: int = 0
    wins: int = 0
    losses: int = 0
    winrate: float = 0.0
    total_r: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    max_dd_r: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    expectancy: float = 0.0
    trades: List[dict] = field(default_factory=list)


SPREAD_POINTS = 0.30
SLIPPAGE_ATR = 0.05
COMMISSION_R = 0.02


def _simulate(df: pd.DataFrame, idx: int, sig: str, entry_ideal: float, sl: float, tp1: float, fill_wait: int = 8, hold: int = 24):
    n = len(df)
    fi = None
    fp = None
    for i in range(idx + 1, min(idx + 1 + fill_wait, n)):
        r = df.iloc[i]
        if sig == "BUY" and r["low"] <= entry_ideal:
            fp = entry_ideal if r["open"] >= entry_ideal else float(r["open"])
            fi = i
            break
        if sig == "SELL" and r["high"] >= entry_ideal:
            fp = entry_ideal if r["open"] <= entry_ideal else float(r["open"])
            fi = i
            break
    if fi is None:
        return None, None, 0.0, "UNFILLED"

    for i in range(fi, min(fi + 1 + hold, n)):
        r = df.iloc[i]
        if sig == "BUY":
            if r["low"] <= sl:
                return fp, fi, sl, "SL"
            if r["high"] >= tp1:
                return fp, fi, tp1, "TP1"
        else:
            if r["high"] >= sl:
                return fp, fi, sl, "SL"
            if r["low"] <= tp1:
                return fp, fi, tp1, "TP1"

    io = min(fi + hold, n - 1)
    return fp, fi, float(df.iloc[io]["close"]), "TIMEOUT"


def run_backtest(df_m30: pd.DataFrame, df_h1: Optional[pd.DataFrame] = None, df_h4: Optional[pd.DataFrame] = None,
                 min_conf: float = 65.0, min_prec: float = 70.0, min_grade: str = "A", warmup: int = 150,
                 cooldown: int = 4, require_mtf: bool = True) -> BacktestResult:
    r = BacktestResult()
    if df_m30 is None or len(df_m30) < warmup + 20:
        return r
    if df_h1 is None:
        df_h1 = df_m30
    if df_h4 is None:
        df_h4 = df_h1

    last = -1000
    i = warmup
    n = len(df_m30)

    while i < n - 10:
        if (i - last) < cooldown:
            i += 1
            continue

        win = df_m30.iloc[max(0, i - 200): i + 1]
        reg = detect_regime(win)

        if isinstance(df_m30.index, pd.DatetimeIndex):
            ct = df_m30.index[i]
            h1s = df_h1[df_h1.index < ct]
            h4s = df_h4[df_h4.index < ct]
        else:
            h1s = df_h1
            h4s = df_h4

        def _t(d):
            if d is None or len(d) < 30:
                return "NEUTRAL"
            c = d["close"].astype(float)
            s = c.rolling(50, min_periods=20).mean().iloc[-1]
            if pd.isna(s):
                return "NEUTRAL"
            p = float(c.iloc[-1])
            return "BULLISH" if p > s else ("BEARISH" if p < s else "SIDEWAYS")

        h1t, h4t = _t(h1s), _t(h4s)
        bw = sw = 0.0
        st: dict = {}
        for name, fn in ENGINES:
            try:
                sc, w = fn(win)
                st[name] = {"sc": sc, "weight": w}
                if sc > 0:
                    bw += w * abs(sc)
                elif sc < 0:
                    sw += w * abs(sc)
            except Exception:
                st[name] = {"sc": 0, "weight": 1.0}

        if "BULLISH" in h1t:
            bw += 2.5
        elif "BEARISH" in h1t:
            sw += 2.5
        if "BULLISH" in h4t:
            bw += 1.2
        elif "BEARISH" in h4t:
            sw += 1.2

        tw = bw + sw
        if tw == 0 or abs(bw - sw) < 0.01:
            i += 1
            continue

        c = (max(bw, sw) / tw) * 100.0
        if c < min_conf:
            i += 1
            continue

        sig = "BUY" if bw > sw else "SELL"
        if require_mtf:
            if not ((sig == "BUY" and "BULLISH" in h1t) or (sig == "SELL" and "BEARISH" in h1t)):
                i += 1
                continue

        atr = wilder_atr_value(win, 14)
        g = grade_signal(c, sig, st, h1t, h4t, atr, {"n": 0, "winrate": 50}, 0.0)
        if not grade_at_least(g["grade"], min_grade):
            i += 1
            continue

        base = float(win["close"].iloc[-1])
        e = calculate_precise_entry(win, sig, base, h4t, h1t, c, atr)
        if e.precision_score < min_prec:
            i += 1
            continue

        r.total_signals += 1
        fp, fi, ep, reason = _simulate(df_m30, i, sig, e.entry_ideal, e.sl, e.tp1)

        if fp is None:
            r.unfilled_orders += 1
            i += 1
            continue

        slip = atr * SLIPPAGE_ATR
        sh = SPREAD_POINTS / 2.0
        fpa = fp + sh + slip if sig == "BUY" else fp - sh - slip
        risk = abs(fpa - e.sl)
        if risk < 0.05:
            i += 1
            continue

        gr = ((ep - fpa) / risk) if sig == "BUY" else ((fpa - ep) / risk)
        pnl = gr - COMMISSION_R

        r.trades.append({
            "pnl_r": pnl, "grade": g["grade"],
            "regime": reg.regime, "exit_reason": reason,
        })
        last = fi
        i = fi + 1

    if r.trades:
        r.filled_trades = len(r.trades)
        rs = np.array([t["pnl_r"] for t in r.trades])
        r.wins = int((rs > 0).sum())
        r.losses = int((rs < 0).sum())
        r.winrate = round(r.wins / r.filled_trades * 100, 2)
        r.total_r = round(float(rs.sum()), 3)
        r.avg_r = round(float(rs.mean()), 4)
        r.expectancy = r.avg_r

        gw = float(rs[rs > 0].sum()) if (rs > 0).any() else 0.0
        gl = float(abs(rs[rs < 0].sum())) if (rs < 0).any() else 0.0
        r.profit_factor = round(gw / gl, 3) if gl > 0 else (999.0 if gw > 0 else 0.0)

        eq = np.cumsum(rs)
        peak = np.maximum.accumulate(eq)
        r.max_dd_r = round(float((peak - eq).max()), 3)

        std = rs.std()
        if std > 1e-9:
            r.sharpe = round(float(rs.mean() / std * np.sqrt(252)), 2)

        dn = rs[rs < 0]
        if len(dn) > 0 and dn.std() > 1e-9:
            r.sortino = round(float(rs.mean() / dn.std() * np.sqrt(252)), 2)

    return r


# =============================================================================
# AUTO TUNER & ZIP UTILS
# =============================================================================
def save_json_safe(path: str, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def build_zip():
    out = "xauusd-agi.zip"
    exclude_dirs = {".git", "__pycache__", ".state_cache", "node_modules", ".venv", "venv", ".pytest_cache"}
    exclude_ext = {".pyc", ".pyo", ".zip", ".png"}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for f in files:
                if os.path.splitext(f)[1] in exclude_ext:
                    continue
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, "."))
    print(f"✅ {out} ({os.path.getsize(out) / 1024:.1f} KB)")


# =============================================================================
# ORCHESTRATION MAIN CLI
# =============================================================================
def main_scan_once() -> int:
    log.info("=" * 60)
    log.info("XAUUSD AGI v35.1 SCAN (TF: M30 & H1)")
    log.info("=" * 60)
    init_db()

    try:
        tg_poll(
            CFG.tg_token, CFG.tg_users, CFG.tg_chats,
            force_scan_cb=lambda: run_scan(force=True),
            report_cb=lambda: build_report(with_chart=True),
            positions_cb=positions_report,
        )
    except Exception as e:
        log.debug(f"TG poll error: {e}")

    if is_paused() and not CFG.force_run:
        log.info("System currently paused by operator.")
        return 0

    rc = run_scan(force=CFG.force_run)

    if CFG.healthcheck:
        try:
            import requests
            requests.get(CFG.healthcheck, timeout=5)
        except Exception:
            pass
    return rc


def main():
    p = argparse.ArgumentParser(prog="xauusd-agi")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("scan")
    sub.add_parser("report")
    sub.add_parser("positions")
    sub.add_parser("zip")
    sub.add_parser("loop")

    bt = sub.add_parser("backtest")
    bt.add_argument("--days", type=int, default=60)

    tn = sub.add_parser("tune")
    tn.add_argument("--days", type=int, default=60)

    args = p.parse_args()

    if args.cmd == "zip":
        build_zip()
        return
    if args.cmd == "report":
        print(build_report().replace("<b>", "").replace("</b>", ""))
        return
    if args.cmd == "positions":
        print(positions_report())
        return
    if args.cmd == "backtest":
        _, _, _, m30, h1, h4 = get_data(CFG.symbol_mt5, CFG.symbol_deriv, CFG.yahoo_ticker)
        r = run_backtest(m30, h1, h4, CFG.min_confluence, CFG.min_precision, CFG.min_grade)
        print(f"Trades: {r.filled_trades} | WR {r.winrate}% | PF {r.profit_factor} | Exp {r.expectancy} | MaxDD {r.max_dd_r}R | Sharpe {r.sharpe}")
        return
    if args.cmd == "loop":
        log.info(f"Loop mode active. Interval: {CFG.interval}s")
        while True:
            try:
                main_scan_once()
            except KeyboardInterrupt:
                log.info("Interrupted by user.")
                break
            except Exception as e:
                log.error(f"Loop scan error: {e}")
            time.sleep(CFG.interval)
        return

    main_scan_once()


if __name__ == "__main__":
    sys.exit(main())
