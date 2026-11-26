import os
import json
import datetime as dt
from pathlib import Path

import numpy as np
import requests
import pandas as pd
import yfinance as yf

# ===== Fixed settings =====
SYMBOLS = ["XAUUSD=X", "GC=F"]   # spot gold first, then gold futures
BAR_MINUTES = {"5m": 5, "15m": 15}
SPREAD = 0.30                     # gold spread in $ (used in self-test)
ACCT = 5000                       # account size in $
RISK_PCT = 0.5                    # risk per trade in %
CONTRACT = 100                    # oz per lot for gold
START_HOUR, END_HOUR = 7, 20      # UTC session, Mon-Fri only
MIN_TRADES = 20                   # self-test needs this many trades to judge
PAUSE_BELOW = -1.0                # pause signals if 59-day net R falls to this
RESUME_ABOVE = 1.0                # resume once net R recovers to this

# ===== Strategy settings (override them by creating config.json) =====
DEFAULTS = {
    "interval": "5m",
    "swing": 36,        # recent swing liquidity lookback (bars)
    "wick": 0.40,       # rejection wick share of candle
    "min_range": 0.6,   # ignore candles smaller than this x ATR
    "sl_buf": 0.2,      # stop beyond wick (x ATR)
    "min_sl": 0.8,      # minimum stop (x ATR)
    "max_sl": 3.0,      # skip trade if stop wider than this (x ATR)
    "rr": 1.5,          # take profit = stop x rr
    "trend": True,      # only trade with the 50/200 EMA trend
    "mode": "sweep",    # "sweep" = liquidity sweep, "meanrev" = Bollinger + RSI stretch
    "bb_k": 2.0,        # meanrev: Bollinger deviation
    "rsi": 30,          # meanrev: RSI oversold level (overbought = 100 - this)
    "sl_atr": 2.0,      # meanrev: stop distance in ATR
}
COOLDOWN_BARS = 6       # no new alert within this many bars of the last one
RR_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

TOKEN = os.environ.get("TG_TOKEN")
CHAT = os.environ.get("TG_CHAT")
STATE = Path("state.json")


def send(text):
    print(text)
    if TOKEN and CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": CHAT, "text": text[:4000]},
                timeout=20,
            )
        except Exception as e:
            print("telegram error", e)


def load_cfg():
    p = dict(DEFAULTS)
    f = Path("config.json")
    if f.exists():
        try:
            p.update(json.loads(f.read_text()))
        except Exception as e:
            print("config error", e)
    if p["interval"] not in BAR_MINUTES:
        p["interval"] = "5m"
    return p


def load_state():
    if STATE.exists():
        try:
            s = json.loads(STATE.read_text())
            return {"last_bar": s.get("last_bar", ""), "paused": s.get("paused", False), "hb": s.get("hb", "")}
        except Exception:
            pass
    return {"last_bar": "", "paused": False, "hb": ""}


def save_state(state):
    STATE.write_text(json.dumps(state))


def get_data(interval):
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, interval=interval, period="59d", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) > 400:
                idx = df.index
                df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
                return sym, df
        except Exception as e:
            print("data error", sym, e)
    return None, None


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def add_base(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = rma(tr, 14)

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df["rsi"] = 100 - 100 / (1 + rma(gain, 14) / rma(loss, 14))
    df["bb_mid"] = c.rolling(20).mean()
    df["bb_sd"] = c.rolling(20).std(ddof=0)

    df["day"] = df.index.normalize()
    asia = df[df.index.hour < 7]
    df["asia_hi"] = df["day"].map(asia.groupby("day")["High"].max())
    df["asia_lo"] = df["day"].map(asia.groupby("day")["Low"].min())
    early = df.index.hour < 7
    df.loc[early, ["asia_hi", "asia_lo"]] = np.nan       # only valid after Asia ends
    d_hi = df.groupby("day")["High"].max().shift(1)
    d_lo = df.groupby("day")["Low"].min().shift(1)
    df["pdh"] = df["day"].map(d_hi)                      # previous day high
    df["pdl"] = df["day"].map(d_lo)                      # previous day low
    return df


def add_swing(df, swing):
    df["ll"] = df["Low"].shift(1).rolling(swing).min()   # recent swing low
    df["hh"] = df["High"].shift(1).rolling(swing).max()  # recent swing high
    return df


def add_signals(df, p):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    atr = df["atr"]
    if p["mode"] == "meanrev":
        lower = df["bb_mid"] - p["bb_k"] * df["bb_sd"]
        upper = df["bb_mid"] + p["bb_k"] * df["bb_sd"]
        if p["trend"]:
            up_ok = df["ema50"] > df["ema200"]
            dn_ok = df["ema50"] < df["ema200"]
        else:
            up_ok = pd.Series(True, index=df.index)
            dn_ok = pd.Series(True, index=df.index)
        buy = (c < lower) & (df["rsi"] < p["rsi"]) & up_ok
        sell = (c > upper) & (df["rsi"] > 100 - p["rsi"]) & dn_ok
        df["buy"] = buy
        df["sell"] = sell
        df["sl_dist"] = np.where(buy | sell, p["sl_atr"] * atr, np.nan)
        return df
    rng = h - l
    body_lo = np.minimum(o, c)
    body_hi = np.maximum(o, c)
    big = rng >= p["min_range"] * atr

    sweep_lo = pd.Series(False, index=df.index)
    sweep_hi = pd.Series(False, index=df.index)
    for col in ["ll", "asia_lo", "pdl"]:
        sweep_lo = sweep_lo | ((l < df[col]) & (c > df[col]))
    for col in ["hh", "asia_hi", "pdh"]:
        sweep_hi = sweep_hi | ((h > df[col]) & (c < df[col]))

    if p["trend"]:
        up_ok = df["ema50"] > df["ema200"]
        dn_ok = df["ema50"] < df["ema200"]
    else:
        up_ok = pd.Series(True, index=df.index)
        dn_ok = pd.Series(True, index=df.index)

    buy = sweep_lo & (c > o) & big & ((body_lo - l) >= p["wick"] * rng) & up_ok
    sell = sweep_hi & (c < o) & big & ((h - body_hi) >= p["wick"] * rng) & dn_ok

    d_buy = np.maximum(c - (l - p["sl_buf"] * atr), p["min_sl"] * atr)
    d_sell = np.maximum((h + p["sl_buf"] * atr) - c, p["min_sl"] * atr)
    buy = buy & (d_buy <= p["max_sl"] * atr)
    sell = sell & (d_sell <= p["max_sl"] * atr)

    df["buy"] = buy
    df["sell"] = sell
    df["sl_dist"] = np.where(buy, d_buy, np.where(sell, d_sell, np.nan))
    return df


def swept_name(row, side):
    if side == "BUY":
        names = [("ll", "recent swing low"), ("asia_lo", "Asia low"), ("pdl", "previous day low")]
        for col, name in names:
            if row["Low"] < row[col] and row["Close"] > row[col]:
                return name
    else:
        names = [("hh", "recent swing high"), ("asia_hi", "Asia high"), ("pdh", "previous day high")]
        for col, name in names:
            if row["High"] > row[col] and row["Close"] < row[col]:
                return name
    return "liquidity level"


def backtest(df, rr, start=210):
    hi, lo, cl = df["High"].values, df["Low"].values, df["Close"].values
    buy, sell, dist = df["buy"].values, df["sell"].values, df["sl_dist"].values
    n = len(df)
    i = start
    trades = wins = 0
    net = 0.0
    while i < n - 1:
        if buy[i] or sell[i]:
            d = dist[i]
            side = 1 if buy[i] else -1
            sl = cl[i] - side * d
            tp = cl[i] + side * d * rr
            res = None
            j = i + 1
            while j < n:
                hit_sl = lo[j] <= sl if side == 1 else hi[j] >= sl
                hit_tp = hi[j] >= tp if side == 1 else lo[j] <= tp
                if hit_sl:
                    res = -1.0
                    break
                if hit_tp:
                    res = rr
                    break
                j += 1
            if res is None:
                break
            trades += 1
            wins += 1 if res > 0 else 0
            net += res - SPREAD / d
            i = j + 1
        else:
            i += 1
    return trades, wins, net


def stats_text(trades, wins, net):
    if trades == 0:
        return "Self-test (last ~59 days): no trades yet"
    return f"Self-test (last ~59 days): {trades} trades, {wins * 100 / trades:.0f}% win, net {net:+.1f}R after spread"


def optimize_report(now):
    """Tries many settings (two strategy types) on the first 70% of the data, then checks them on the last 30%."""
    messages = []
    for interval in ["5m", "15m"]:
        sym, df = get_data(interval)
        if df is None:
            messages.append(f"Optimizer {interval}: price data failed to load.")
            continue
        bar_min = BAR_MINUTES[interval]
        base = add_base(df)
        closed = base[base.index + pd.Timedelta(minutes=bar_min) <= now]
        split = int(len(closed) * 0.7)
        cands = []

        def consider(sig, cfg):
            tr, va = sig.iloc[:split], sig.iloc[split:]
            for rr in RR_GRID:
                t1, w1, n1 = backtest(tr, rr, 210)
                t2, w2, n2 = backtest(va, rr, 0)
                if t1 >= 25 and t2 >= 10 and n1 > 0 and n2 > 0:
                    cands.append({"t1": t1, "w1": w1, "n1": n1, "t2": t2, "w2": w2, "n2": n2,
                                  "net": n1 + n2, "win": (w1 + w2) / (t1 + t2),
                                  "cfg": dict(cfg, rr=rr)})

        for swing in [24, 36, 72]:
            b = add_swing(closed.copy(), swing)
            for wick in [0.3, 0.4]:
                for min_range in [0.6, 1.0]:
                    for trend in [True, False]:
                        for min_sl in [0.8, 1.5, 2.5]:
                            cfg = {"interval": interval, "mode": "sweep", "swing": swing, "wick": wick,
                                   "min_range": min_range, "trend": trend, "min_sl": min_sl, "max_sl": 4.0}
                            sig = add_signals(b.copy(), dict(DEFAULTS, **cfg))
                            consider(sig, cfg)

        for bb_k in [2.0, 2.5]:
            for rsi in [25, 30]:
                for sl_atr in [1.5, 2.5, 3.5]:
                    for trend in [True, False]:
                        cfg = {"interval": interval, "mode": "meanrev", "bb_k": bb_k, "rsi": rsi,
                               "sl_atr": sl_atr, "trend": trend}
                        sig = add_signals(closed.copy(), dict(DEFAULTS, **cfg))
                        consider(sig, cfg)

        if not cands:
            messages.append(f"Optimizer {interval} ({sym}): NO setting was profitable in both the training "
                            f"and the test period. Do not trade {interval} gold with these strategies.")
            continue

        hi = sorted([c for c in cands if c["win"] >= 0.50], key=lambda c: c["net"], reverse=True)[:2]
        best = sorted(cands, key=lambda c: c["net"], reverse=True)[:1]
        picks = [("HIGH WIN RATE", c) for c in hi]
        if best and best[0] not in hi:
            picks.append(("BEST PROFIT", best[0]))
        if not hi:
            picks.insert(0, ("NO setting with 50%+ win rate was profitable", None))

        lines = [f"Optimizer {interval} ({sym})"]
        for label, c in picks:
            if c is None:
                lines.append(label)
                continue
            lines.append(
                f"{label} [{c['cfg']['mode']}]\n"
                f"Train: {c['t1']} trades, {c['w1'] * 100 / c['t1']:.0f}% win, {c['n1']:+.1f}R\n"
                f"Test: {c['t2']} trades, {c['w2'] * 100 / c['t2']:.0f}% win, {c['n2']:+.1f}R\n"
                f"TO USE IT: create a file named config.json in your GitHub repo and paste this one line:\n{json.dumps(c['cfg'])}"
            )
        messages.append("\n\n".join(lines))
    messages.append("Note: high win-rate setups use small targets, so one loss can erase several wins. "
                    "Judge by net R, and demo-test any setting for weeks before using it on a challenge.")
    return messages


def ist_time(ts):
    return (ts + pd.Timedelta(hours=5, minutes=30)).strftime("%I:%M %p")


def main():
    now = dt.datetime.now(dt.timezone.utc)
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

    p = load_cfg()
    interval = p["interval"]
    bar_min = BAR_MINUTES[interval]

    state = load_state()
    if not STATE.exists():
        save_state(state)

    if not manual and not (START_HOUR <= now.hour < END_HOUR and now.weekday() < 5):
        return

    sym, df = get_data(interval)
    if df is None:
        if manual:
            send("Warning: the bot is connected but could not load price data. Try again in a few minutes.")
        return

    df = add_signals(add_swing(add_base(df), p["swing"]), p)
    closed = df[df.index + pd.Timedelta(minutes=bar_min) <= now]
    if len(closed) < 300:
        return

    trades, wins, net = backtest(closed, p["rr"])
    summary = stats_text(trades, wins, net)

    if trades >= MIN_TRADES:
        if not state["paused"] and net <= PAUSE_BELOW:
            state["paused"] = True
            save_state(state)
            send("PAUSED. The strategy is losing on recent gold data, so the bot will stay quiet on purpose. "
                 "Do NOT trade until you get a BUY or SELL message.\n" + summary)
        elif state["paused"] and net >= RESUME_ABOVE:
            state["paused"] = False
            save_state(state)
            send("RESUMED. The strategy recovered, so BUY and SELL messages are back on.\n" + summary)

    today = now.strftime("%Y-%m-%d")
    if manual or state.get("hb") != today:
        if state["paused"]:
            status = ("PAUSED - the strategy is losing right now, so NO trades today unless it recovers. "
                      "Staying quiet is on purpose.")
        else:
            status = "RUNNING - watching gold. You will get a message only when there is a BUY or SELL."
        send(f"Daily check ({today}): the bot is alive.\nStatus: {status}\n"
             f"Chart: gold {interval}, data from {sym}\n{summary}")
        state["hb"] = today
        save_state(state)

    if manual:
        for m in optimize_report(now):
            send(m)

    if state["paused"]:
        return

    bar = closed.iloc[-1]
    t = closed.index[-1]
    close_time = t + pd.Timedelta(minutes=bar_min)
    age = (now - close_time).total_seconds() / 60
    if age > 2 * bar_min:
        return  # signal is too old to act on
    if not (bar["buy"] or bar["sell"]):
        return
    if state["last_bar"]:
        try:
            if (t - pd.Timestamp(state["last_bar"])).total_seconds() < COOLDOWN_BARS * bar_min * 60:
                return  # too soon after the last alert
        except Exception:
            pass

    buy = bool(bar["buy"])
    action = "BUY" if buy else "SELL"
    icon = "GREEN" if buy else "RED"
    d = float(bar["sl_dist"])
    tp_d = d * p["rr"]
    lots = int((ACCT * RISK_PCT / 100) / (d * CONTRACT) * 100) / 100
    lots_txt = f"{lots:.2f}" if lots >= 0.01 else "0.01 (minimum - this trade risks a bit more than usual)"
    sl_side, tp_side = ("below", "above") if buy else ("above", "below")
    ex = 3500.00
    ex_sl = ex - d if buy else ex + d
    ex_tp = ex + tp_d if buy else ex - tp_d
    risk_usd = ACCT * RISK_PCT / 100

    msg = (
        f"{icon} SIGNAL: {action} GOLD NOW\n"
        f"Time: {ist_time(close_time)} IST ({close_time.strftime('%H:%M')} UTC). "
        f"Valid for about {2 * bar_min} minutes only.\n\n"
        f"DO THIS IN MT5:\n"
        f"1) Open XAUUSD, tap New Order, choose {action}\n"
        f"2) Lots: {lots_txt}\n"
        f"3) Stop Loss: ${d:.2f} {sl_side} your {action.lower()} price\n"
        f"4) Take Profit: ${tp_d:.2f} {tp_side} your {action.lower()} price\n"
        f"5) Tap {action}\n\n"
        f"Example only: if you {action.lower()} at {ex:,.2f}, Stop Loss = {ex_sl:,.2f} and Take Profit = {ex_tp:,.2f}\n"
        f"Money at risk: about ${risk_usd:.0f} ({RISK_PCT}% of ${ACCT:,})\n\n"
        f"SKIP THIS TRADE if more than {2 * bar_min} minutes have passed, or gold already moved ${d / 3:.2f} or more.\n"
        f"One trade at a time. Never remove or move the Stop Loss."
    )
    send(msg)
    state["last_bar"] = str(t)
    save_state(state)


if __name__ == "__main__":
    main()
