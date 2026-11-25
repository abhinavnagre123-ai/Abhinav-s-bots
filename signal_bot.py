import os
import json
import datetime as dt
from pathlib import Path

import numpy as np
import requests
import pandas as pd
import yfinance as yf

# ===== Settings =====
SYMBOLS = ["XAUUSD=X", "GC=F"]   # tries spot gold first, then gold futures
INTERVAL = "5m"                   # "5m" = most trades, "15m" = fewer
BAR_MIN = 5                       # must match INTERVAL (5 or 15)
SWING_BARS = 36                   # recent swing high/low liquidity lookback
WICK_RATIO = 0.40                 # rejection wick must be this share of the candle
MIN_RANGE_ATR = 0.6               # ignore tiny candles
SL_BUFFER_ATR = 0.2               # stop sits this far beyond the sweep wick
MIN_SL_ATR = 0.8                  # stop never tighter than this
MAX_SL_ATR = 3.0                  # skip trade if stop would be wider than this
RR = 1.5                          # take profit = stop x this (lower = higher win rate)
SPREAD = 0.30                     # gold spread in $ (used in self-test)
ACCT = 5000                       # account size in $
RISK_PCT = 0.5                    # risk per trade in %
CONTRACT = 100                    # oz per lot for gold
START_HOUR, END_HOUR = 7, 20      # UTC session, Mon-Fri only
MIN_TRADES = 20                   # self-test needs this many trades to judge
PAUSE_BELOW = -1.0                # pause signals if 60-day net R falls to this
RESUME_ABOVE = 1.0                # resume once net R recovers to this

TOKEN = os.environ.get("TG_TOKEN")
CHAT = os.environ.get("TG_CHAT")
STATE = Path("state.json")


def send(text):
    print(text)
    if TOKEN and CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": CHAT, "text": text},
                timeout=20,
            )
        except Exception as e:
            print("telegram error", e)


def load_state():
    if STATE.exists():
        try:
            s = json.loads(STATE.read_text())
            return {"last_bar": s.get("last_bar", ""), "paused": s.get("paused", False)}
        except Exception:
            pass
    return {"last_bar": "", "paused": False}


def save_state(state):
    STATE.write_text(json.dumps(state))


def get_data():
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, interval=INTERVAL, period="59d", progress=False, auto_adjust=False)
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


def add_indicators(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = rma(tr, 14)

    # ----- liquidity levels -----
    df["ll"] = l.shift(1).rolling(SWING_BARS).min()      # recent swing low
    df["hh"] = h.shift(1).rolling(SWING_BARS).max()      # recent swing high
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


def add_signals(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    atr = df["atr"]
    rng = h - l
    body_lo = np.minimum(o, c)
    body_hi = np.maximum(o, c)
    big = rng >= MIN_RANGE_ATR * atr

    sweep_lo = pd.Series(False, index=df.index)
    sweep_hi = pd.Series(False, index=df.index)
    for col in ["ll", "asia_lo", "pdl"]:
        sweep_lo = sweep_lo | ((l < df[col]) & (c > df[col]))
    for col in ["hh", "asia_hi", "pdh"]:
        sweep_hi = sweep_hi | ((h > df[col]) & (c < df[col]))

    buy = sweep_lo & (c > o) & big & ((body_lo - l) >= WICK_RATIO * rng) & (df["ema50"] > df["ema200"])
    sell = sweep_hi & (c < o) & big & ((h - body_hi) >= WICK_RATIO * rng) & (df["ema50"] < df["ema200"])

    d_buy = np.maximum(c - (l - SL_BUFFER_ATR * atr), MIN_SL_ATR * atr)
    d_sell = np.maximum((h + SL_BUFFER_ATR * atr) - c, MIN_SL_ATR * atr)
    buy = buy & (d_buy <= MAX_SL_ATR * atr)
    sell = sell & (d_sell <= MAX_SL_ATR * atr)

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


def backtest(df):
    hi, lo, cl = df["High"].values, df["Low"].values, df["Close"].values
    buy, sell, dist = df["buy"].values, df["sell"].values, df["sl_dist"].values
    n = len(df)
    i = 210
    trades = wins = 0
    net = 0.0
    while i < n - 1:
        if buy[i] or sell[i]:
            d = dist[i]
            side = 1 if buy[i] else -1
            sl = cl[i] - side * d
            tp = cl[i] + side * d * RR
            res = None
            j = i + 1
            while j < n:
                hit_sl = lo[j] <= sl if side == 1 else hi[j] >= sl
                hit_tp = hi[j] >= tp if side == 1 else lo[j] <= tp
                if hit_sl:
                    res = -1.0
                    break
                if hit_tp:
                    res = RR
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


def main():
    now = dt.datetime.now(dt.timezone.utc)
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

    state = load_state()
    if not STATE.exists():
        save_state(state)

    if not manual and not (START_HOUR <= now.hour < END_HOUR and now.weekday() < 5):
        return

    sym, df = get_data()
    if df is None:
        if manual:
            send("Bot is connected, but price data failed to load. Try again in a few minutes.")
        return

    df = add_signals(add_indicators(df))
    closed = df[df.index + pd.Timedelta(minutes=BAR_MIN) <= now]
    if len(closed) < 300:
        return

    trades, wins, net = backtest(closed)
    summary = stats_text(trades, wins, net)

    if manual:
        send(f"Gold liquidity bot connected ({sym}, {INTERVAL}, RR {RR}).\n{summary}")

    changed = False
    if trades >= MIN_TRADES:
        if not state["paused"] and net <= PAUSE_BELOW:
            state["paused"] = True
            changed = True
            send(f"Signals PAUSED: the strategy is losing right now.\n{summary}")
        elif state["paused"] and net >= RESUME_ABOVE:
            state["paused"] = False
            changed = True
            send(f"Signals RESUMED: the strategy recovered.\n{summary}")
    if changed:
        save_state(state)
    if state["paused"]:
        return

    bar = closed.iloc[-1]
    t = closed.index[-1]
    age = (now - (t + pd.Timedelta(minutes=BAR_MIN))).total_seconds() / 60
    if age > 2 * BAR_MIN:
        return  # signal is too old to act on
    if not (bar["buy"] or bar["sell"]):
        return
    if state["last_bar"] == str(t):
        return  # already alerted for this candle

    buy = bool(bar["buy"])
    side = "BUY" if buy else "SELL"
    close = float(bar["Close"])
    d = float(bar["sl_dist"])
    sl = close - d if buy else close + d
    tp = close + d * RR if buy else close - d * RR
    lots = int((ACCT * RISK_PCT / 100) / (d * CONTRACT) * 100) / 100
    lots_txt = f"{lots:.2f}" if lots >= 0.01 else "<0.01 (skip)"

    msg = (
        f"GOLD {side} ({INTERVAL}) - swept {swept_name(bar, side)}\n"
        f"Entry ~{close:.2f}\n"
        f"SL {sl:.2f} (${d:.2f})\n"
        f"TP {tp:.2f}\n"
        f"Lots {lots_txt}\n"
        f"{summary}\n"
        f"Candle closed {age:.0f} min ago. Check live price first and skip if it already moved."
    )
    send(msg)
    state["last_bar"] = str(t)
    save_state(state)


if __name__ == "__main__":
    main()
