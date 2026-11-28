"""Gold multi-timeframe champion bot.
Trains many strategies on 4 timeframes, keeps only the ones that pass a train/test check,
scans them all in parallel, and sends plain-English BUY / SELL messages to Telegram."""
import os
import json
import datetime as dt
from pathlib import Path

import numpy as np
import requests
import pandas as pd
import yfinance as yf

# ===== Settings =====
SYMBOLS = ["XAUUSD=X", "GC=F"]                       # spot gold first, then futures
TFS = ["5m", "15m", "30m", "1h"]
TF_MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
YF_INT = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m"}
YF_PERIOD = {"5m": "59d", "15m": "59d", "30m": "59d", "1h": "180d"}
RR = 2.0                                             # take profit = 2 x stop (1:2)
SPREAD = 0.30                                        # gold spread in $ for the tests
ACCT = 5000
RISK_PCT = 0.5
CONTRACT = 100
START_HOUR, END_HOUR = 7, 20                         # UTC, Mon-Fri
MAX_SIGNALS_PER_DAY = 4
DAILY_STOP_R = 2.0                                   # stop alerts after this loss today
MAX_LOSS_STREAK = 2                                  # ...or this many losses in a row
MIN_GAP_MIN = 30                                     # minutes between two signal messages
MAX_CHAMPIONS = 6
MAX_PER_TF = 2
MIN_EXPECTANCY = 0.10                                # avg R per trade needed (train + test)
MIN_TRAIN = {"5m": 25, "15m": 20, "30m": 15, "1h": 12}
MIN_TEST = {"5m": 10, "15m": 8, "30m": 6, "1h": 5}
BENCH_BELOW = -1.5                                   # sit a strategy out if its recent R is this bad

DEFAULTS = {"mode": "sweep", "swing": 36, "wick": 0.40, "min_range": 0.6, "sl_buf": 0.2,
            "min_sl": 0.8, "max_sl": 4.0, "trend": True, "bb_k": 2.0, "rsi": 30, "sl_atr": 2.0}
MODE_NAME = {"sweep": "Liquidity sweep", "meanrev": "Mean reversion", "pullback": "Trend pullback",
             "engulf": "Price-action engulfing at liquidity"}

TOKEN = os.environ.get("TG_TOKEN")
CHAT = os.environ.get("TG_CHAT")
STATE = Path("state.json")


# ---------------------------------------------------------------- helpers
def send(text):
    print(text)
    if TOKEN and CHAT:
        try:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          data={"chat_id": CHAT, "text": text[:4000]}, timeout=20)
        except Exception as e:
            print("telegram error", e)


def ist_time(ts):
    return (ts + pd.Timedelta(hours=5, minutes=30)).strftime("%I:%M %p")


def fresh_state():
    return {"train_date": "", "champions": [], "last_alert": {}, "last_signal": "", "open": [],
            "today": {"date": "", "signals": 0, "wins": 0, "losses": 0, "r": 0.0, "streak": 0, "stopped": False}}


def load_state():
    s = fresh_state()
    if STATE.exists():
        try:
            s.update({k: v for k, v in json.loads(STATE.read_text()).items() if k in s})
        except Exception:
            pass
    return s


def save_state(state):
    STATE.write_text(json.dumps(state))


def roll_today(state, today):
    if state["today"].get("date") != today:
        state["today"] = {"date": today, "signals": 0, "wins": 0, "losses": 0, "r": 0.0, "streak": 0, "stopped": False}


def today_text(state):
    t = state["today"]
    if t["signals"] == 0:
        return "TODAY: no signals yet"
    plural = "" if t["signals"] == 1 else "s"
    return f"TODAY: {t['signals']} signal{plural} | {t['wins']} won, {t['losses']} lost | {t['r']:+.1f}R"


# ---------------------------------------------------------------- data + indicators
def fetch(tf):
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, interval=YF_INT[tf], period=YF_PERIOD[tf], progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) > 400:
                idx = df.index
                df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
                return sym, df
        except Exception as e:
            print("data error", tf, sym, e)
    return None, None


def load_tf(tf, cache, now):
    if tf in cache:
        return cache[tf]
    sym, df = fetch(tf)
    if df is None:
        cache[tf] = None
        return None
    base = add_base(df)
    closed = base[base.index + pd.Timedelta(minutes=TF_MIN[tf]) <= now]
    cache[tf] = {"sym": sym, "closed": closed} if len(closed) > 300 else None
    return cache[tf]


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def add_base(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = rma(tr, 14)
    delta = c.diff()
    df["rsi"] = 100 - 100 / (1 + rma(delta.clip(lower=0), 14) / rma(-delta.clip(upper=0), 14))
    df["bb_mid"] = c.rolling(20).mean()
    df["bb_sd"] = c.rolling(20).std(ddof=0)
    df["day"] = df.index.normalize()
    asia = df[df.index.hour < 7]
    df["asia_hi"] = df["day"].map(asia.groupby("day")["High"].max())
    df["asia_lo"] = df["day"].map(asia.groupby("day")["Low"].min())
    df.loc[df.index.hour < 7, ["asia_hi", "asia_lo"]] = np.nan
    df["pdh"] = df["day"].map(df.groupby("day")["High"].max().shift(1))
    df["pdl"] = df["day"].map(df.groupby("day")["Low"].min().shift(1))
    return df


def add_swing(df, swing):
    df["ll"] = df["Low"].shift(1).rolling(swing).min()
    df["hh"] = df["High"].shift(1).rolling(swing).max()
    return df


# ---------------------------------------------------------------- strategies
def add_signals(df, p):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    atr = df["atr"]
    rng = h - l
    if p["trend"]:
        up_ok, dn_ok = df["ema50"] > df["ema200"], df["ema50"] < df["ema200"]
    else:
        up_ok = pd.Series(True, index=df.index)
        dn_ok = up_ok.copy()
    mode = p["mode"]

    if mode == "meanrev":
        lower = df["bb_mid"] - p["bb_k"] * df["bb_sd"]
        upper = df["bb_mid"] + p["bb_k"] * df["bb_sd"]
        buy = (c < lower) & (df["rsi"] < p["rsi"]) & up_ok
        sell = (c > upper) & (df["rsi"] > 100 - p["rsi"]) & dn_ok
        d_buy = d_sell = p["sl_atr"] * atr

    elif mode == "pullback":
        ema20 = c.ewm(span=20, adjust=False).mean()
        big = rng >= p["min_range"] * atr
        up, dn = df["ema50"] > df["ema200"], df["ema50"] < df["ema200"]
        buy = up & (l <= ema20) & (c > ema20) & (c > o) & (df["rsi"] > 45) & (df["rsi"] < 65) & big
        sell = dn & (h >= ema20) & (c < ema20) & (c < o) & (df["rsi"] < 55) & (df["rsi"] > 35) & big
        d_buy = d_sell = p["sl_atr"] * atr

    elif mode == "engulf":
        po, pc = o.shift(1), c.shift(1)
        t_lo = (l.shift(1) <= df["ll"].shift(1)) | (l.shift(1) <= df["asia_lo"].shift(1)) | (l.shift(1) <= df["pdl"].shift(1))
        t_hi = (h.shift(1) >= df["hh"].shift(1)) | (h.shift(1) >= df["asia_hi"].shift(1)) | (h.shift(1) >= df["pdh"].shift(1))
        bull = (pc < po) & (c > o) & (c >= po) & (o <= pc)
        bear = (pc > po) & (c < o) & (c <= po) & (o >= pc)
        big = rng >= p["min_range"] * atr
        buy = bull & t_lo & big & up_ok
        sell = bear & t_hi & big & dn_ok
        d_buy = np.maximum(c - (np.minimum(l, l.shift(1)) - p["sl_buf"] * atr), p["min_sl"] * atr)
        d_sell = np.maximum((np.maximum(h, h.shift(1)) + p["sl_buf"] * atr) - c, p["min_sl"] * atr)
        buy = buy & (d_buy <= p["max_sl"] * atr)
        sell = sell & (d_sell <= p["max_sl"] * atr)

    else:  # liquidity sweep
        body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
        big = rng >= p["min_range"] * atr
        s_lo = pd.Series(False, index=df.index)
        s_hi = pd.Series(False, index=df.index)
        for col in ["ll", "asia_lo", "pdl"]:
            s_lo = s_lo | ((l < df[col]) & (c > df[col]))
        for col in ["hh", "asia_hi", "pdh"]:
            s_hi = s_hi | ((h > df[col]) & (c < df[col]))
        buy = s_lo & (c > o) & big & ((body_lo - l) >= p["wick"] * rng) & up_ok
        sell = s_hi & (c < o) & big & ((h - body_hi) >= p["wick"] * rng) & dn_ok
        d_buy = np.maximum(c - (l - p["sl_buf"] * atr), p["min_sl"] * atr)
        d_sell = np.maximum((h + p["sl_buf"] * atr) - c, p["min_sl"] * atr)
        buy = buy & (d_buy <= p["max_sl"] * atr)
        sell = sell & (d_sell <= p["max_sl"] * atr)

    df["buy"], df["sell"] = buy, sell
    df["sl_dist"] = np.where(buy, d_buy, np.where(sell, d_sell, np.nan))
    return df


def backtest(df, start=210):
    hi, lo, cl = df["High"].values, df["Low"].values, df["Close"].values
    buy, sell, dist = df["buy"].values, df["sell"].values, df["sl_dist"].values
    n, i = len(df), start
    trades = wins = 0
    net = 0.0
    while i < n - 1:
        if buy[i] or sell[i]:
            d, side = dist[i], (1 if buy[i] else -1)
            sl, tp = cl[i] - side * d, cl[i] + side * d * RR
            res, j = None, i + 1
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


# ---------------------------------------------------------------- training
def candidate_configs():
    cfgs = []
    for swing in [24, 36]:
        for wick in [0.3, 0.4]:
            for mr in [0.6, 1.0]:
                for trend in [True, False]:
                    for msl in [0.8, 1.5]:
                        cfgs.append({"mode": "sweep", "swing": swing, "wick": wick, "min_range": mr, "trend": trend, "min_sl": msl})
    for k in [2.0, 2.5]:
        for rsi in [25, 30]:
            for sl in [1.5, 2.5]:
                for trend in [True, False]:
                    cfgs.append({"mode": "meanrev", "bb_k": k, "rsi": rsi, "sl_atr": sl, "trend": trend})
    for sl in [1.0, 1.5, 2.0]:
        for mr in [0.0, 0.6]:
            cfgs.append({"mode": "pullback", "sl_atr": sl, "min_range": mr})
    for swing in [24, 36]:
        for mr in [0.6, 1.0]:
            for trend in [True, False]:
                for msl in [0.8, 1.5]:
                    cfgs.append({"mode": "engulf", "swing": swing, "min_range": mr, "trend": trend, "min_sl": msl})
    return cfgs


def train_tf(tf, closed):
    split = int(len(closed) * 0.7)
    found, seen = [], []
    for swing in sorted({c.get("swing", 36) for c in candidate_configs()}):
        b = add_swing(closed.copy(), swing)
        for cfg in candidate_configs():
            if cfg.get("swing", 36) != swing:
                continue
            p = dict(DEFAULTS, **cfg)
            sig = add_signals(b.copy(), p)
            t1, w1, n1 = backtest(sig.iloc[:split], 210)
            t2, w2, n2 = backtest(sig.iloc[split:], 0)
            row = {"tf": tf, "cfg": cfg, "t1": t1, "w1": w1, "n1": round(n1, 2),
                   "t2": t2, "w2": w2, "n2": round(n2, 2)}
            if t1 >= MIN_TRAIN[tf] and t2 >= MIN_TEST[tf]:
                seen.append(row)
                tot = t1 + t2
                if n1 > 0 and n2 > 0 and (n1 + n2) / tot >= MIN_EXPECTANCY:
                    found.append(row)
    found.sort(key=lambda r: r["n1"] + r["n2"], reverse=True)
    seen.sort(key=lambda r: r["n1"] + r["n2"], reverse=True)
    return found, (seen[0] if seen else None)


def line_for(row):
    w1 = row["w1"] * 100 / row["t1"]
    w2 = row["w2"] * 100 / row["t2"]
    return (f"{MODE_NAME[row['cfg']['mode']]}: practice {row['t1']} trades {w1:.0f}% win {row['n1']:+.1f}R | "
            f"unseen test {row['t2']} trades {w2:.0f}% win {row['n2']:+.1f}R")


def train_all(now, cache):
    all_found, report, loaded = [], [], 0
    for tf in TFS:
        got = load_tf(tf, cache, now)
        if got is None:
            report.append(f"{tf}: price data not available")
            continue
        loaded += 1
        found, closest = train_tf(tf, got["closed"])
        if found:
            report.append(f"{tf}: {line_for(found[0])}")
            all_found += found[:MAX_PER_TF]
        elif closest:
            report.append(f"{tf}: nothing passed (closest: {MODE_NAME[closest['cfg']['mode']]}, "
                          f"practice {closest['n1']:+.1f}R, test {closest['n2']:+.1f}R)")
        else:
            report.append(f"{tf}: too few trades to judge")
    all_found.sort(key=lambda r: r["n1"] + r["n2"], reverse=True)
    champs = []
    for r in all_found:
        if len(champs) >= MAX_CHAMPIONS:
            break
        r["key"] = r["tf"] + "|" + json.dumps(r["cfg"], sort_keys=True)
        champs.append(r)
    return champs, report, loaded


# ---------------------------------------------------------------- live
def scan(state, cache, now):
    hits = []
    for ch in state["champions"]:
        got = load_tf(ch["tf"], cache, now)
        if got is None:
            continue
        p = dict(DEFAULTS, **ch["cfg"])
        sig = add_signals(add_swing(got["closed"].copy(), p["swing"]), p)
        recent = sig.iloc[int(len(sig) * 0.6):]
        t, w, n = backtest(recent, 0)
        if t >= 10 and n <= BENCH_BELOW:
            continue  # this strategy is cold right now, the others cover for it
        bar = sig.iloc[-1]
        bar_t = sig.index[-1]
        close_time = bar_t + pd.Timedelta(minutes=TF_MIN[ch["tf"]])
        age = (now - close_time).total_seconds() / 60
        if age > 2 * TF_MIN[ch["tf"]] or not (bar["buy"] or bar["sell"]):
            continue
        if state["last_alert"].get(ch["key"]) == str(bar_t):
            continue
        hits.append({"ch": ch, "side": 1 if bar["buy"] else -1, "d": float(bar["sl_dist"]),
                     "close": float(bar["Close"]), "close_time": close_time, "bar_t": str(bar_t), "age": age,
                     "stats": (t, w, n)})
    return hits


def resolve_open(state, cache, now):
    keep = []
    for s in state["open"]:
        t0 = pd.Timestamp(s["t"])
        if (now - t0).total_seconds() > 24 * 3600:
            continue
        got = load_tf(s["tf"], cache, now)
        if got is None:
            keep.append(s)
            continue
        res = None
        for _, row in got["closed"][got["closed"].index >= t0].iterrows():
            if s["side"] == 1:
                if row["Low"] <= s["sl"]:
                    res = "SL"
                elif row["High"] >= s["tp"]:
                    res = "TP"
            else:
                if row["High"] >= s["sl"]:
                    res = "SL"
                elif row["Low"] <= s["tp"]:
                    res = "TP"
            if res:
                break
        if res is None:
            keep.append(s)
            continue
        r = (RR if res == "TP" else -1.0) - SPREAD / s["d"]
        td = state["today"]
        td["r"] = round(td["r"] + r, 2)
        if res == "TP":
            td["wins"] += 1
            td["streak"] = 0
        else:
            td["losses"] += 1
            td["streak"] += 1
        word = "hit TAKE PROFIT (win)" if res == "TP" else "hit STOP LOSS (loss)"
        send(f"RESULT: the {s['tf']} {'BUY' if s['side'] == 1 else 'SELL'} signal {word}. (approximate, from the price feed)\n"
             f"{today_text(state)}")
        if not td["stopped"] and (td["r"] <= -DAILY_STOP_R or td["streak"] >= MAX_LOSS_STREAK):
            td["stopped"] = True
            send("DAILY STOP: today has been rough, so the bot will send NO more signals today. "
                 "Close the app and come back tomorrow. Protecting the account is the job.")
    state["open"] = keep


def signal_message(lead, others, state):
    ch, side, d = lead["ch"], lead["side"], lead["d"]
    action = "BUY" if side == 1 else "SELL"
    icon = "GREEN" if side == 1 else "RED"
    tp_d = d * RR
    lots = int((ACCT * RISK_PCT / 100) / (d * CONTRACT) * 100) / 100
    lots_txt = f"{lots:.2f}" if lots >= 0.01 else "0.01 (minimum - risks a little more than usual)"
    sl_side, tp_side = ("below", "above") if side == 1 else ("above", "below")
    ex = 3500.00
    ex_sl = ex - d if side == 1 else ex + d
    ex_tp = ex + tp_d if side == 1 else ex - tp_d
    t, w, n = lead["stats"]
    valid = 2 * TF_MIN[ch["tf"]]
    extra = ""
    if others:
        extra = "Also agreeing: " + ", ".join(f"{o['ch']['tf']} {MODE_NAME[o['ch']['cfg']['mode']]}" for o in others) + "\n"
    return (
        f"{icon} SIGNAL: {action} GOLD NOW\n"
        f"Time: {ist_time(lead['close_time'])} IST ({lead['close_time'].strftime('%H:%M')} UTC). Valid for about {valid} minutes only.\n"
        f"Why: {MODE_NAME[ch['cfg']['mode']]} on the {ch['tf']} chart\n{extra}\n"
        f"DO THIS IN MT5:\n"
        f"1) Open XAUUSD, tap New Order, choose {action}\n"
        f"2) Lots: {lots_txt}\n"
        f"3) Stop Loss: ${d:.2f} {sl_side} your {action.lower()} price\n"
        f"4) Take Profit: ${tp_d:.2f} {tp_side} your {action.lower()} price\n"
        f"5) Tap {action}\n\n"
        f"Example only: if you {action.lower()} at {ex:,.2f}, Stop Loss = {ex_sl:,.2f} and Take Profit = {ex_tp:,.2f}\n"
        f"Money at risk: about ${ACCT * RISK_PCT / 100:.0f} ({RISK_PCT}% of ${ACCT:,})\n\n"
        f"SKIP if more than {valid} minutes have passed, or gold already moved ${d / 3:.2f} or more.\n"
        f"One trade at a time. Never remove or move the Stop Loss.\n"
        f"{today_text(state)}"
    )


def main():
    now = dt.datetime.now(dt.timezone.utc)
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    today = now.strftime("%Y-%m-%d")
    state = load_state()
    roll_today(state, today)
    if not STATE.exists():
        save_state(state)
    if not manual and not (START_HOUR <= now.hour < END_HOUR and now.weekday() < 5):
        return

    cache = {}
    if manual or state["train_date"] != today:
        champs, report, loaded = train_all(now, cache)
        if loaded == 0:
            if manual:
                send("The bot is connected, but price data could not be loaded. Try again in a few minutes.")
            return
        state["champions"], state["train_date"] = champs, today
        if champs:
            plan = "ACTIVE TODAY:\n" + "\n".join(f"- {c['tf']} {MODE_NAME[c['cfg']['mode']]}" for c in champs)
        else:
            plan = "STAND ASIDE TODAY: no strategy passed the tests, so NO signals will be sent. That is on purpose."
        send(f"Daily check ({today}): the bot is alive and re-trained on fresh data.\n\n"
             f"BEST STRATEGY ON EACH TIMEFRAME (practice = data it learned from, test = data it never saw):\n"
             + "\n".join(report) + f"\n\n{plan}\n\n{today_text(state)}\n"
             f"Limits: max {MAX_SIGNALS_PER_DAY} signals a day, stops after {MAX_LOSS_STREAK} losses in a row or -{DAILY_STOP_R:.0f}R.")
        save_state(state)

    resolve_open(state, cache, now)

    td = state["today"]
    if state["champions"] and not td["stopped"] and td["signals"] < MAX_SIGNALS_PER_DAY:
        gap_ok = True
        if state["last_signal"]:
            gap_ok = (now - pd.Timestamp(state["last_signal"])).total_seconds() >= MIN_GAP_MIN * 60
        hits = scan(state, cache, now)
        if hits and gap_ok:
            sides = {h["side"] for h in hits}
            if len(sides) > 1:
                print("strategies disagree, no signal this time")
            else:
                hits.sort(key=lambda h: h["ch"]["n1"] + h["ch"]["n2"], reverse=True)
                lead = hits[0]
                seen, others = {(lead["ch"]["tf"], lead["ch"]["cfg"]["mode"])}, []
                for h in hits[1:]:
                    k = (h["ch"]["tf"], h["ch"]["cfg"]["mode"])
                    if k not in seen:
                        seen.add(k)
                        others.append(h)
                td["signals"] += 1
                send(signal_message(lead, others, state))
                d, side = lead["d"], lead["side"]
                state["open"].append({"tf": lead["ch"]["tf"], "side": side, "d": d,
                                      "sl": lead["close"] - side * d, "tp": lead["close"] + side * d * RR,
                                      "t": str(lead["close_time"])})
                state["last_signal"] = str(now)
                for h in hits:
                    state["last_alert"][h["ch"]["key"]] = h["bar_t"]
    save_state(state)


if __name__ == "__main__":
    main()
