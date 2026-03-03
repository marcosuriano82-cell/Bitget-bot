"""
╔══════════════════════════════════════════════════════════════════╗
║     BITGET PAPER TRADING BOT  v2.0  —  Railway + Telegram       ║
║   RSI · MACD · EMA · Bollinger Bands · ATR · Volume             ║
╚══════════════════════════════════════════════════════════════════╝

VARIABILI D'AMBIENTE (da impostare su Railway):
    BITGET_API_KEY
    BITGET_API_SECRET
    BITGET_API_PASSPHRASE
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
import requests
from datetime import datetime

# ─────────────────────────────────────────────
#   CONFIGURAZIONE — legge da variabili ambiente
# ─────────────────────────────────────────────

API_KEY        = os.environ.get("BITGET_API_KEY",        "LA_TUA_API_KEY")
API_SECRET     = os.environ.get("BITGET_API_SECRET",     "IL_TUO_API_SECRET")
API_PASSPHRASE = os.environ.get("BITGET_API_PASSPHRASE", "LA_TUA_PASSPHRASE")
TG_TOKEN       = os.environ.get("TELEGRAM_BOT_TOKEN",    "")   # es. 123456:ABCdef...
TG_CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID",      "")   # es. 987654321

# Impostazioni trading
SYMBOLS        = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME      = "15m"
PAPER_BALANCE  = 1000.0
RISK_PER_TRADE = 0.02
LOOP_INTERVAL  = 60

# Parametri indicatori
RSI_PERIOD     = 14
RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70
EMA_FAST       = 9
EMA_SLOW       = 21
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
BB_PERIOD      = 20
BB_STD         = 2.0
ATR_PERIOD     = 14
ATR_MULTIPLIER = 2.0

# ─────────────────────────────────────────────
#   STATO PAPER TRADING
# ─────────────────────────────────────────────

paper_state = {
    "balance":        PAPER_BALANCE,
    "positions":      {},
    "trades_history": [],
    "total_pnl":      0.0,
    "wins":           0,
    "losses":         0,
}

# ─────────────────────────────────────────────
#   TELEGRAM
# ─────────────────────────────────────────────

def tg_send(msg: str):
    """Invia un messaggio Telegram. Silenzioso se non configurato."""
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id":    TG_CHAT_ID,
            "text":       msg,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception as e:
        log(f"Telegram error: {e}", "WARN")

# ─────────────────────────────────────────────
#   LOGGING
# ─────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

# ─────────────────────────────────────────────
#   EXCHANGE
# ─────────────────────────────────────────────

def connect_exchange():
    exchange = ccxt.bitget({
        "apiKey":   API_KEY,
        "secret":   API_SECRET,
        "password": API_PASSPHRASE,
        "options":  {"defaultType": "spot"},
    })
    exchange.load_markets()
    return exchange

def fetch_ohlcv(exchange, symbol, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# ─────────────────────────────────────────────
#   INDICATORI TECNICI
# ─────────────────────────────────────────────

def compute_indicators(df):
    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # EMA
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # MACD
    ef = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    es = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"]        = ef - es
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_mid"]   = df["close"].rolling(BB_PERIOD).mean()
    bb_std         = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std

    # ATR
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()

    # Volume
    df["vol_sma"]   = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"]

    return df

# ─────────────────────────────────────────────
#   SEGNALE
# ─────────────────────────────────────────────

def generate_signal(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    score, reasons = 0, []

    # RSI
    if last["rsi"] < RSI_OVERSOLD:
        score += 25; reasons.append(f"RSI ipervenduto ({last['rsi']:.1f})")
    elif last["rsi"] > RSI_OVERBOUGHT:
        score -= 25; reasons.append(f"RSI ipercomprato ({last['rsi']:.1f})")

    # EMA cross
    if last["ema_fast"] > last["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]:
        score += 30; reasons.append("🟢 Golden cross EMA")
    elif last["ema_fast"] < last["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"]:
        score -= 30; reasons.append("🔴 Death cross EMA")
    elif last["ema_fast"] > last["ema_slow"]:
        score += 10; reasons.append("Trend rialzista (EMA)")
    else:
        score -= 10; reasons.append("Trend ribassista (EMA)")

    # MACD
    if last["macd"] > last["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
        score += 20; reasons.append("MACD cross rialzista")
    elif last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
        score -= 20; reasons.append("MACD cross ribassista")
    elif last["macd_hist"] > 0:
        score += 5
    else:
        score -= 5

    # Bollinger
    if last["close"] <= last["bb_lower"]:
        score += 15; reasons.append("Prezzo su BB inferiore → rimbalzo")
    elif last["close"] >= last["bb_upper"]:
        score -= 15; reasons.append("Prezzo su BB superiore → inversione")

    # Volume
    if last["vol_ratio"] > 1.5:
        boost = 10 if score > 0 else -10
        score += boost
        reasons.append(f"Volume {last['vol_ratio']:.1f}x sopra media")

    signal = "BUY" if score >= 40 else ("SELL" if score <= -40 else "HOLD")
    return signal, min(abs(score), 100), reasons, last

# ─────────────────────────────────────────────
#   PAPER TRADING ENGINE
# ─────────────────────────────────────────────

def paper_open_position(symbol, side, price, atr):
    risk_amount = paper_state["balance"] * RISK_PER_TRADE
    sl = price - ATR_MULTIPLIER * atr if side == "BUY" else price + ATR_MULTIPLIER * atr
    tp = price + ATR_MULTIPLIER * atr * 2 if side == "BUY" else price - ATR_MULTIPLIER * atr * 2
    sl_dist = abs(price - sl)
    if sl_dist == 0:
        return
    qty = risk_amount / sl_dist

    paper_state["positions"][symbol] = {
        "side": side, "entry": price, "qty": qty,
        "sl": sl, "tp": tp, "open_time": datetime.now().isoformat(),
    }

    msg = (f"📂 <b>POSIZIONE APERTA</b>\n"
           f"💱 {symbol} — {side}\n"
           f"💲 Entrata: <b>{price:.4f}</b>\n"
           f"🛑 Stop-Loss: {sl:.4f}\n"
           f"🎯 Take-Profit: {tp:.4f}\n"
           f"📦 Qty: {qty:.6f}\n"
           f"💰 Bilancio: {paper_state['balance']:.2f} USDT")
    log(msg.replace("<b>","").replace("</b>",""))
    tg_send(msg)


def paper_check_positions(prices):
    to_close = []
    for sym, pos in paper_state["positions"].items():
        price = prices.get(sym)
        if price is None:
            continue
        hit_sl = (pos["side"]=="BUY" and price<=pos["sl"]) or (pos["side"]=="SELL" and price>=pos["sl"])
        hit_tp = (pos["side"]=="BUY" and price>=pos["tp"]) or (pos["side"]=="SELL" and price<=pos["tp"])
        if hit_sl:
            to_close.append((sym, price, "STOP-LOSS"))
        elif hit_tp:
            to_close.append((sym, price, "TAKE-PROFIT"))
    for sym, price, reason in to_close:
        paper_close_position(sym, price, reason)


def paper_close_position(symbol, price, reason):
    pos = paper_state["positions"].pop(symbol, None)
    if not pos:
        return
    pnl = (price - pos["entry"]) * pos["qty"] if pos["side"] == "BUY" \
          else (pos["entry"] - price) * pos["qty"]
    paper_state["balance"]   += pnl
    paper_state["total_pnl"] += pnl
    if pnl >= 0:
        paper_state["wins"] += 1
        emoji = "✅"
    else:
        paper_state["losses"] += 1
        emoji = "❌"

    paper_state["trades_history"].append({
        "symbol": symbol, "side": pos["side"],
        "entry": pos["entry"], "exit": price,
        "pnl": pnl, "reason": reason,
        "time": datetime.now().isoformat(),
    })

    total  = paper_state["wins"] + paper_state["losses"]
    wr     = paper_state["wins"] / total * 100 if total > 0 else 0
    profit = paper_state["balance"] - PAPER_BALANCE

    msg = (f"{emoji} <b>POSIZIONE CHIUSA</b> — {reason}\n"
           f"💱 {symbol} — {pos['side']}\n"
           f"📥 Entrata: {pos['entry']:.4f}\n"
           f"📤 Uscita:  {price:.4f}\n"
           f"{'🟢' if pnl>=0 else '🔴'} PnL: <b>{pnl:+.2f} USDT</b>\n\n"
           f"📊 <b>Statistiche</b>\n"
           f"💰 Bilancio: {paper_state['balance']:.2f} USDT\n"
           f"📈 Profitto totale: {profit:+.2f} USDT\n"
           f"🏆 Win rate: {wr:.1f}% ({paper_state['wins']}W/{paper_state['losses']}L)")
    log(msg.replace("<b>","").replace("</b>",""))
    tg_send(msg)
    save_state()


def send_daily_report():
    """Manda un riepilogo giornaliero su Telegram."""
    s      = paper_state
    total  = s["wins"] + s["losses"]
    wr     = s["wins"] / total * 100 if total > 0 else 0
    profit = s["balance"] - PAPER_BALANCE
    msg = (f"📊 <b>REPORT GIORNALIERO</b>\n\n"
           f"💰 Bilancio: <b>{s['balance']:.2f} USDT</b>\n"
           f"{'🟢' if profit>=0 else '🔴'} Profitto: {profit:+.2f} USDT\n"
           f"🏆 Win rate: {wr:.1f}%\n"
           f"✅ Vincite: {s['wins']} | ❌ Perdite: {s['losses']}\n"
           f"📂 Posizioni aperte: {len(s['positions'])}\n\n"
           f"🔄 Bot attivo e funzionante ✓")
    tg_send(msg)

# ─────────────────────────────────────────────
#   PERSISTENZA
# ─────────────────────────────────────────────

STATE_FILE = "/tmp/paper_state.json"

def save_state():
    with open(STATE_FILE, "w") as f:
        json.dump(paper_state, f, indent=2)

def load_state():
    global paper_state
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            paper_state = json.load(f)
        log("Stato precedente caricato.")

# ─────────────────────────────────────────────
#   MAIN LOOP
# ─────────────────────────────────────────────

def main():
    log("=" * 55)
    log("  BITGET PAPER TRADING BOT v2.0 — Railway + Telegram")
    log("=" * 55)

    load_state()

    try:
        log("Connessione a Bitget...")
        exchange = connect_exchange()
        log("Connesso con successo!")
    except Exception as e:
        log(f"Connessione non riuscita: {e} — uso dati simulati.", "WARN")
        exchange = None

    tg_send("🤖 <b>Bot avviato!</b>\n"
            f"📊 Coppie: {', '.join(SYMBOLS)}\n"
            f"⏱ Timeframe: {TIMEFRAME}\n"
            f"💰 Capitale virtuale: {PAPER_BALANCE} USDT\n"
            f"🔄 Analisi ogni {LOOP_INTERVAL}s")

    cycle      = 0
    report_every = 1440 // (LOOP_INTERVAL // 60) if LOOP_INTERVAL >= 60 else 1440

    while True:
        try:
            cycle += 1
            log(f"--- Ciclo #{cycle} ---")
            current_prices = {}

            for symbol in SYMBOLS:
                try:
                    log(f"Analisi {symbol} [{TIMEFRAME}]...")

                    if exchange:
                        df = fetch_ohlcv(exchange, symbol)
                    else:
                        idx   = pd.date_range(end=datetime.now(), periods=200, freq="15min")
                        close = 45000 + np.cumsum(np.random.randn(200) * 200)
                        df    = pd.DataFrame({
                            "open":   close - np.random.rand(200)*50,
                            "high":   close + np.random.rand(200)*150,
                            "low":    close - np.random.rand(200)*150,
                            "close":  close,
                            "volume": np.random.rand(200)*1000+300,
                        }, index=idx)

                    df = compute_indicators(df)
                    signal, confidence, reasons, last = generate_signal(df)
                    price = float(last["close"])
                    atr   = float(last["atr"])
                    current_prices[symbol] = price

                    log(f"  {symbol} → {signal} (confidenza {confidence}%) @ {price:.4f}")
                    for r in reasons:
                        log(f"    · {r}")

                    if confidence >= 60 and signal in ("BUY","SELL") and symbol not in paper_state["positions"]:
                        paper_open_position(symbol, signal, price, atr)

                except Exception as e:
                    log(f"Errore su {symbol}: {e}", "WARN")

            paper_check_positions(current_prices)

            # Report giornaliero
            if cycle % report_every == 0:
                send_daily_report()

            log(f"Prossima analisi tra {LOOP_INTERVAL}s")
            time.sleep(LOOP_INTERVAL)

        except KeyboardInterrupt:
            log("Bot fermato dall'utente.")
            tg_send("🛑 <b>Bot fermato.</b>")
            save_state()
            break
        except Exception as e:
            log(f"Errore nel loop: {e}", "ERROR")
            time.sleep(10)


if __name__ == "__main__":
    main()
