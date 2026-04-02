# 🐆 FX-Leopard

> An AI-powered FX market surveillance agent that watches markets 24/5 across multiple instruments and timeframes, detects high-probability trade setups, and delivers rich signal alerts to your Telegram — so you never have to stare at charts again.

---

## 🎯 What It Does

FX-Leopard acts as your personal trading analyst running around the clock. It:

- 📡 **Watches** all major FX pairs and commodities simultaneously in real-time
- 📊 **Analyzes** multiple timeframes (M5 → D1) for technical confluence
- 📰 **Reads** live financial news and scores sentiment per currency using GPT-4o
- 📅 **Monitors** the economic calendar and pre-alerts before high-impact events
- ⚡ **Detects** sudden volatility spikes and momentum moves
- 📲 **Notifies** you on Telegram with full signal briefs including entry, SL, TP, RR, and reasoning

You stay sharp for the decision. The agent does all the stalking.

---

## 📡 Instruments Covered

**FX Majors:** EURUSD · GBPUSD · USDJPY · USDCHF · AUDUSD · USDCAD · NZDUSD

**Commodities:** XAUUSD (Gold) · XAGUSD (Silver) · USOIL · UKOIL

---

## 🧠 Signal Detection Logic

### Timeframe Stack
| Timeframe | Role |
|-----------|------|
| M5  | Entry trigger & precision timing |
| M15 | Setup confirmation |
| H1  | Structure & trend direction |
| H4  | Bias & key levels |
| D1  | Overall market context |

### Technical Indicators
- **Trend:** EMA 20/50/200, ADX
- **Momentum:** RSI (divergence + oversold/overbought), MACD crossover
- **Volatility:** Bollinger Band squeeze → expansion, ATR spike
- **Structure:** S/R breaks, higher highs/lows, round numbers, daily levels
- **Candles:** Engulfing, Pin Bar, Inside Bar, Hammer, Shooting Star at key levels

### Confluence Scoring
| Score | Action |
|-------|--------|
| ≥ 7.0 | 🟢 Full Signal Alert sent |
| 5.0 – 6.9 | 🟡 Watch Alert sent |
| < 5.0 | Silent — ignored |

---

## 📲 Notification Types

### 🟢 Full Signal Alert
```
🐆 SIGNAL — XAUUSD | LONG | H1 Confirmed

📍 Entry Zone:   1,987.50 – 1,989.00
🛡️ Stop Loss:    1,982.10  (≈ 45 pips)
🎯 Take Profit:  2,001.00  (≈ 120 pips)
⚖️  R:R Ratio:   1 : 2.7

📊 Technical Confluence (8.5/10):
  ✅ M15 Bullish pin bar at H1 support
  ✅ RSI recovering from oversold (M15 + H1)
  ✅ Price reclaimed 200 EMA on M15
  ✅ MACD bullish cross forming on H1
  ✅ ADX = 28 (trending)
  ⚠️  D1 still in consolidation range

📰 Fundamental Backdrop:
  ✅ USD softened after weak ADP data
  ✅ Gold sentiment bullish (safe haven bid)
  ⚠️  Fed speech in 2h — volatility risk

⏱️ Signal generated: 14:32 UTC
🔕 Invalidated if: 4H candle closes below 1,981.00
```

### 🟡 Watch Alert
Notifies when a setup is forming but not yet confirmed.

### ⚡ Volatility Alert
Fires when a sudden pip spike or ATR expansion is detected with a news cause identified.

### 📅 Pre-News Alert
Fires 15 minutes before high-impact calendar events (NFP, CPI, FOMC, etc.) with forecast vs previous data.

---

## ⚙️ Configuration

All settings live in `config.yaml` — no code changes needed:

```yaml
pairs:
  - EURUSD
  - GBPUSD
  - USDJPY
  - USDCHF
  - AUDUSD
  - USDCAD
  - NZDUSD
  - XAUUSD
  - XAGUSD
  - USOIL
  - UKOIL

timeframes: [M5, M15, H1, H4, D1]

confluence:
  signal_threshold: 7.0
  watch_threshold: 5.0

notifications:
  channel: telegram
  bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
  chat_id: "YOUR_TELEGRAM_CHAT_ID"

api_keys:
  twelvedata: "YOUR_TWELVEDATA_API_KEY"
  openai: "YOUR_OPENAI_API_KEY"
  newsapi: "YOUR_NEWSAPI_KEY"

volatility:
  atr_multiplier: 1.5
  pip_spike_threshold: 30
  spike_window_minutes: 5
```

---

## 🏗️ Architecture

```
Data Ingestion
  ├── Price Feed (TwelveData WebSocket)
  ├── News Feed (NewsAPI + Reuters RSS)
  └── Economic Calendar (ForexFactory scraper)
          │
          ▼
AI Agent Core
  ├── Technical Analysis Engine (pandas-ta, multi-TF)
  ├── News/Sentiment Engine (GPT-4o)
  ├── Volatility Monitor (ATR + pip spike)
  └── Confluence Scoring Engine
          │
          ▼
Notification Layer
  └── Telegram Bot (python-telegram-bot)
          │
          ▼
Signal Logger
  └── SQLite (all signals logged for performance review)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Price Data | TwelveData WebSocket API |
| Indicators | pandas-ta |
| News NLP | OpenAI GPT-4o API |
| Scheduling | APScheduler + asyncio |
| Notifications | python-telegram-bot |
| Storage | SQLite |
| Deployment | Docker + Hetzner VPS (CX11 ~€4/mo) |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/israfilzadehemin/FX-Leopard.git
cd FX-Leopard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your API keys and Telegram credentials
```

### 4. Run
```bash
python main.py
```

### 5. Deploy (optional)
```bash
docker build -t fx-leopard .
docker run -d --restart always -v $(pwd)/config.yaml:/app/config.yaml fx-leopard
```

---

## 📁 Project Structure

```
FX-Leopard/
├── main.py                    # Entry point
├── config.yaml                # Your personal config (git-ignored)
├── config.example.yaml        # Template config
├── requirements.txt
├── Dockerfile
├── .env.example
├── agent/
│   ├── __init__.py
│   ├── engine.py              # Main agent orchestrator
│   ├── price_feed.py          # TwelveData WebSocket client
│   ├── indicators.py          # Multi-TF technical analysis
│   ├── confluence.py          # Signal scoring engine
│   ├── news.py                # News fetch + GPT-4o sentiment
│   ├── calendar.py            # Economic calendar monitor
│   └── volatility.py          # Spike detection
├── notifications/
│   ├── __init__.py
│   ├── telegram_bot.py        # Telegram notification sender
│   └── formatters.py          # Alert message formatting
├── storage/
│   ├── __init__.py
│   └── signal_logger.py       # SQLite signal logger
└── tests/
    ├── test_indicators.py
    ├── test_confluence.py
    ├── test_news.py
    └── test_notifications.py
```

---

## 📊 Signal Logging

Every signal is logged to SQLite for performance review:

| Column | Description |
|--------|-------------|
| timestamp | When signal was generated |
| instrument | e.g. XAUUSD |
| direction | LONG / SHORT |
| entry_zone | Price range |
| stop_loss | SL price |
| take_profit | TP price |
| rr_ratio | Risk/reward |
| confluence_score | Score out of 10 |
| technical_reasons | JSON array |
| news_context | Sentiment summary |
| outcome | WIN / LOSS / PENDING / INVALIDATED |

---

## ⚠️ Disclaimer

FX-Leopard is a **market surveillance and signal notification tool**. It does not execute trades automatically. All trading decisions remain with the user. Trading FX and commodities carries significant financial risk.

---

## 📄 License

MIT
