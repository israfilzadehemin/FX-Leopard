# 🐆 FX-Leopard

> An AI-powered FX market surveillance agent that watches charts, news & volatility 24/5 — and sends you Telegram signals so you don't have to stare at screens.

---

## 🧠 Concept

FX-Leopard is a **discretionary intraday trading assistant**. It does all the legwork:
- Watches 10+ instruments simultaneously across 5 timeframes
- Reads real-time news and parses sentiment using GPT-4o
- Detects volatility spikes and scheduled high-impact events
- Scores setups using a confluence engine
- Sends you a rich Telegram alert only when a high-probability setup is forming

You stay sharp for the **kill decision only**. 🐆

---

## 📡 Instruments Covered

| Type | Pairs |
|---|---|
| FX Majors | EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD |
| Commodities | XAUUSD, XAGUSD, USOIL |

---

## 🕐 Timeframe Stack

| Timeframe | Role |
|---|---|
| M5 | Entry trigger & precision timing |
| M15 | Setup confirmation |
| H1 | Structure & trend direction |
| H4 | Bias & key levels |
| D1 | Overall market context |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                  DATA INGESTION                  │
│  Price Feed (WebSocket)  │  News  │  Calendar    │
└──────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│                 AI AGENT CORE                    │
│                                                  │
│  Technical Engine  │  News/NLP  │  Volatility   │
│                                                  │
│           Confluence Scoring Engine              │
│     Score ≥ 7 → SIGNAL | 5-6 → WATCH | <5 →    │
└──────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│              NOTIFICATION LAYER                  │
│                 Telegram Bot                     │
└──────────────────────────────────────────────────┘
```

---

## 🧩 Components

| # | Module | Description |
|---|---|---|
| 1 | `price_feed` | WebSocket connection to TwelveData for real-time OHLCV |
| 2 | `indicator_engine` | Multi-timeframe technical analysis using pandas-ta |
| 3 | `confluence_engine` | Scores setups across technical + fundamental signals |
| 4 | `news_engine` | Fetches news, parses sentiment via GPT-4o |
| 5 | `calendar_engine` | Monitors economic calendar, fires pre/post event alerts |
| 6 | `volatility_monitor` | Detects ATR spikes and sudden pip movements |
| 7 | `notifier` | Telegram bot that formats and sends signal messages |
| 8 | `signal_logger` | SQLite-based logging of all signals for performance review |
| 9 | `deploy` | Dockerfile + VPS deployment config |

---

## ⚙️ Configuration

All configuration lives in `config/config.yaml`:

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

timeframes: [5min, 15min, 1h, 4h, 1day]

confluence:
  signal_threshold: 7.0
  watch_threshold: 5.0

notifications:
  channel: telegram
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"

api_keys:
  twelvedata: "${TWELVEDATA_API_KEY}"
  openai: "${OPENAI_API_KEY}"
  newsapi: "${NEWSAPI_KEY}"

volatility:
  atr_multiplier: 1.5
  pip_spike_threshold: 30
  spike_window_minutes: 5
```

---

## 📲 Signal Format

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
  ✅ Gold demand sentiment bullish
  ⚠️  Fed speech in 2h — volatility risk

⏱️ Signal generated: 14:32 UTC
🔕 Invalidated if: 4H candle closes below 1,981.00
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- TwelveData API key (free tier)
- OpenAI API key
- NewsAPI key
- Telegram Bot token + chat ID

### Setup
```bash
git clone https://github.com/israfilzadehemin/FX-Leopard.git
cd FX-Leopard
cp config/config.example.yaml config/config.yaml
# Fill in your API keys in config/config.yaml or set env vars
pip install -r requirements.txt
python main.py
```

### Docker
```bash
docker build -t fx-leopard .
docker run --env-file .env fx-leopard
```

---

## 📁 Project Structure

```
FX-Leopard/
├── main.py                  # Entry point
├── config/
│   ├── config.yaml          # Your config (gitignored)
│   └── config.example.yaml  # Template
├── src/
│   ├── price_feed/          # WebSocket price ingestion
│   ├── indicator_engine/    # Technical analysis
│   ├── confluence_engine/   # Signal scoring
│   ├── news_engine/         # News fetch + GPT sentiment
│   ├── calendar_engine/     # Economic calendar
│   ├── volatility_monitor/  # Spike detection
│   ├── notifier/            # Telegram bot
│   └── signal_logger/       # SQLite signal log
├── tests/
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## ⚠️ Disclaimer

FX-Leopard is a **signal assistant**, not an execution bot. All trade decisions remain yours. This tool does not place orders. Past signal performance does not guarantee future results. Trade responsibly.

---

## 📄 License

MIT