# 🐆 FX-Leopard

> An AI-powered FX market surveillance agent that watches markets 24/5, detects high-probability trade setups, and sends rich Telegram alerts — so you don't have to stare at charts.

---

## 🎯 What It Does

FX-Leopard acts as your personal trading analyst running 24/5 on a VPS. It:

- 📡 **Watches** real-time price feeds across major FX pairs and commodities simultaneously
- 📊 **Analyses** multiple timeframes (M5, M15, H1, H4, D1) per instrument using best-practice indicators
- 📰 **Reads** live financial news and parses sentiment using GPT-4o
- 📅 **Monitors** the economic calendar and pre-alerts before high-impact events
- ⚡ **Detects** volatility spikes and momentum moves in real time
- 🔔 **Notifies** you on Telegram with a full signal brief including entry, SL, TP, R:R, and reasoning

You stay sharp for the **kill decision only**. The leopard does the stalking.

---

## 📦 Instruments Covered

| Category | Pairs |
|---|---|
| FX Majors | EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD |
| Commodities | XAUUSD, XAGUSD, USOIL, UKOIL |

---

## 🧠 Signal Types

| Alert Type | Description |
|---|---|
| 🟢 SIGNAL | Full confluence setup — entry, SL, TP, reasoning |
| 🟡 WATCH | Setup forming, waiting for confirmation |
| ⚡ VOLATILITY | Sudden price spike detected |
| 📰 NEWS | High-impact event incoming or just released |

---

## 🏗️ Architecture

```
Data Layer         →   Price Feed (WebSocket) + News API + Economic Calendar
Analysis Layer     →   Technical Engine + NLP Sentiment + Volatility Monitor
Scoring Layer      →   Confluence Engine (threshold: 7/10)
Notification Layer →   Telegram Bot
```

---

## ⚙️ Configuration

All settings live in `config/config.yaml` (copy from `config/config.yaml.example`):

```yaml
pairs:
  - EURUSD
  - GBPUSD
  - XAUUSD
  - USOIL

timeframes: [M5, M15, H1, H4, D1]

confluence_threshold: 7.0

notifications:
  channel: telegram
  bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
  chat_id: "YOUR_TELEGRAM_CHAT_ID"

api_keys:
  twelvedata: "YOUR_TWELVEDATA_API_KEY"
  openai: "YOUR_OPENAI_API_KEY"
  newsapi: "YOUR_NEWSAPI_KEY"
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/israfilzadehemin/FX-Leopard.git
cd FX-Leopard

# 2. Copy and fill in your config
cp config/config.yaml.example config/config.yaml

# 3. Build and run with Docker
docker-compose up -d

# 4. Watch your Telegram for alerts 🐆
```

---

## 🗂️ Project Structure

```
FX-Leopard/
├── config/
│   ├── config.yaml.example      # Template config
│   └── config.yaml              # Your config (gitignored)
├── src/
│   ├── main.py                  # Entry point
│   ├── data/
│   │   ├── price_feed.py        # WebSocket price stream
│   │   ├── news_feed.py         # News fetcher
│   │   └── calendar_feed.py     # Economic calendar
│   ├── analysis/
│   │   ├── technical.py         # Indicator engine
│   │   ├── sentiment.py         # GPT-4o news sentiment
│   │   ├── volatility.py        # Spike detector
│   │   └── confluence.py        # Signal scoring engine
│   ├── notifications/
│   │   └── telegram_bot.py      # Telegram alert sender
│   └── storage/
│       └── signal_logger.py     # SQLite signal log
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Price Data | TwelveData WebSocket API |
| Indicators | pandas-ta |
| News NLP | OpenAI GPT-4o |
| Scheduler | APScheduler + asyncio |
| Notifications | python-telegram-bot |
| Storage | SQLite |
| Deployment | Docker + Hetzner VPS |

---

## 📋 Issues / Roadmap

See [GitHub Issues](https://github.com/israfilzadehemin/FX-Leopard/issues) for the full build task list.

---

## ⚠️ Disclaimer

FX-Leopard is a **signal advisory tool only**. It does not execute trades. All trading decisions remain yours. Trading FX and commodities carries significant risk.

---

*Built with 🐆 and GitHub Copilot*