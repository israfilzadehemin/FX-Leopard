"""
FX-Leopard Signal Report CLI.

Examples::

    python src/storage/report.py --since 2026-03-01 --symbol EURUSD
    python src/storage/report.py --summary --since 2026-04-01
    python src/storage/report.py --last 20
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running directly or as part of the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage.signal_logger import SignalLogger  # noqa: E402

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_DIVIDER = "─" * 40

_TYPE_ICONS: dict[str, str] = {
    "SIGNAL": "🟢",
    "WATCH": "🟡",
    "VOLATILITY": "⚡",
    "NEWS": "📰",
    "CALENDAR": "📅",
}


def _print_summary(sl: SignalLogger, since: str | None) -> None:
    stats = sl.get_summary(since=since)
    total = stats["total_signals"]
    by_type: dict[str, int] = stats.get("by_type") or {}
    by_symbol: dict[str, int] = stats.get("by_symbol") or {}
    by_direction: dict[str, int] = stats.get("by_direction") or {}
    avg_score: float = stats.get("avg_score") or 0.0
    avg_rr: float = stats.get("avg_rr") or 0.0
    period: str = stats.get("period") or "all time"

    period_display = period.replace(" to ", " → ")

    print("📊 FX-Leopard Signal Report")
    print(f"Period: {period_display}")
    print(_DIVIDER)
    print(f"Total signals fired : {total}")

    for stype, count in sorted(by_type.items()):
        icon = _TYPE_ICONS.get(stype, "  ")
        print(f"  {icon} {stype:<14}: {count}")

    if by_direction:
        print()
        for direction, count in sorted(by_direction.items()):
            icon = "🔵" if direction == "BUY" else "🔴"
            print(f"  {icon} {direction:<14}: {count}")

    if by_symbol:
        print()
        print("Top symbols:")
        for sym, count in list(by_symbol.items())[:10]:
            print(f"  {sym:<8}: {count} signals")

    print()
    print(f"Avg confluence score: {avg_score} / 10")
    print(f"Avg R:R ratio       : {avg_rr}")
    print(_DIVIDER)


def _print_signals(rows: list[dict]) -> None:
    if not rows:
        print("No signals found.")
        return

    print(f"{'#':<5} {'Time':<22} {'Symbol':<10} {'Type':<10} {'Dir':<6} {'Score':<7} {'RR'}")
    print(_DIVIDER)
    for row in rows:
        icon = _TYPE_ICONS.get(row.get("signal_type", ""), "  ")
        print(
            f"{row['id']:<5} "
            f"{str(row.get('fired_at', ''))[:19]:<22} "
            f"{row.get('symbol', ''):<10} "
            f"{icon} {row.get('signal_type', ''):<8} "
            f"{row.get('direction', ''):<6} "
            f"{row.get('score') or 0.0:<7.2f} "
            f"{row.get('rr_ratio') or '-'}"
        )
    print(_DIVIDER)
    print(f"Showing {len(rows)} signal(s).")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FX-Leopard Signal Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        default="data/signals.db",
        help="Path to the SQLite database (default: data/signals.db)",
    )
    parser.add_argument(
        "--since",
        metavar="DATE",
        default=None,
        help="Filter signals on or after this ISO8601 date, e.g. 2026-03-01",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Filter to a specific symbol, e.g. EURUSD",
    )
    parser.add_argument(
        "--type",
        dest="signal_type",
        default=None,
        help="Filter to signal type: SIGNAL | WATCH",
    )
    parser.add_argument(
        "--direction",
        default=None,
        help="Filter to direction: BUY | SELL",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Show the last N signals (default: 100)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="Print an aggregated summary instead of the raw signal list",
    )

    args = parser.parse_args()

    try:
        sl = SignalLogger(db_path=args.db)
    except Exception as exc:
        print(f"Error opening database '{args.db}': {exc}", file=sys.stderr)
        sys.exit(1)

    if args.summary:
        _print_summary(sl, since=args.since)
    else:
        limit = args.last if args.last is not None else 100
        rows = sl.get_signals(
            since=args.since,
            symbol=args.symbol,
            signal_type=args.signal_type,
            direction=args.direction,
            limit=limit,
        )
        _print_signals(rows)


if __name__ == "__main__":
    main()
