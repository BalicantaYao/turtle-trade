#!/usr/bin/env python3
"""
台股 5MA 向上突破篩選器

篩選條件：
  1. 昨日收盤 < 5MA，且今日收盤 > 5MA（剛突破 5MA）
  2. 多頭排列：10MA > 20MA > 60MA

股票清單來源：twstock 函式庫（上市 + 上櫃一般股票）

使用方式:
    python screen_ma5_breakout.py [--output results.csv] [--markets twse tpex]
"""

import argparse
import pickle
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import twstock
import yfinance as yf
from tabulate import tabulate
from tqdm import tqdm

BATCH_SIZE = 100
BATCH_SLEEP = 1.5
CACHE_DIR = Path(".cache")


# ── 快取 ──────────────────────────────────────────────────────────────────────

def _cache_path(markets: list[str]) -> Path:
    key = "_".join(sorted(markets))
    return CACHE_DIR / f"prices_{date.today():%Y%m%d}_{key}.pkl"


def load_cache(markets: list[str]) -> dict[str, pd.DataFrame] | None:
    path = _cache_path(markets)
    if path.exists():
        print(f"載入今日快取：{path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(prices: dict[str, pd.DataFrame], markets: list[str]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(markets)
    with open(path, "wb") as f:
        pickle.dump(prices, f)
    print(f"快取已儲存：{path}")


# ── 股票清單 ──────────────────────────────────────────────────────────────────

def get_stock_list(markets: list[str]) -> pd.DataFrame:
    """從 twstock.codes 取得上市/上櫃一般股票清單。"""
    market_map = {
        "twse": ("上市", ".TW"),
        "tpex": ("上櫃", ".TWO"),
    }
    wanted_markets = {market_map[m][0]: market_map[m][1] for m in markets if m in market_map}

    rows = []
    for code, info in twstock.codes.items():
        if not (code.isdigit() and len(code) == 4):
            continue
        if getattr(info, "type", "") != "股票":
            continue
        market_name = getattr(info, "market", "")
        if market_name not in wanted_markets:
            continue
        rows.append({
            "code": code,
            "name": getattr(info, "name", ""),
            "market": market_name,
            "yf_ticker": f"{code}{wanted_markets[market_name]}",
        })

    if not rows:
        print("錯誤：無法從 twstock 取得任何股票清單", file=sys.stderr)
        sys.exit(1)

    result = pd.DataFrame(rows).drop_duplicates(subset="code").reset_index(drop=True)
    print(f"共取得 {len(result)} 檔股票（{', '.join(markets).upper()}）")
    return result


# ── 下載股價 ──────────────────────────────────────────────────────────────────

def fetch_prices_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """批次下載 yfinance 歷史股價（最近 3 個月）。"""
    results = {}

    for i in tqdm(range(0, len(tickers), BATCH_SIZE), desc="下載進度", unit="批"):
        batch = tickers[i: i + BATCH_SIZE]
        try:
            raw = yf.download(
                " ".join(batch),
                period="6mo",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        close = raw["Close"][ticker].dropna()
                        if len(close) >= 61:
                            results[ticker] = pd.DataFrame({"Close": close})
                    except KeyError:
                        pass
            else:
                ticker = batch[0]
                if "Close" in raw.columns:
                    close = raw["Close"].dropna()
                    if len(close) >= 61:
                        results[ticker] = pd.DataFrame({"Close": close})

        except Exception as e:
            tqdm.write(f"警告：批次 {i // BATCH_SIZE + 1} 下載失敗 - {e}")

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_SLEEP)

    return results


# ── 訊號計算 ──────────────────────────────────────────────────────────────────

def calculate_signal(df: pd.DataFrame) -> dict | None:
    """判斷是否符合 5MA 突破 + 多頭排列（MA10 > MA20 > MA60）。

    Returns:
        dict with signal details, 或 None（不符合）
    """
    if len(df) < 61:
        return None

    close = df["Close"]
    ma5  = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    today_close    = float(close.iloc[-1])
    today_ma5      = float(ma5.iloc[-1])
    today_ma10     = float(ma10.iloc[-1])
    today_ma20     = float(ma20.iloc[-1])
    today_ma60     = float(ma60.iloc[-1])
    yesterday_close = float(close.iloc[-2])
    yesterday_ma5   = float(ma5.iloc[-2])

    if any(pd.isna(v) for v in [today_ma5, today_ma10, today_ma20, today_ma60, yesterday_ma5]):
        return None

    # 5MA 昨下今上
    if not (yesterday_close < yesterday_ma5 and today_close > today_ma5):
        return None

    # 多頭排列
    if not (today_ma10 > today_ma20 > today_ma60):
        return None

    return {
        "現價":   round(today_close, 2),
        "昨收":   round(yesterday_close, 2),
        "MA5":    round(today_ma5, 2),
        "MA10":   round(today_ma10, 2),
        "MA20":   round(today_ma20, 2),
        "MA60":   round(today_ma60, 2),
        "突破幅%": round((today_close / today_ma5 - 1) * 100, 2),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def screen_stocks(markets: list[str]) -> pd.DataFrame:
    """取清單 → 批次抓價格（含快取）→ 計算訊號 → 篩選 → 排序。"""
    stock_list = get_stock_list(markets)
    tickers = stock_list["yf_ticker"].tolist()

    prices = load_cache(markets)
    if prices is None:
        print(f"\n下載歷史股價（共 {len(tickers)} 檔，每批 {BATCH_SIZE} 檔）...")
        prices = fetch_prices_batch(tickers)
        save_cache(prices, markets)
    else:
        print(f"使用快取資料（共 {len(prices)} 檔），略過下載。")

    print("\n計算 5MA 突破訊號...")
    records = []
    ticker_to_info = stock_list.set_index("yf_ticker").to_dict("index")

    for ticker, df in prices.items():
        signal = calculate_signal(df)
        if signal is None:
            continue

        info = ticker_to_info.get(ticker, {})
        code = info.get("code", ticker)
        records.append({
            "代號": code,
            "名稱": info.get("name", ""),
            "市場": info.get("market", ""),
            **signal,
            "screened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "技術圖": f"https://www.wantgoo.com/stock/{code}/technical-chart",
        })

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values("突破幅%", ascending=False)
        .reset_index(drop=True)
    )


def print_table(df: pd.DataFrame) -> None:
    display_cols = ["代號", "名稱", "市場", "現價", "昨收", "MA5", "MA10", "MA20", "MA60", "突破幅%"]
    print("\n" + "=" * 75)
    print(f"  5MA 向上突破（MA10 > MA20 > MA60）：共 {len(df)} 檔股票")
    print("=" * 75)
    print(tabulate(df[display_cols], headers="keys", tablefmt="simple", showindex=False))


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n結果已儲存至：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="台股 5MA 向上突破篩選器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
篩選條件:
  1. 昨日收盤 < 5MA，且今日收盤 > 5MA（剛穿越 5 日均線向上）
  2. 多頭排列：MA10 > MA20 > MA60

範例:
  python screen_ma5_breakout.py
  python screen_ma5_breakout.py --markets twse
  python screen_ma5_breakout.py --output my_results.csv
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV 輸出路徑（預設：ma5_breakout_YYYYMMDD.csv）",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["twse", "tpex"],
        default=["twse", "tpex"],
        help="篩選市場：twse（上市）、tpex（上櫃），預設兩者皆選",
    )

    args = parser.parse_args()
    output_path = args.output or f"ma5_breakout_{datetime.now().strftime('%Y%m%d')}.csv"

    print("台股 5MA 向上突破篩選器")
    print(f"  市場：{', '.join(args.markets).upper()}")
    print(f"  條件：昨收 < MA5，且今收 > MA5，且 MA10 > MA20 > MA60")
    print(f"  輸出：{output_path}\n")

    results = screen_stocks(markets=args.markets)

    if results.empty:
        print("\n沒有符合條件的股票。")
    else:
        print_table(results)
        save_csv(results, output_path)


if __name__ == "__main__":
    main()
