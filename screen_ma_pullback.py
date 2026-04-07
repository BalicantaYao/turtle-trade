#!/usr/bin/env python3
"""
台股均線回檔篩選器

篩選條件：
  1. 20MA 向上（今日 20MA > 5 日前 20MA）
  2. 股價在 5MA 之下（當前收盤價 < 5MA）

使用方式:
    python screen_ma_pullback.py [--markets twse tpex] [--output results.csv]
"""

import argparse
import sys
import time
from datetime import datetime

import pandas as pd
import twstock
import yfinance as yf
from tabulate import tabulate
from tqdm import tqdm

BATCH_SIZE = 100
BATCH_SLEEP = 1.5


def get_stock_list(markets: list[str]) -> pd.DataFrame:
    """使用 twstock 取得上市/上櫃普通股清單。

    Returns:
        DataFrame: columns = [code, name, market, yf_ticker]
    """
    market_map = {
        "twse": ("上市", ".TW"),
        "tpex": ("上櫃", ".TWO"),
    }

    rows = []
    for market_key in markets:
        market_label, suffix = market_map[market_key]
        for code, info in twstock.codes.items():
            if (
                code.isdigit()
                and len(code) == 4
                and info.type == "股票"
                and info.market == market_label
            ):
                rows.append({
                    "code": code,
                    "name": info.name,
                    "market": market_label,
                    "yf_ticker": f"{code}{suffix}",
                })

    if not rows:
        print("錯誤：無法取得任何股票清單", file=sys.stderr)
        sys.exit(1)

    result = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)
    print(f"共取得 {len(result)} 檔股票（{', '.join(markets).upper()}）")
    return result


def check_ma_pullback(close: pd.Series) -> bool:
    """判斷是否符合「20MA 向上 且 股價在 5MA 之下」。

    Args:
        close: 收盤價序列（已按日期排序），至少需 25 筆

    Returns:
        True 表示符合條件，False 則否
    """
    if len(close) < 25:
        return False

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()

    ma20_rising = float(ma20.iloc[-1]) > float(ma20.iloc[-6])   # 今日 > 5日前
    price_below_ma5 = float(close.iloc[-1]) < float(ma5.iloc[-1])

    return ma20_rising and price_below_ma5


def screen_stocks(markets: list[str]) -> pd.DataFrame:
    """主流程：取清單 → 批次抓價格 → 套用均線篩選 → 排序。"""
    stock_list = get_stock_list(markets)
    tickers = stock_list["yf_ticker"].tolist()

    print(f"\n下載歷史股價（共 {len(tickers)} 檔，每批 {BATCH_SIZE} 檔）...")
    prices: dict[str, pd.Series] = {}

    for i in tqdm(range(0, len(tickers), BATCH_SIZE), desc="下載進度", unit="批"):
        batch = tickers[i: i + BATCH_SIZE]
        try:
            raw = yf.download(
                " ".join(batch),
                period="3mo",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        s = raw["Close"][ticker].dropna()
                        if len(s) >= 25:
                            prices[ticker] = s
                    except KeyError:
                        pass
            else:
                ticker = batch[0]
                if "Close" in raw.columns:
                    s = raw["Close"].dropna()
                    if len(s) >= 25:
                        prices[ticker] = s

        except Exception as e:
            tqdm.write(f"警告：批次 {i // BATCH_SIZE + 1} 下載失敗 - {e}")

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_SLEEP)

    print("\n套用篩選條件：20MA 向上 且 股價 < 5MA ...")
    ticker_info = stock_list.set_index("yf_ticker").to_dict("index")
    records = []

    for ticker, close in prices.items():
        if not check_ma_pullback(close):
            continue

        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        price = float(close.iloc[-1])
        info = ticker_info.get(ticker, {})

        records.append({
            "代號": info.get("code", ticker),
            "名稱": info.get("name", ""),
            "市場": info.get("market", ""),
            "現價": round(price, 2),
            "5MA": round(ma5, 2),
            "20MA": round(ma20, 2),
            "價格/5MA%": round(price / ma5 * 100, 2),
            "screened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values("價格/5MA%", ascending=True)
        .reset_index(drop=True)
    )


def print_table(df: pd.DataFrame) -> None:
    display_cols = ["代號", "名稱", "市場", "現價", "5MA", "20MA", "價格/5MA%"]
    print("\n" + "=" * 65)
    print(f"  篩選結果：共 {len(df)} 檔股票（20MA 向上 且 股價 < 5MA）")
    print("=" * 65)
    print(tabulate(df[display_cols], headers="keys", tablefmt="simple", index=False))


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n結果已儲存至：{path}")


def main():
    parser = argparse.ArgumentParser(
        description="台股均線回檔篩選器（20MA 向上 且 股價在 5MA 之下）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python screen_ma_pullback.py
  python screen_ma_pullback.py --markets twse
  python screen_ma_pullback.py --output pullback.csv
        """,
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["twse", "tpex"],
        default=["twse", "tpex"],
        help="篩選市場：twse（上市）、tpex（上櫃），預設兩者皆選",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV 輸出路徑（預設：ma_pullback_YYYYMMDD.csv）",
    )
    args = parser.parse_args()

    output_path = args.output or f"ma_pullback_{datetime.now().strftime('%Y%m%d')}.csv"

    print("台股均線回檔篩選器")
    print(f"  市場：{', '.join(args.markets).upper()}")
    print(f"  條件：20MA 向上（今日 > 5日前）且 股價 < 5MA")
    print(f"  輸出：{output_path}\n")

    results = screen_stocks(markets=args.markets)

    if results.empty:
        print("\n沒有符合條件的股票。")
    else:
        print_table(results)
        save_csv(results, output_path)


if __name__ == "__main__":
    main()
