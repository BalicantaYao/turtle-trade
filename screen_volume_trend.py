#!/usr/bin/env python3
"""
台股量增趨勢篩選器

篩選條件：
  1. 今日成交量 > 昨日成交量 × 1.5
  2. 收盤價 > 5 日均線
  3. 多頭排列：5 日均線 > 10 日均線 > 20 日均線

股票清單來源：twstock 函式庫（上市 + 上櫃一般股票）

使用方式:
    python screen_volume_trend.py [--volume-ratio 1.5] [--output results.csv] [--markets twse tpex]
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
BATCH_SLEEP = 1.5  # seconds between yfinance batches


def get_stock_list(markets: list[str]) -> pd.DataFrame:
    """從 twstock.codes 取得上市/上櫃一般股票清單。

    Returns:
        DataFrame with columns: code, name, market, yf_ticker
    """
    market_map = {
        "twse": ("上市", ".TW"),
        "tpex": ("上櫃", ".TWO"),
    }

    wanted_markets = {}
    for m in markets:
        if m in market_map:
            wanted_markets[market_map[m][0]] = market_map[m][1]

    rows = []
    for code, info in twstock.codes.items():
        # 只保留 4 位數字代碼（排除 ETF、權證、指數等）
        if not (code.isdigit() and len(code) == 4):
            continue
        # 只保留一般股票（type == '股票'）
        if getattr(info, "type", "") != "股票":
            continue
        market_name = getattr(info, "market", "")
        if market_name not in wanted_markets:
            continue
        suffix = wanted_markets[market_name]
        rows.append({
            "code": code,
            "name": getattr(info, "name", ""),
            "market": market_name,
            "yf_ticker": f"{code}{suffix}",
        })

    if not rows:
        print("錯誤：無法從 twstock 取得任何股票清單", file=sys.stderr)
        sys.exit(1)

    result = pd.DataFrame(rows).drop_duplicates(subset="code").reset_index(drop=True)
    print(f"共取得 {len(result)} 檔股票（{', '.join(markets).upper()}）")
    return result


def fetch_prices_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """批次下載 yfinance 歷史股價（最近 60 個交易日）。

    Returns:
        {yf_ticker: DataFrame with columns Close, Volume}
    """
    results = {}
    batches = range(0, len(tickers), BATCH_SIZE)

    for i in tqdm(batches, desc="下載進度", unit="批"):
        batch = tickers[i: i + BATCH_SIZE]
        batch_str = " ".join(batch)
        try:
            raw = yf.download(
                batch_str,
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
                        close = raw["Close"][ticker].dropna()
                        volume = raw["Volume"][ticker].dropna()
                        if len(close) >= 21 and len(volume) >= 2:
                            results[ticker] = pd.DataFrame(
                                {"Close": close, "Volume": volume}
                            ).dropna()
                    except KeyError:
                        pass
            else:
                ticker = batch[0]
                if "Close" in raw.columns and "Volume" in raw.columns:
                    df = raw[["Close", "Volume"]].dropna()
                    if len(df) >= 21:
                        results[ticker] = df

        except Exception as e:
            tqdm.write(f"警告：批次 {i // BATCH_SIZE + 1} 下載失敗 - {e}")

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_SLEEP)

    return results


def calculate_signals(df: pd.DataFrame, volume_ratio: float) -> dict | None:
    """計算量增 + 均線訊號。

    Args:
        df:           DataFrame with columns Close, Volume（已按日期升序排列）
        volume_ratio: 今日成交量需超過昨日的倍數

    Returns:
        dict with signal details, 或 None（不符合或資料不足）
    """
    if len(df) < 21:
        return None

    close = df["Close"]
    volume = df["Volume"]

    # 計算均線
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()

    last = df.index[-1]
    prev = df.index[-2]

    cur_close = float(close.iloc[-1])
    cur_vol = float(volume.iloc[-1])
    prev_vol = float(volume.iloc[-2])
    cur_ma5 = float(ma5.iloc[-1])
    cur_ma10 = float(ma10.iloc[-1])
    cur_ma20 = float(ma20.iloc[-1])

    if prev_vol <= 0:
        return None

    ratio = cur_vol / prev_vol

    # 三個條件
    cond_volume = ratio >= volume_ratio
    cond_above_ma5 = cur_close > cur_ma5
    cond_bull = cur_ma5 > cur_ma10 > cur_ma20

    if not (cond_volume and cond_above_ma5 and cond_bull):
        return None

    return {
        "現價": round(cur_close, 2),
        "今量": int(cur_vol),
        "昨量": int(prev_vol),
        "量比": round(ratio, 2),
        "MA5": round(cur_ma5, 2),
        "MA10": round(cur_ma10, 2),
        "MA20": round(cur_ma20, 2),
    }


def screen_stocks(volume_ratio: float, markets: list[str]) -> pd.DataFrame:
    """主流程：取清單 → 批次抓價格 → 計算訊號 → 篩選 → 排序。"""
    stock_list = get_stock_list(markets)
    tickers = stock_list["yf_ticker"].tolist()

    print(f"\n下載歷史股價（共 {len(tickers)} 檔，每批 {BATCH_SIZE} 檔）...")
    prices = fetch_prices_batch(tickers)

    print(f"\n計算量增 + 均線訊號（量比門檻：{volume_ratio}x）...")
    records = []
    ticker_to_info = stock_list.set_index("yf_ticker").to_dict("index")

    for ticker, df in prices.items():
        signal = calculate_signals(df, volume_ratio)
        if signal is None:
            continue

        info = ticker_to_info.get(ticker, {})
        records.append({
            "代號": info.get("code", ticker),
            "名稱": info.get("name", ""),
            "市場": info.get("market", ""),
            **signal,
            "screened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    if not records:
        return pd.DataFrame()

    result = (
        pd.DataFrame(records)
        .sort_values("量比", ascending=False)
        .reset_index(drop=True)
    )
    return result


def print_table(df: pd.DataFrame) -> None:
    """在 Console 顯示結果表格。"""
    display_cols = ["代號", "名稱", "市場", "現價", "今量", "昨量", "量比", "MA5", "MA10", "MA20"]
    print("\n" + "=" * 70)
    print(f"  篩選結果：共 {len(df)} 檔股票")
    print("=" * 70)
    print(tabulate(df[display_cols], headers="keys", tablefmt="simple", showindex=False))


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """儲存結果為 CSV。"""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n結果已儲存至：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="台股量增趨勢篩選器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
篩選條件:
  1. 今日成交量 > 昨日成交量 × volume-ratio（預設 1.5 倍）
  2. 收盤價 > 5 日均線
  3. 多頭排列：5MA > 10MA > 20MA

範例:
  python screen_volume_trend.py
  python screen_volume_trend.py --volume-ratio 2.0
  python screen_volume_trend.py --markets twse
  python screen_volume_trend.py --output my_results.csv
        """,
    )
    parser.add_argument(
        "--volume-ratio",
        type=float,
        default=1.5,
        help="今日成交量需超過昨日的倍數（預設 1.5）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV 輸出路徑（預設：volume_trend_YYYYMMDD.csv）",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["twse", "tpex"],
        default=["twse", "tpex"],
        help="篩選市場：twse（上市）、tpex（上櫃），預設兩者皆選",
    )

    args = parser.parse_args()

    if args.volume_ratio <= 0:
        print("錯誤：volume-ratio 需大於 0", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or f"volume_trend_{datetime.now().strftime('%Y%m%d')}.csv"

    print("台股量增趨勢篩選器")
    print(f"  市場：{', '.join(args.markets).upper()}")
    print(f"  量比門檻：今日量 >= 昨日量 × {args.volume_ratio}")
    print(f"  均線條件：收盤 > MA5，且 MA5 > MA10 > MA20")
    print(f"  輸出：{output_path}\n")

    results = screen_stocks(volume_ratio=args.volume_ratio, markets=args.markets)

    if results.empty:
        print("\n沒有符合條件的股票。")
    else:
        print_table(results)
        save_csv(results, output_path)


if __name__ == "__main__":
    main()
