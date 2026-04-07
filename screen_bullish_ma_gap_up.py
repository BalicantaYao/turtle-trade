#!/usr/bin/env python3
"""
台股多頭排列＋跳空高開篩選器

篩選條件：
1. 多頭排列：MA5 > MA10 > MA20 > MA60
2. 前一日收盤價 < 前一日開盤價（前一日為陰線）
3. 今日開盤價 > 前一日收盤價（今日跳空高開）

使用方式:
    python screen_bullish_ma_gap_up.py [--markets twse tpex] [--output results.csv]
"""

import argparse
import sys
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from tabulate import tabulate
from tqdm import tqdm

TWSE_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
TPEX_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"

BATCH_SIZE = 100
BATCH_SLEEP = 1.5  # seconds between yfinance batches


def get_stock_list(markets: list[str]) -> pd.DataFrame:
    """從 isin.twse.com.tw 爬取上市/上櫃股票清單。

    Returns:
        DataFrame with columns: code, name, market, yf_ticker
    """
    rows = []

    market_configs = []
    if "twse" in markets:
        market_configs.append(("TWSE", TWSE_URL, ".TW"))
    if "tpex" in markets:
        market_configs.append(("TPEX", TPEX_URL, ".TWO"))

    for market_name, url, suffix in market_configs:
        print(f"取得 {market_name} 股票清單...")
        try:
            resp = requests.get(url, timeout=30)
            resp.encoding = "big5"
            tables = pd.read_html(resp.text, header=0)
            df = tables[0]

            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)

            code_col = "有價證券代號及名稱"
            if code_col not in df.columns:
                code_col = df.columns[0]

            for _, row in df.iterrows():
                cell = str(row.get(code_col, "")).strip()
                if not cell or "\u3000" not in cell:
                    continue
                parts = cell.split("\u3000", 1)
                if len(parts) != 2:
                    continue
                code, name = parts[0].strip(), parts[1].strip()
                if not (code.isdigit() and len(code) == 4):
                    continue
                rows.append({
                    "code": code,
                    "name": name,
                    "market": market_name,
                    "yf_ticker": f"{code}{suffix}",
                })
        except Exception as e:
            print(f"警告：取得 {market_name} 清單失敗 - {e}", file=sys.stderr)

    if not rows:
        print("錯誤：無法取得任何股票清單", file=sys.stderr)
        sys.exit(1)

    result = pd.DataFrame(rows).drop_duplicates(subset="code").reset_index(drop=True)
    print(f"共取得 {len(result)} 檔股票（{', '.join(markets).upper()}）")
    return result


def calculate_signal(df: pd.DataFrame) -> tuple | None:
    """計算多頭排列＋跳空高開訊號。

    條件：
    1. MA5 > MA10 > MA20 > MA60（多頭排列）
    2. 前一日收盤 < 前一日開盤（前一日陰線）
    3. 今日開盤 > 前一日收盤（跳空高開）

    Args:
        df: DataFrame with columns Open, Close（已按日期排序）

    Returns:
        (today_open, prev_close, prev_open, gap_pct, ma5, ma10, ma20, ma60)
        或 None（資料不足或條件不符）
    """
    if len(df) < 61:  # 需要至少 61 筆：60 天 MA + 1 天前日資料
        return None

    close = df["Close"]
    open_ = df["Open"]

    # 計算移動平均（不含今日，以昨日收盤為基準）
    ma5 = float(close.iloc[-6:-1].mean())
    ma10 = float(close.iloc[-11:-1].mean())
    ma20 = float(close.iloc[-21:-1].mean())
    ma60 = float(close.iloc[-61:-1].mean())

    # 條件 1：多頭排列
    if not (ma5 > ma10 > ma20 > ma60):
        return None

    today_open = float(open_.iloc[-1])
    prev_close = float(close.iloc[-2])
    prev_open = float(open_.iloc[-2])

    # 條件 2：前一日為陰線（收盤 < 開盤）
    if not (prev_close < prev_open):
        return None

    # 條件 3：今日跳空高開（今日開盤 > 前一日收盤）
    if not (today_open > prev_close):
        return None

    gap_pct = (today_open - prev_close) / prev_close * 100

    return today_open, prev_close, prev_open, gap_pct, ma5, ma10, ma20, ma60


def screen_stocks(markets: list[str]) -> pd.DataFrame:
    """主流程：取清單 → 批次抓價格 → 篩選 → 排序。"""
    stock_list = get_stock_list(markets)
    tickers = stock_list["yf_ticker"].tolist()

    print(f"\n下載歷史股價（共 {len(tickers)} 檔，每批 {BATCH_SIZE} 檔）...")
    prices = {}
    batches = range(0, len(tickers), BATCH_SIZE)
    for i in tqdm(batches, desc="下載進度", unit="批"):
        batch = tickers[i: i + BATCH_SIZE]
        batch_str = " ".join(batch)
        try:
            raw = yf.download(
                batch_str,
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
                        open_ = raw["Open"][ticker]
                        close = raw["Close"][ticker]
                        combined = pd.DataFrame({"Open": open_, "Close": close}).dropna()
                        if len(combined) > 0:
                            prices[ticker] = combined
                    except KeyError:
                        pass
            else:
                ticker = batch[0]
                if "Open" in raw.columns and "Close" in raw.columns:
                    prices[ticker] = raw[["Open", "Close"]].dropna()

        except Exception as e:
            tqdm.write(f"警告：批次 {i // BATCH_SIZE + 1} 下載失敗 - {e}")

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_SLEEP)

    print("\n計算多頭排列＋跳空高開訊號...")
    records = []
    ticker_to_info = stock_list.set_index("yf_ticker").to_dict("index")

    for ticker, df in prices.items():
        signal = calculate_signal(df)
        if signal is None:
            continue

        today_open, prev_close, prev_open, gap_pct, ma5, ma10, ma20, ma60 = signal
        info = ticker_to_info.get(ticker, {})
        records.append({
            "代號": info.get("code", ticker),
            "名稱": info.get("name", ""),
            "市場": info.get("market", ""),
            "今日開盤": round(today_open, 2),
            "前日收盤": round(prev_close, 2),
            "前日開盤": round(prev_open, 2),
            "跳空%": round(gap_pct, 2),
            "MA5": round(ma5, 2),
            "MA10": round(ma10, 2),
            "MA20": round(ma20, 2),
            "MA60": round(ma60, 2),
            "screened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).sort_values("跳空%", ascending=False).reset_index(drop=True)
    return result


def print_table(df: pd.DataFrame) -> None:
    """在 Console 顯示結果表格。"""
    display_cols = ["代號", "名稱", "市場", "今日開盤", "前日收盤", "前日開盤", "跳空%", "MA5", "MA10", "MA20", "MA60"]
    print("\n" + "=" * 90)
    print(f"  篩選結果：共 {len(df)} 檔股票")
    print("=" * 90)
    print(tabulate(df[display_cols], headers="keys", tablefmt="simple", index=False))


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """儲存結果為 CSV。"""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n結果已儲存至：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="台股多頭排列＋跳空高開篩選器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
篩選條件：
  1. 多頭排列：MA5 > MA10 > MA20 > MA60
  2. 前一日收盤價 < 前一日開盤價（前一日為陰線）
  3. 今日開盤價 > 前一日收盤價（今日跳空高開）

範例:
  python screen_bullish_ma_gap_up.py
  python screen_bullish_ma_gap_up.py --markets twse
  python screen_bullish_ma_gap_up.py --output my_results.csv
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV 輸出路徑（預設：results_bullish_gap_YYYYMMDD.csv）",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["twse", "tpex"],
        default=["twse", "tpex"],
        help="篩選市場：twse（上市）、tpex（上櫃），預設兩者皆選",
    )

    args = parser.parse_args()
    output_path = args.output or f"results_bullish_gap_{datetime.now().strftime('%Y%m%d')}.csv"

    print("台股多頭排列＋跳空高開篩選器")
    print("  篩選條件：")
    print("    1. 多頭排列（MA5 > MA10 > MA20 > MA60）")
    print("    2. 前一日為陰線（收盤 < 開盤）")
    print("    3. 今日跳空高開（今日開盤 > 前一日收盤）")
    print(f"  市場：{', '.join(args.markets).upper()}")
    print(f"  輸出：{output_path}\n")

    results = screen_stocks(markets=args.markets)

    if results.empty:
        print("\n沒有符合條件的股票。")
    else:
        print_table(results)
        save_csv(results, output_path)


if __name__ == "__main__":
    main()
