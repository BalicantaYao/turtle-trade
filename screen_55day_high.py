#!/usr/bin/env python3
"""
台股 55 天高點篩選器 (Turtle Trading System)

篩選台股（上市/上櫃）中當前收盤價接近 55 天最高點的股票。
使用方式:
    python screen_55day_high.py [--threshold 0.95] [--output results.csv] [--markets twse tpex]
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

# twstock market type 對應
_TWSTOCK_MARKET_MAP = {
    "上市": ("TWSE", ".TW"),
    "上櫃": ("TPEX", ".TWO"),
}


def _fetch_from_isin(markets: list[str]) -> list[dict]:
    """從 isin.twse.com.tw 爬取股票清單（原始方法）。"""
    rows = []
    market_configs = []
    if "twse" in markets:
        market_configs.append(("TWSE", TWSE_URL, ".TW"))
    if "tpex" in markets:
        market_configs.append(("TPEX", TPEX_URL, ".TWO"))

    for market_name, url, suffix in market_configs:
        print(f"取得 {market_name} 股票清單（isin）...")
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
    return rows


def _fetch_from_twstock(markets: list[str]) -> list[dict]:
    """使用 twstock 函式庫取得股票清單。"""
    try:
        import twstock
    except ImportError:
        print("錯誤：請先安裝 twstock：pip install twstock", file=sys.stderr)
        sys.exit(1)

    print("取得股票清單（twstock）...")
    want_twse = "twse" in markets
    want_tpex = "tpex" in markets

    rows = []
    for code, info in twstock.codes.items():
        # 只保留 4 位數字代碼
        if not (code.isdigit() and len(code) == 4):
            continue
        market_label = getattr(info, "market", "")
        if market_label not in _TWSTOCK_MARKET_MAP:
            continue
        market_name, suffix = _TWSTOCK_MARKET_MAP[market_label]
        if market_name == "TWSE" and not want_twse:
            continue
        if market_name == "TPEX" and not want_tpex:
            continue
        rows.append({
            "code": code,
            "name": getattr(info, "name", ""),
            "market": market_name,
            "yf_ticker": f"{code}{suffix}",
        })
    return rows


def get_stock_list(markets: list[str], source: str = "isin") -> pd.DataFrame:
    """取得上市/上櫃股票清單。

    Args:
        markets: 市場列表，可包含 "twse" 和/或 "tpex"
        source:  資料來源，"isin"（預設）或 "twstock"

    Returns:
        DataFrame with columns: code, name, market, yf_ticker
    """
    if source == "twstock":
        rows = _fetch_from_twstock(markets)
    else:
        rows = _fetch_from_isin(markets)

    if not rows:
        print("錯誤：無法取得任何股票清單", file=sys.stderr)
        sys.exit(1)

    result = pd.DataFrame(rows).drop_duplicates(subset="code").reset_index(drop=True)
    print(f"共取得 {len(result)} 檔股票（{', '.join(markets).upper()}，來源：{source}）")
    return result


def fetch_prices_batch(tickers: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """批次下載 yfinance 歷史股價。

    Returns:
        {yf_ticker: DataFrame(Date, High, Close)} — 只含 High 和 Close 欄位
    """
    results = {}

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i: i + BATCH_SIZE]
        batch_str = " ".join(batch)
        try:
            raw = yf.download(
                batch_str,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                # 多檔：columns = (price_type, ticker)
                for ticker in batch:
                    try:
                        high = raw["High"][ticker].dropna()
                        close = raw["Close"][ticker].dropna()
                        if len(high) > 0 and len(close) > 0:
                            results[ticker] = pd.DataFrame({"High": high, "Close": close})
                    except KeyError:
                        pass
            else:
                # 單檔
                ticker = batch[0]
                if "High" in raw.columns and "Close" in raw.columns:
                    results[ticker] = raw[["High", "Close"]].dropna()

        except Exception as e:
            print(f"警告：批次下載失敗（第 {i // BATCH_SIZE + 1} 批）- {e}", file=sys.stderr)

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_SLEEP)

    return results


def calculate_55day_signal(df: pd.DataFrame) -> tuple[float, float, float] | None:
    """計算 55 天高點訊號。

    Args:
        df: DataFrame with columns High, Close (已按日期排序)

    Returns:
        (current_price, high_55d, ratio) 或 None（資料不足 55 天）
    """
    if len(df) < 55:
        return None

    window = df.tail(55)
    high_55d = float(window["High"].max())
    current_price = float(df["Close"].iloc[-1])

    if high_55d <= 0:
        return None

    ratio = current_price / high_55d
    return current_price, high_55d, ratio


def screen_stocks(threshold: float, markets: list[str], source: str = "isin") -> pd.DataFrame:
    """主流程：取清單 → 批次抓價格 → 計算 55 天高點 → 篩選 → 排序。"""
    stock_list = get_stock_list(markets, source=source)
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
                        high = raw["High"][ticker].dropna()
                        close = raw["Close"][ticker].dropna()
                        if len(high) > 0 and len(close) > 0:
                            prices[ticker] = pd.DataFrame({"High": high, "Close": close})
                    except KeyError:
                        pass
            else:
                ticker = batch[0]
                if "High" in raw.columns and "Close" in raw.columns:
                    prices[ticker] = raw[["High", "Close"]].dropna()

        except Exception as e:
            tqdm.write(f"警告：批次 {i // BATCH_SIZE + 1} 下載失敗 - {e}")

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_SLEEP)

    print(f"\n計算 55 天高點訊號（門檻：{threshold * 100:.1f}%）...")
    records = []
    ticker_to_info = stock_list.set_index("yf_ticker").to_dict("index")

    for ticker, df in prices.items():
        signal = calculate_55day_signal(df)
        if signal is None:
            continue
        current_price, high_55d, ratio = signal
        if ratio < threshold:
            continue

        info = ticker_to_info.get(ticker, {})
        records.append({
            "代號": info.get("code", ticker),
            "名稱": info.get("name", ""),
            "市場": info.get("market", ""),
            "現價": round(current_price, 2),
            "55天高點": round(high_55d, 2),
            "距高點%": round(ratio * 100, 2),
            "screened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).sort_values("距高點%", ascending=False).reset_index(drop=True)
    return result


def print_table(df: pd.DataFrame) -> None:
    """在 Console 顯示結果表格。"""
    display_cols = ["代號", "名稱", "市場", "現價", "55天高點", "距高點%"]
    print("\n" + "=" * 60)
    print(f"  篩選結果：共 {len(df)} 檔股票")
    print("=" * 60)
    print(tabulate(df[display_cols], headers="keys", tablefmt="simple", index=False))


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """儲存結果為 CSV。"""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n結果已儲存至：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="台股 55 天高點篩選器（海龜交易法）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python screen_55day_high.py
  python screen_55day_high.py --threshold 0.98
  python screen_55day_high.py --markets twse
  python screen_55day_high.py --threshold 0.95 --output my_results.csv
  python screen_55day_high.py --source twstock
        """,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="篩選門檻：現價 / 55天高點 >= 此值（預設 0.95，即 95%%）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV 輸出路徑（預設：results_YYYYMMDD.csv）",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["twse", "tpex"],
        default=["twse", "tpex"],
        help="篩選市場：twse（上市）、tpex（上櫃），預設兩者皆選",
    )
    parser.add_argument(
        "--source",
        choices=["isin", "twstock"],
        default="isin",
        help="股票清單來源：isin（預設，爬 isin.twse.com.tw）或 twstock（使用 twstock 函式庫）",
    )

    args = parser.parse_args()

    if not (0 < args.threshold <= 1.0):
        print("錯誤：threshold 需介於 0 到 1 之間", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or f"results_{datetime.now().strftime('%Y%m%d')}.csv"

    print(f"台股 55 天高點篩選器")
    print(f"  市場：{', '.join(args.markets).upper()}")
    print(f"  門檻：現價 >= 55天高點 × {args.threshold * 100:.1f}%")
    print(f"  來源：{args.source}")
    print(f"  輸出：{output_path}\n")

    results = screen_stocks(threshold=args.threshold, markets=args.markets, source=args.source)

    if results.empty:
        print("\n沒有符合條件的股票。")
    else:
        print_table(results)
        save_csv(results, output_path)


if __name__ == "__main__":
    main()
