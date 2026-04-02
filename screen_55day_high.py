#!/usr/bin/env python3
"""
台股 55 天高點篩選器 (Turtle Trading System)

篩選台股（上市/上櫃）中當前收盤價接近 55 天最高點的股票。
使用方式:
    python screen_55day_high.py [--threshold 0.95] [--output results.csv] [--markets twse tpex]
"""

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import twstock
import yfinance as yf
from tabulate import tabulate
from tqdm import tqdm

BATCH_SIZE = 100
BATCH_SLEEP = 1.5  # seconds between yfinance batches
CACHE_PATH = Path(".prices_cache.pkl")

_TWSTOCK_MARKET_MAP = {
    "上市": ("TWSE", ".TW"),
    "上櫃": ("TPEX", ".TWO"),
}


def get_stock_list(markets: list[str]) -> pd.DataFrame:
    """使用 twstock 取得上市/上櫃股票清單。

    Returns:
        DataFrame with columns: code, name, market, yf_ticker
    """
    print("取得股票清單（twstock）...")
    want_twse = "twse" in markets
    want_tpex = "tpex" in markets

    rows = []
    for code, info in twstock.codes.items():
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
            "industry": getattr(info, "group", ""),
            "market": market_name,
            "yf_ticker": f"{code}{suffix}",
        })

    if not rows:
        print("錯誤：無法取得任何股票清單", file=sys.stderr)
        sys.exit(1)

    result = pd.DataFrame(rows).drop_duplicates(subset="code").reset_index(drop=True)
    print(f"共取得 {len(result)} 檔股票（{', '.join(markets).upper()}）")
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


def calculate_signal(df: pd.DataFrame, period: int) -> tuple | None:
    """計算 N 天高點訊號。

    Args:
        df:     DataFrame with columns High, Low, Close（已按日期排序）
        period: 回顧天數（例如 20 或 55）

    Returns:
        (current_price, high_nd, ratio, high_nd_date, ma5, ma10, ma20, atr20, recent_high)
        或 None（資料不足 period 天）
    """
    if len(df) < period:
        return None

    window = df.tail(period)
    high_nd = float(window["High"].max())
    current_price = float(df["Close"].iloc[-1])

    if high_nd <= 0:
        return None

    ratio = current_price / high_nd
    high_nd_date = window["High"].idxmax().strftime("%Y-%m-%d")
    ma5  = round(float(df["Close"].tail(5).mean()),  2)
    ma10 = round(float(df["Close"].tail(10).mean()), 2)
    ma20 = round(float(df["Close"].tail(20).mean()), 2)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr20 = round(float(tr.tail(20).mean()), 2)
    last_10_dates = set(df.index[-10:].strftime("%Y-%m-%d"))
    recent_high = high_nd_date in last_10_dates
    return current_price, high_nd, ratio, high_nd_date, ma5, ma10, ma20, atr20, recent_high


def _load_cache() -> dict | None:
    """載入當日價格快取，若不存在或已過期則回傳 None。"""
    if not CACHE_PATH.exists():
        return None
    cached = pickle.loads(CACHE_PATH.read_bytes())
    if cached.get("date") != datetime.now().strftime("%Y-%m-%d"):
        return None
    prices = cached["prices"]
    # 若快取缺少必要欄位（舊格式），強制重新下載
    if prices and not {"High", "Low", "Close", "Volume"}.issubset(next(iter(prices.values())).columns):
        return None
    print("使用今日快取股價資料。")
    return prices


def _save_cache(prices: dict) -> None:
    """將價格資料寫入快取檔。"""
    CACHE_PATH.write_bytes(pickle.dumps({"date": datetime.now().strftime("%Y-%m-%d"), "prices": prices}))


def screen_stocks(threshold: float, markets: list[str], period: int = 55) -> pd.DataFrame:
    """主流程：取清單 → 批次抓價格 → 計算 N 天高點 → 篩選 → 排序。"""
    stock_list = get_stock_list(markets)
    tickers = stock_list["yf_ticker"].tolist()

    prices = _load_cache()
    if prices is None:
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
                            low = raw["Low"][ticker].dropna()
                            close = raw["Close"][ticker].dropna()
                            volume = raw["Volume"][ticker].dropna()
                            if len(high) > 0 and len(close) > 0:
                                prices[ticker] = pd.DataFrame({"High": high, "Low": low, "Close": close, "Volume": volume})
                        except KeyError:
                            pass
                else:
                    ticker = batch[0]
                    if "High" in raw.columns and "Close" in raw.columns:
                        prices[ticker] = raw[["High", "Low", "Close", "Volume"]].dropna()

            except Exception as e:
                tqdm.write(f"警告：批次 {i // BATCH_SIZE + 1} 下載失敗 - {e}")

            if i + BATCH_SIZE < len(tickers):
                time.sleep(BATCH_SLEEP)

        _save_cache(prices)

    high_col = f"{period}天高點"
    print(f"\n計算 {period} 天高點訊號（門檻：{threshold * 100:.1f}%）...")
    records = []
    ticker_to_info = stock_list.set_index("yf_ticker").to_dict("index")
    today = datetime.now().strftime("%Y-%m-%d")

    for ticker, df in prices.items():
        signal = calculate_signal(df, period)
        if signal is None:
            continue
        current_price, high_nd, ratio, high_nd_date, ma5, ma10, ma20, atr20, recent_high = signal
        if ratio < threshold or ratio >= 1.0 or high_nd_date == today:
            continue

        # 20日模式：排除同時符合55日高點門檻的股票
        if period == 20 and len(df) >= 55:
            high_55 = float(df.tail(55)["High"].max())
            if high_55 > 0 and current_price / high_55 >= threshold:
                continue

        info = ticker_to_info.get(ticker, {})
        records.append({
            "代號": info.get("code", ticker),
            "名稱": info.get("name", ""),
            "產業": info.get("industry", ""),
            "市場": info.get("market", ""),
            "現價": round(current_price, 2),
            high_col: round(high_nd, 2),
            "高點日期": high_nd_date,
            "距高點%": round(ratio * 100, 2),
            "MA5":   ma5,
            "MA10":  ma10,
            "MA20":  ma20,
            "ATR20": atr20,
            "成交量": int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0,
            "趨勢": "多頭排列" if current_price > ma5 > ma10 > ma20 else "",
            "近況": "十天創高中" if recent_high else "",
            "入手價":    round(high_nd, 2),
            "停損價":    round(high_nd - 2 * atr20, 2),
            "第一次加碼": round(high_nd + 0.5 * atr20, 2),
            "第二次加碼": round(high_nd + 1.0 * atr20, 2),
            "第三次加碼": round(high_nd + 1.5 * atr20, 2),
            "screened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).sort_values(["產業", "距高點%"], ascending=[True, False]).reset_index(drop=True)
    return result


def print_table(df: pd.DataFrame, period: int = 55) -> None:
    """在 Console 顯示結果表格。"""
    high_col = f"{period}天高點"
    display_cols = ["代號", "名稱", "產業", "市場", "現價", "MA5", "MA10", "MA20", "ATR20", high_col, "高點日期", "距高點%", "成交量", "趨勢", "近況", "入手價", "停損價", "第一次加碼", "第二次加碼", "第三次加碼"]
    print("\n" + "=" * 60)
    print(f"  篩選結果：共 {len(df)} 檔股票")
    print("=" * 60)
    print(tabulate(df[display_cols], headers="keys", tablefmt="pipe", showindex=False))


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """儲存結果為 CSV。"""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n結果已儲存至：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="台股高點篩選器（海龜交易法）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python screen_55day_high.py
  python screen_55day_high.py --period 20
  python screen_55day_high.py --threshold 0.98
  python screen_55day_high.py --markets twse
  python screen_55day_high.py --threshold 0.95 --output my_results.csv
        """,
    )
    parser.add_argument(
        "--period",
        type=int,
        choices=[20, 55],
        default=55,
        help="高點回顧天數：55（預設）或 20。20日模式會自動排除同時符合55日門檻的股票",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="篩選門檻：現價 / N天高點 >= 此值（預設 0.95，即 95%%）",
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

    args = parser.parse_args()

    if not (0 < args.threshold <= 1.0):
        print("錯誤：threshold 需介於 0 到 1 之間", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or f"results_{args.period}d_{datetime.now().strftime('%Y%m%d')}.csv"

    print(f"台股 {args.period} 天高點篩選器")
    print(f"  市場：{', '.join(args.markets).upper()}")
    print(f"  門檻：現價 >= {args.period}天高點 × {args.threshold * 100:.1f}%")
    if args.period == 20:
        print(f"  模式：排除同時符合55日高點門檻的股票")
    print(f"  輸出：{output_path}\n")

    results = screen_stocks(threshold=args.threshold, markets=args.markets, period=args.period)

    if results.empty:
        print("\n沒有符合條件的股票。")
    else:
        print_table(results, period=args.period)
        save_csv(results, output_path)


if __name__ == "__main__":
    main()
