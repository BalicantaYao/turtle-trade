#!/usr/bin/env python3
"""
台股大盤創新高但量縮隔日下跌機率回測

回測條件：
  1. 當日收盤 創 N 日新高（收盤超越過去 N 個交易日最高收盤）
  2. 當日成交量 未創 N 日新高（成交量 <= 過去 N 日最大量）

分析：滿足以上條件時，隔日下跌的機率是多少。

資料來源（擇一）：
  A. 本機 CSV 檔（--csv 參數指定路徑）
     格式：Date,Close,Volume（逗號分隔，Date 格式 YYYY-MM-DD）
     可從 Yahoo Finance 或 TWSE 手動下載
  B. yfinance 自動下載（需網路連線）

取得資料方式：
  ① Yahoo Finance：https://finance.yahoo.com/quote/%5ETWII/history/
     下載後指定 --csv 路徑即可
  ② TWSE API：https://www.twse.com.tw/zh/trading/historical/fmtqik.html

用法：
  python backtest_taiex_volume_divergence.py --csv taiex.csv
  python backtest_taiex_volume_divergence.py --csv taiex.csv --lookback 55
  python backtest_taiex_volume_divergence.py --csv taiex.csv --compare
  python backtest_taiex_volume_divergence.py  # 嘗試 yfinance（需網路）
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

TAIEX_TICKER = "^TWII"


def fetch_from_yfinance(start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("請安裝 yfinance: pip install yfinance")

    print(f"下載台股大盤（{TAIEX_TICKER}）歷史資料：{start} ~ {end}")
    try:
        raw = yf.download(
            TAIEX_TICKER,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        raise RuntimeError(f"yfinance 下載失敗：{e}，請改用 --csv 選項提供本機資料")

    if raw.empty:
        raise RuntimeError("yfinance 回傳空資料，請改用 --csv 選項提供本機資料")

    # 單檔下載仍可能產生 MultiIndex（與 screen_55day_high.py 處理方式一致）
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][TAIEX_TICKER].dropna()
        volume = raw["Volume"][TAIEX_TICKER].dropna()
        df = pd.DataFrame({"Close": close, "Volume": volume}).dropna()
    else:
        df = raw[["Close", "Volume"]].dropna()

    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_from_csv(path: str) -> pd.DataFrame:
    """
    支援以下 CSV 格式：
    - Yahoo Finance 格式：Date,Open,High,Low,Close,Adj Close,Volume
    - 精簡格式：Date,Close,Volume
    - TWSE 格式（自動偵測）
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    print(f"讀取本機資料：{path}")

    # 嘗試自動偵測分隔符
    raw = p.read_text(encoding="utf-8-sig")
    sep = "," if raw.count(",") > raw.count("\t") else "\t"

    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # 找出日期欄
    date_col = next((c for c in df.columns if "date" in c.lower() or "日期" in c), None)
    if date_col is None:
        raise ValueError(f"找不到日期欄位，現有欄位：{list(df.columns)}")

    # 找出收盤欄
    close_col = next((c for c in df.columns if c.lower() in ("close", "adj close", "收盤", "收盤價")), None)
    if close_col is None:
        raise ValueError(f"找不到收盤價欄位，現有欄位：{list(df.columns)}")

    # 找出成交量欄
    vol_col = next((c for c in df.columns if c.lower() in ("volume", "成交量", "成交股數", "成交金額")), None)
    if vol_col is None:
        raise ValueError(f"找不到成交量欄位，現有欄位：{list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[close_col] = pd.to_numeric(df[close_col].astype(str).str.replace(",", ""), errors="coerce")
    df[vol_col] = pd.to_numeric(df[vol_col].astype(str).str.replace(",", ""), errors="coerce")

    result = df[[date_col, close_col, vol_col]].copy()
    result.columns = ["Date", "Close", "Volume"]
    result = result.dropna().set_index("Date").sort_index()
    return result


def fetch_data(csv_path: str | None, start: str, end: str) -> pd.DataFrame:
    if csv_path:
        df = load_from_csv(csv_path)
        # 過濾日期範圍
        df = df.loc[start:end]
    else:
        df = fetch_from_yfinance(start, end)

    print(f"共 {len(df)} 個交易日（{df.index[0].date()} ~ {df.index[-1].date()}）\n")

    print("【最近 10 個交易日資料】")
    recent = df.tail(10).copy()
    recent.index = recent.index.strftime("%Y-%m-%d")
    recent["Close"] = recent["Close"].map(lambda x: f"{x:,.2f}")
    recent["Volume"] = recent["Volume"].map(lambda x: f"{x:,.0f}")
    recent.columns = ["收盤指數", "成交量"]
    print(recent.to_string())
    print()

    return df


def run_backtest(df: pd.DataFrame) -> dict:
    """
    創新高定義  ：當日收盤 > 歷史最高收盤（cummax，不含當日）
    量縮定義    ：當日成交量 <= 前 volume_lookback 個交易日的最大成交量
    """
    close = df["Close"]
    volume = df["Volume"]

    # 收盤創歷史新高（累積最大值，不含當日）
    hist_max_close = close.shift(1).cummax()

    new_high = close > hist_max_close        # 條件①：收盤創歷史新高
    volume_not_max = volume <= volume.shift(1)  # 條件②：量未超過前一天（量縮）

    signal = new_high & volume_not_max

    # 隔日報酬（+1 代表下一個交易日）
    next_day_return = close.pct_change().shift(-1)

    returns_on_signal = next_day_return[signal].dropna()
    total = len(returns_on_signal)
    down = (returns_on_signal < 0).sum()
    up = (returns_on_signal > 0).sum()
    flat = (returns_on_signal == 0).sum()

    # 另外統計：創新高（不論量）時的隔日表現
    all_new_high_returns = next_day_return[new_high].dropna()
    all_new_high_down_pct = (all_new_high_returns < 0).mean() * 100 if len(all_new_high_returns) > 0 else 0

    return {
        "total_signals": total,
        "down_count": int(down),
        "up_count": int(up),
        "flat_count": int(flat),
        "down_pct": down / total * 100 if total > 0 else 0,
        "up_pct": up / total * 100 if total > 0 else 0,
        "avg_next_return_pct": float(returns_on_signal.mean() * 100) if total > 0 else 0,
        "median_next_return_pct": float(returns_on_signal.median() * 100) if total > 0 else 0,
        "signal_dates": df.index[signal],
        "returns": returns_on_signal,
        "all_new_high_count": len(all_new_high_returns),
        "all_new_high_down_pct": all_new_high_down_pct,
    }


def print_results(result: dict, df: pd.DataFrame) -> None:
    all_returns = df["Close"].pct_change().shift(-1).dropna()
    baseline_down = (all_returns < 0).mean() * 100

    print("=" * 65)
    print(f"  台股大盤收盤創歷史新高 + 量縮 → 隔日下跌機率回測")
    print("=" * 65)
    print()
    print("  【回測條件】")
    print(f"  ① 當日收盤 創歷史新高（超越所有歷史收盤最高價）")
    print(f"  ② 當日成交量 未超過前一個交易日成交量（量縮）")
    print()
    print("  【統計結果】")
    print(f"  滿足雙條件的交易日：{result['total_signals']} 次")
    print()
    print(f"  隔日下跌：{result['down_count']:>4} 次  {result['down_pct']:>6.1f}%  ← 目標機率")
    print(f"  隔日上漲：{result['up_count']:>4} 次  {result['up_pct']:>6.1f}%")
    print(f"  隔日平盤：{result['flat_count']:>4} 次")
    print()
    print(f"  隔日平均報酬：{result['avg_next_return_pct']:+.3f}%")
    print(f"  隔日中位數報酬：{result['median_next_return_pct']:+.3f}%")
    print()
    print("  【與基準比較】")
    print(f"  所有交易日 隔日下跌機率：{baseline_down:.1f}%  （基準）")
    print(f"  創新高（任意量）隔日下跌：{result['all_new_high_down_pct']:.1f}%  "
          f"（共 {result['all_new_high_count']} 次）")
    print(f"  創新高 + 量縮 隔日下跌：{result['down_pct']:.1f}%  "
          f"← 比基準 {result['down_pct'] - baseline_down:+.1f} 個百分點")
    print()

    # 最近 10 筆訊號
    recent = result["signal_dates"][-10:]
    recent_returns = result["returns"].reindex(recent).dropna()
    if len(recent) > 0:
        print(f"  【最近 {len(recent)} 筆訊號】")
        for dt in recent:
            r = recent_returns.get(dt)
            if r is not None:
                arrow = "↓" if r < 0 else "↑" if r > 0 else "─"
                print(f"    {dt.date()}  隔日 {arrow} {r*100:+.2f}%")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="台股大盤創新高但量縮時隔日下跌機率回測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python backtest_taiex_volume_divergence.py --csv taiex.csv
  python backtest_taiex_volume_divergence.py --csv taiex.csv --start 2010-01-01

取得大盤歷史資料（CSV）：
  Yahoo Finance → https://finance.yahoo.com/quote/%5ETWII/history/
  下載後用 --csv 參數指定路徑
        """,
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="本機 CSV 資料路徑（Date,Close,Volume 格式）")
    parser.add_argument("--start", type=str, default="1995-01-01",
                        help="回測起始日期（預設 1995-01-01）")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"),
                        help="回測結束日期（預設今天）")
    args = parser.parse_args()

    try:
        df = fetch_data(args.csv, args.start, args.end)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n錯誤：{e}", file=sys.stderr)
        print("\n請提供本機 CSV 資料：", file=sys.stderr)
        print("  1. 前往 https://finance.yahoo.com/quote/%5ETWII/history/", file=sys.stderr)
        print("  2. 下載歷史資料為 CSV", file=sys.stderr)
        print("  3. 執行：python backtest_taiex_volume_divergence.py --csv <檔案路徑>", file=sys.stderr)
        sys.exit(1)

    result = run_backtest(df)
    print_results(result, df)


if __name__ == "__main__":
    main()
