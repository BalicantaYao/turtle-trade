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

def ticker_to_yf(ticker: str) -> str:
    """將台股代碼轉換為 yfinance 格式（4 位數字自動加 .TW）。"""
    if ticker.isdigit() and len(ticker) == 4:
        return f"{ticker}.TW"
    return ticker


def fetch_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("請安裝 yfinance: pip install yfinance")

    yf_ticker = ticker_to_yf(ticker)
    print(f"下載 {ticker}（{yf_ticker}）歷史資料：{start} ~ {end}")
    try:
        raw = yf.download(
            yf_ticker,
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
        close = raw["Close"][yf_ticker].dropna()
        volume = raw["Volume"][yf_ticker].dropna()
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


def fetch_data(ticker: str, csv_path: str | None, start: str, end: str) -> pd.DataFrame:
    # 決定預設 CSV 快取路徑（與代碼同名）
    default_csv = f"{ticker.replace('^', '').replace('.', '_')}.csv"

    if csv_path and Path(csv_path).exists():
        df = load_from_csv(csv_path)
        df = df.loc[start:end]
    else:
        if csv_path:
            print(f"找不到 {csv_path}，改為自動下載...")
        df = fetch_from_yfinance(ticker, start, end)
        save_path = csv_path or default_csv
        df.to_csv(save_path, index=True, encoding="utf-8-sig")
        print(f"資料已儲存至 {save_path}，下次可直接使用 --csv {save_path}")

    print(f"共 {len(df)} 個交易日（{df.index[0].date()} ~ {df.index[-1].date()}）\n")

    close_label = "收盤指數" if ticker in ("^TWII", "大盤") else "收盤價"
    print(f"【最近 10 個交易日資料】")
    recent = df.tail(10).copy()
    recent.index = recent.index.strftime("%Y-%m-%d")
    recent["Close"] = recent["Close"].map(lambda x: f"{x:,.2f}")
    recent["Volume"] = recent["Volume"].map(lambda x: f"{x:,.0f}")
    recent.columns = [close_label, "成交量"]
    print(recent.to_string())
    print()

    return df


def run_backtest(df: pd.DataFrame) -> dict:
    """
    創新高定義  ：當日收盤 > 歷史最高收盤（cummax，不含當日）
    量縮定義    ：當日成交量 <= 前一個交易日成交量
    """
    close = df["Close"]
    volume = df["Volume"]

    # 收盤創歷史新高（累積最大值，不含當日）
    hist_max_close = close.shift(1).cummax()

    new_high = close > hist_max_close           # 收盤創歷史新高
    volume_shrink = volume <= volume.shift(1)   # 量縮：當日量 <= 前一天
    volume_expand = volume > volume.shift(1)    # 量增：當日量 > 前一天

    signal_shrink = new_high & volume_shrink    # 創新高 + 量縮
    signal_expand = new_high & volume_expand    # 創新高 + 量增

    # 隔日報酬
    next_day_return = close.pct_change().shift(-1)

    # 相較前一天的成交量落差（%）
    volume_chg_pct = (volume / volume.shift(1) - 1) * 100

    def calc_stats(signal: pd.Series) -> dict:
        returns = next_day_return[signal].dropna()
        vol_chg = volume_chg_pct[signal].reindex(returns.index)
        n = len(returns)
        down = (returns < 0).sum()
        up = (returns > 0).sum()
        return {
            "total": n,
            "down_count": int(down),
            "up_count": int(up),
            "flat_count": int(n - down - up),
            "down_pct": down / n * 100 if n > 0 else 0,
            "up_pct": up / n * 100 if n > 0 else 0,
            "avg_next_return_pct": float(returns.mean() * 100) if n > 0 else 0,
            "median_next_return_pct": float(returns.median() * 100) if n > 0 else 0,
            "signal_dates": df.index[signal],
            "returns": returns,
            "vol_chg_pct": vol_chg,
        }

    # 創新高（不論量）基準
    all_new_high_returns = next_day_return[new_high].dropna()
    all_new_high_down_pct = (all_new_high_returns < 0).mean() * 100 if len(all_new_high_returns) > 0 else 0

    return {
        "shrink": calc_stats(signal_shrink),
        "expand": calc_stats(signal_expand),
        "all_new_high_count": len(all_new_high_returns),
        "all_new_high_down_pct": all_new_high_down_pct,
    }


def print_section(label: str, stats: dict, baseline_down: float, all_new_high_count: int, all_new_high_down_pct: float) -> None:
    s = stats
    print(f"  【統計結果】")
    print(f"  滿足條件的交易日：{s['total']} 次")
    print()
    print(f"  隔日下跌：{s['down_count']:>4} 次  {s['down_pct']:>6.1f}%")
    print(f"  隔日上漲：{s['up_count']:>4} 次  {s['up_pct']:>6.1f}%")
    print(f"  隔日平盤：{s['flat_count']:>4} 次")
    print()
    print(f"  隔日平均報酬：{s['avg_next_return_pct']:+.3f}%")
    print(f"  隔日中位數報酬：{s['median_next_return_pct']:+.3f}%")
    print()
    print(f"  【與基準比較】")
    print(f"  所有交易日 隔日下跌機率：{baseline_down:.1f}%  （基準）")
    print(f"  創新高（任意量）隔日下跌：{all_new_high_down_pct:.1f}%  （共 {all_new_high_count} 次）")
    print(f"  {label} 隔日下跌：{s['down_pct']:.1f}%  ← 比基準 {s['down_pct'] - baseline_down:+.1f} 個百分點")
    print()
    recent = s["signal_dates"][-10:]
    recent_returns = s["returns"].reindex(recent).dropna()
    if len(recent) > 0:
        print(f"  【最近 {len(recent)} 筆訊號】")
        for dt in recent:
            r = recent_returns.get(dt)
            if r is not None:
                arrow = "↓" if r < 0 else "↑" if r > 0 else "─"
                print(f"    {dt.date()}  隔日 {arrow} {r*100:+.2f}%")


def print_results(result: dict, df: pd.DataFrame, ticker: str) -> None:
    all_returns = df["Close"].pct_change().shift(-1).dropna()
    baseline_down = (all_returns < 0).mean() * 100
    nh_count = result["all_new_high_count"]
    nh_down = result["all_new_high_down_pct"]

    # ── 量縮 ──
    print("=" * 65)
    print(f"  {ticker}｜創歷史新高 + 量縮 → 隔日機率")
    print(f"  ① 當日收盤創歷史新高  ② 當日量 <= 前一日量")
    print("=" * 65)
    print_section("創新高+量縮", result["shrink"], baseline_down, nh_count, nh_down)
    print("=" * 65)

    # 量縮落差分布
    print_volume_distribution(result["shrink"], mode="shrink")

    # ── 量增 ──
    print()
    print("=" * 65)
    print(f"  {ticker}｜創歷史新高 + 量增 → 隔日機率")
    print(f"  ① 當日收盤創歷史新高  ② 當日量 > 前一日量")
    print("=" * 65)
    print_section("創新高+量增", result["expand"], baseline_down, nh_count, nh_down)
    print("=" * 65)

    # 量增落差分布
    print_volume_distribution(result["expand"], mode="expand")


def print_volume_distribution(stats: dict, mode: str) -> None:
    """成交量相較前一天落差（%）的分布，並區分隔日上漲 vs 下跌。"""
    vol_chg = stats["vol_chg_pct"]
    returns = stats["returns"]

    vol_up = vol_chg[returns > 0].dropna()
    vol_down = vol_chg[returns < 0].dropna()

    if mode == "shrink":
        bins = [-100, -50, -30, -20, -10, -5, 0, 0.001]
        col_labels = ["-100~-50%", "-50~-30%", "-30~-20%", "-20~-10%", "-10~-5%", "-5~0%", "0%"]
    else:
        bins = [0, 5, 10, 20, 30, 50, 100, float("inf")]
        col_labels = ["0~5%", "5~10%", "10~20%", "20~30%", "30~50%", "50~100%", ">100%"]

    print()
    title = "量縮" if mode == "shrink" else "量增"
    print(f"【{title}訊號日：成交量 vs 前日落差分布】")
    print(f"  落差 % = (當日量 - 前日量) / 前日量 × 100")
    print()

    for label, series in [("隔日上漲", vol_up), ("隔日下跌", vol_down), ("全部訊號", vol_chg.dropna())]:
        if series.empty:
            continue
        cuts = pd.cut(series, bins=bins, right=True)
        counts = cuts.value_counts(sort=False)
        pcts = (counts / len(series) * 100).round(1)
        count_str = "  ".join(f"{c:>4}({p:>4.1f}%)" for c, p in zip(counts, pcts))
        print(f"  {label:8}  {len(series):>4}  {count_str}")

    print()
    print("  【各落差區間的平均隔日報酬】")
    analysis = pd.DataFrame({"vol_chg": vol_chg, "next_ret": returns}).dropna()
    analysis["區間"] = pd.cut(analysis["vol_chg"], bins=bins, right=True)
    summary = analysis.groupby("區間", observed=True)["next_ret"].agg(
        次數="count",
        下跌次數=lambda x: (x < 0).sum(),
        下跌率=lambda x: f"{(x < 0).mean()*100:.1f}%",
        平均報酬=lambda x: f"{x.mean()*100:+.3f}%",
    )
    print(summary.to_string())
    print()


def main():
    parser = argparse.ArgumentParser(
        description="台股創新高但量縮時隔日下跌機率回測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python backtest_taiex_volume_divergence.py                        # 預設大盤
  python backtest_taiex_volume_divergence.py --ticker 2330          # 台積電
  python backtest_taiex_volume_divergence.py --ticker ^TWII --csv taiex.csv

取得歷史資料（CSV）：
  Yahoo Finance → https://finance.yahoo.com/quote/2330.TW/history/
  下載後用 --csv 參數指定路徑
        """,
    )
    parser.add_argument("--ticker", type=str, default="^TWII",
                        help="標的代碼，台股 4 碼會自動加 .TW（預設 ^TWII 大盤）")
    parser.add_argument("--csv", type=str, default=None,
                        help="本機 CSV 資料路徑（Date,Close,Volume 格式）")
    parser.add_argument("--start", type=str, default="1995-01-01",
                        help="回測起始日期（預設 1995-01-01）")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"),
                        help="回測結束日期（預設今天）")
    args = parser.parse_args()

    try:
        df = fetch_data(args.ticker, args.csv, args.start, args.end)
    except (RuntimeError, FileNotFoundError) as e:
        yf_ticker = ticker_to_yf(args.ticker)
        print(f"\n錯誤：{e}", file=sys.stderr)
        print(f"\n請手動下載 CSV 後用 --csv 指定：", file=sys.stderr)
        print(f"  Yahoo Finance: https://finance.yahoo.com/quote/{yf_ticker}/history/", file=sys.stderr)
        sys.exit(1)

    result = run_backtest(df)
    print_results(result, df, args.ticker)


if __name__ == "__main__":
    main()
