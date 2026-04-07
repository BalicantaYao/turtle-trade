# Project Rules

## Python: tabulate() 用法

使用 `tabulate` 函式庫的 `tabulate()` 時，隱藏索引的正確參數是 `showindex=False`，**不是** `index=False`。

```python
# 正確
print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))

# 錯誤 — 會拋出 TypeError: tabulate() got an unexpected keyword argument 'index'
print(tabulate(df, headers="keys", tablefmt="simple", index=False))
```

## 股價資料快取規則

每支篩選器都必須對下載的歷史股價做**當日快取**，避免同一天重複下載：

- 快取目錄：`.cache/`（已加入 `.gitignore`）
- 快取檔名格式：`prices_YYYYMMDD_<markets>.pkl`（pickle 格式）
- cache key 包含**日期 + 排序後的市場清單**，不同市場組合分開快取
- 流程：先呼叫 `load_cache()`；有快取就直接用，沒有才下載，下載完呼叫 `save_cache()`

```python
def _cache_path(markets): ...  # 產生 .cache/prices_YYYYMMDD_key.pkl
def load_cache(markets): ...   # 存在則 pickle.load，否則回傳 None
def save_cache(prices, markets): ...  # CACHE_DIR.mkdir + pickle.dump

# 在主流程中：
prices = load_cache(markets)
if prices is None:
    prices = fetch_prices_batch(tickers)
    save_cache(prices, markets)
```
