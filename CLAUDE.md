# Project Rules

## Python: tabulate() 用法

使用 `tabulate` 函式庫的 `tabulate()` 時，隱藏索引的正確參數是 `showindex=False`，**不是** `index=False`。

```python
# 正確
print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))

# 錯誤 — 會拋出 TypeError: tabulate() got an unexpected keyword argument 'index'
print(tabulate(df, headers="keys", tablefmt="simple", index=False))
```
