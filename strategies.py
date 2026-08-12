import pandas as pd
returns = pd.read_csv("data/weekly_returns.csv", index_col=0)
print(returns["CELH"].describe())
print(returns["CELH"].abs().sort_values(ascending=False).head(5))