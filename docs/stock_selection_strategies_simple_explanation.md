# How My New Stock-Selection Strategies Work

## Why I created these strategies

The main idea behind this project is to test whether companies that help people cope with everyday life can also produce strong investment returns. This includes companies connected to food, drinks, entertainment, technology, tobacco, restaurants, and other products that people regularly use.

I did not choose CELH because I knew it had high returns. It was part of the thematic stock list before I knew what its historical performance looked like.

The original portfolio contained 30 stocks. I was able to build acceptable point-in-time fundamental data for 25 of them at every annual rebalance. I then tested two different ways to select 12 stocks from the eligible group:

1. **Partitioning Selection**, which uses PAM clustering.
2. **Density Selection**, which uses HDBSCAN clustering.

The purpose of clustering is to avoid simply choosing 12 stocks that all behave in the same way.

## The strategy in one sentence

Every January, I use only the information available at that time, group similar stocks together, select 12 stocks with a balance of value, historical risk-adjusted returns, and diversification, and then equally weight them for the next holding period.

## Step 1: Only use information available before the rebalance

The backtest rebalances once per year from 2020 through 2025. At each January rebalance, the strategy is not allowed to see future information.

For example, the portfolio selected on January 1, 2020 can use:

- financial statements filed before January 1, 2020;
- historical prices from before January 1, 2020; and
- the previous 104 weeks of returns.

It cannot use earnings, prices, or returns that happened later in 2020.

This is the walk-forward part of the experiment. The strategy moves through time one rebalance at a time instead of using the full dataset to make every decision.

## Step 2: Create three simple ranks for each stock

I convert the raw information into ranks for each stock. The market capitalization, P/E ratios, and Sharpe ratios use very different scales. A rank near 1 means the stock ranks highly compared with the other eligible stocks at that rebalance.

### Value rank

The value rank is mainly based on the trailing P/E ratio. Among profitable companies, a lower positive P/E ratio receives a better rank. A company with negative earnings receives a value rank of zero instead of being given a fake or imputed P/E ratio.

This feature asks:

> Is the stock relatively inexpensive compared with its recent earnings?

### Size rank

The size rank is based on the logarithm of point-in-time market capitalization. Larger companies receive higher ranks.

Market capitalization is reconstructed using a historical unadjusted share price and a historical share count. Current market capitalization is not copied backward into previous years.

This feature is used when measuring how similar companies are. It is not directly included in the final value-and-Sharpe selection score.

### Trailing Sharpe rank

I calculate a trailing Sharpe ratio from the previous 104 weeks of returns using the frozen 4% annual risk-free rate. Stocks with better historical return relative to volatility receive higher ranks.

This feature asks:

> Has the stock produced strong recent returns without taking an unreasonable amount of risk?

A high trailing Sharpe rank does not guarantee that the stock will perform well next year. It only summarizes information that was already known at the rebalance.

## Step 3: Measure how different the stocks are

The strategies need a distance between each pair of stocks before they can create clusters. I use two equally weighted parts:

1. **50% return-correlation distance.** Stocks that moved together are considered more similar.
2. **50% feature distance.** Stocks with similar value, size, and Sharpe ranks are considered more similar.

This means two stocks can be considered similar because their returns moved together, because their financial characteristics were similar, or because of a mixture of both.

## Step 4: Give each stock a basic selection score

The basic score is:

```text
50% value rank + 50% trailing Sharpe rank
```

A stock therefore needs a useful combination of valuation and historical risk-adjusted performance. Market size affects the clustering distance, but it does not directly increase this basic score.

## Strategy 1: Partitioning Selection using PAM

PAM stands for **Partitioning Around Medoids**. It is similar to k-means, but the center of each group is an actual stock instead of an imaginary average point.

I tell PAM to divide the eligible stocks into six groups. The process is:

1. Build the stock-to-stock distance matrix.
2. Split the stocks into six clusters.
3. Rank the stocks inside each cluster using the selection score.
4. Take up to the two highest-scoring stocks from each cluster.
5. If fewer than 12 stocks have been selected, fill the remaining spaces using the diversification rule.

The main benefit is that the strategy tries to take representatives from different types of companies instead of allowing one cluster to dominate the portfolio.

## Strategy 2: Density Selection using HDBSCAN

HDBSCAN stands for **Hierarchical Density-Based Spatial Clustering of Applications with Noise**. The name sounds complicated, but the basic idea is simple: it looks for natural groups of stocks that are close together.

Unlike PAM, HDBSCAN does not need every stock to fit neatly into one of six fixed groups. It can identify dense groups and label unusual stocks as noise or outliers.

The process is:

1. Build the same stock-to-stock distance matrix.
2. Use HDBSCAN to find natural clusters with at least three nearby members under the frozen settings.
3. Take the highest-scoring representative from each normal cluster.
4. Fill the remaining portfolio spaces until 12 stocks are selected.
5. Noise stocks are not automatic representatives, but they can still be selected during the diversified filling stage if their score and correlation are attractive.

This last point is important. HDBSCAN does not automatically reject every unusual company. An unusual stock can still improve the portfolio.

## The diversification rule

When either strategy needs to fill an empty position, it does not only choose the next stock with the highest basic score. It subtracts a penalty when a candidate is highly correlated with stocks already selected:

```text
adjusted score = basic score - 0.25 × average correlation with selected stocks
```

This encourages the strategy to choose a stock that adds something different to the portfolio.

## Step 5: Equally weight the 12 selected stocks

After selection, every chosen stock receives approximately:

```text
1 / 12 = 8.33%
```

No stock reached the 25% maximum position cap. The experiment is mainly testing the stock-selection method, not a complicated weighting method.

Equal weighting also makes the comparison easier to understand. If performance changes, it is mainly because the strategies selected different stocks rather than because an optimizer gave one stock an extremely large allocation.

## Step 6: Hold the portfolio and move forward

After the January selection:

1. The 12-stock portfolio is held during the next period.
2. Weekly returns are recorded.
3. A 10-basis-point transaction cost is applied based on portfolio turnover.
4. At the next January rebalance, all features and clusters are rebuilt using only the information available at that new date.

The strategy does not keep a stock forever just because it performed well once.

## The CELH example

CELH shows that the strategies were making changing walk-forward decisions rather than permanently holding the eventual winner.

| Rebalance year | Partitioning Selection | Density Selection |
| -------------- | ---------------------: | ----------------: |
| 2020           |           Not selected |          Selected |
| 2021           |               Selected |          Selected |
| 2022           |               Selected |      Not selected |
| 2023           |           Not selected |      Not selected |
| 2024           |           Not selected |          Selected |
| 2025           |           Not selected |      Not selected |

Partitioning did not select CELH in January 2020, when its trailing Sharpe rank was only 36% relative to the eligible stocks. It selected CELH in 2021 and 2022 after the stock's trailing Sharpe rank improved.

Density selected CELH in January 2020 before seeing its later 2020 holding-period return. At that rebalance, CELH had a strong value rank even though its trailing Sharpe rank was not especially high. The combination of its features, clustering position, and diversification value allowed it to be selected.

The strategies also rejected CELH in several later years, including some years when its trailing Sharpe rank was still high. This happened because selection depended on more than past returns. Valuation, clusters, correlations, and the other available stocks also mattered.

CELH is still an outlier and the results are sensitive to it. However, that does not mean CELH was selected using future knowledge. The ex-CELH test is a robustness test that shows how much the final performance depended on one exceptional winner.

## The comparison strategies

I compared the new selectors with:

- SPY;
- Equal Weight across the full stock universe;
- Equal Weight across the exact fundamental-eligible universe;
- Max Sharpe;
- Maximum Diversification;
- Minimum Variance;
- Risk Parity;
- Inverse Volatility;
- Hierarchical Risk Parity.

The eligible-universe equal-weight portfolio is especially important. It shows whether the selector added value compared with simply buying every stock for which the same point-in-time fundamental data was available.

## Main result in simple terms

Density Selection had the strongest full-sample result:

- 21.82% annualized return;
- 0.949 Sharpe ratio; and
- -25.40% maximum drawdown.

Partitioning Selection had:

- 16.60% annualized return;
- 0.759 Sharpe ratio; and
- -25.44% maximum drawdown.

The eligible-universe equal-weight baseline returned 18.24% annually, while SPY returned 14.85% annually.

Density produced the best overall performance, but it beat the eligible baseline in only three of six individual years. Partitioning beat the eligible baseline in five of six years, but its total compounded performance was lower.

When CELH was removed, Density returned 12.38% annually and Partitioning returned 12.60%, compared with 12.40% for the ex-CELH eligible baseline and 14.85% for SPY. This shows that the strongest headline result depended heavily on CELH even though CELH was not held every year.

## What I think the experiment shows

The experiment gives positive historical evidence that this group of coping-related companies produced strong returns during the test period. It also shows that a clustering strategy can identify a smaller group that performs well without putting more than 8.33% into one stock.

At the same time, the experiment does not prove that the result will continue in the future. There are only six annual holding periods, five intended stocks are missing from the point-in-time fundamental dataset, and CELH had a large effect on the result.

My conclusion is that the idea is worth researching further, but it is not ready to be treated as a guaranteed or fully validated investment strategy.
