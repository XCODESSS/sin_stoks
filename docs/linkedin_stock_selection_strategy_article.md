# How I built two clustering-based stock-selection strategies

I recently tested two ways to select 12 stocks from a 30-stock portfolio. Both strategies use the same information and the same scoring system. The main difference is how they group similar stocks before choosing which ones to hold.

One strategy uses PAM clustering and the other uses HDBSCAN. I wanted to see whether either method could select a smaller and more diversified group of stocks than simply holding the full portfolio.

The results were mixed. HDBSCAN had the strongest overall return, but much of that result depended on CELH. PAM was more consistent from year to year, but it did not beat the fairest equal-weight comparison over the full period. For that reason, I see both strategies as working experiments rather than finished investment strategies.

## Why I tested this idea

The project started with a question: can companies connected to the ways people cope with everyday life also produce strong investment returns?

The portfolio includes companies from categories such as food, drinks, entertainment, technology, tobacco, and restaurants. I chose the categories myself, but the individual stocks were selected randomly before I knew how their historical results would look. This point became important later because CELH turned out to have a large effect on the final performance.

I could have tested the idea by holding all 30 stocks. I also wanted to know whether a systematic method could choose 12 of them without filling the portfolio with companies that behaved in much the same way. That is why I used clustering.

## How I avoided using future information

The backtest rebalances once a year from 2020 through 2025. At each January rebalance, the strategy uses only information that was available before that date.

For example, the portfolio chosen in January 2020 can use financial statements filed before 2020, prices from before 2020, and the previous 104 weeks of returns. It cannot use anything that happened later in the year.

I also did not copy today's market capitalization or valuation ratios into earlier years. Historical market capitalization was reconstructed from the share price and share count available at the time. A fundamental record was allowed into the strategy only when its availability date was earlier than the rebalance date.

I found acceptable point-in-time fundamental data for 25 of the 30 stocks at every rebalance. The other five were excluded instead of being filled with made-up estimates. This gave the test 83.33% coverage of the original portfolio.

## How each stock was ranked

At every rebalance, I created three ranks for each eligible stock.

The value rank was mainly based on trailing P/E. Among profitable companies, a lower positive P/E received a better rank. A company with negative earnings received a value rank of zero rather than an artificial P/E ratio.

The size rank was based on the logarithm of historical market capitalization. Larger companies received higher ranks.

The trailing Sharpe rank measured return relative to volatility over the previous 104 weeks. I used a fixed annual risk-free rate of 4%.

I then calculated a basic selection score:

```text
50% value rank + 50% trailing Sharpe rank
```

The size rank was used to compare companies during clustering. It did not directly increase a stock's selection score.

A high score was not supposed to predict the next year's return with certainty. It only described how a stock compared with the other eligible stocks using information already available at the rebalance.

## How the stocks were grouped

The clustering methods needed a way to measure the distance between two stocks. I gave equal weight to two types of distance.

The first was based on return correlation. Stocks whose prices had moved together were treated as more similar.

The second was based on the value, size, and trailing Sharpe ranks. Stocks with similar financial and market characteristics were also treated as more similar.

The final distance was 50% return-correlation distance and 50% feature distance. Both strategies used the same distance matrix.

### Partitioning Selection using PAM

PAM stands for Partitioning Around Medoids. It is similar to k-means, except that the center of each group is an actual stock rather than an average point that does not exist in the portfolio.

I fixed the number of groups at six. PAM divided the eligible stocks into those groups, ranked the members of each group by their basic score, and took up to the two highest-scoring stocks from each one.

This method tries to stop one type of company from taking over the portfolio. A stock needs a useful value-and-Sharpe score, but it also competes mainly with other stocks in its group.

### Density Selection using HDBSCAN

HDBSCAN stands for Hierarchical Density-Based Spatial Clustering of Applications with Noise. The name is complicated, but the idea is fairly simple. It looks for groups that form naturally instead of forcing every stock into one of six fixed groups.

HDBSCAN can also label an unusual stock as noise. The strategy first selected the highest-scoring stock from each normal cluster. It then filled the remaining places until the portfolio contained 12 stocks.

A noise stock was not automatically rejected. It could still be selected during the filling stage if it had a good score and helped diversify the portfolio.

In the actual runs, HDBSCAN usually found one main cluster and treated several stocks as noise. This meant that the score and diversification rule did much of the work when the strategy filled the 12 positions.

## How the final 12 stocks were chosen

When either strategy still had an empty position, it adjusted each remaining stock's basic score for its average correlation with the stocks already selected:

```text
adjusted score = basic score - 0.25 x average correlation with selected stocks
```

This gives an advantage to a stock that adds something different to the portfolio. The effect is limited because valuation and trailing Sharpe still determine the basic score.

Each of the 12 selected stocks received an equal weight of about 8.33%. After the January selection, the portfolio was held until the next rebalance. I applied a 10-basis-point transaction cost based on turnover, then rebuilt the ranks and clusters using the information available at the next rebalance.

## The results

Density Selection produced the strongest full-period result:

- 21.82% annualized return
- 0.949 Sharpe ratio
- -25.40% maximum drawdown

Partitioning Selection produced:

- 16.60% annualized return
- 0.759 Sharpe ratio
- -25.44% maximum drawdown

The equal-weight portfolio containing the same 25 eligible stocks returned 18.24% annually. The full 30-stock equal-weight portfolio returned 15.70%, while SPY returned 14.85%.

Density had the best full-period performance, but it beat the eligible equal-weight portfolio in only three of the six individual years. Partitioning beat that baseline in five of six years, even though its compounded return was lower.

Recurring one-way turnover was about 44% for both strategies. This was below the 60% limit I had set before running the test.

## Why CELH changes the interpretation

CELH was in the original stock list before I knew what its later performance would be. The strategies also made different decisions about it over time.

PAM did not select CELH in 2020. It selected the stock in 2021 and 2022, then left it out for the next three rebalances. HDBSCAN selected CELH in 2020 and 2021, left it out in 2022 and 2023, selected it again in 2024, and left it out in 2025.

This shows that neither strategy simply held the eventual winner throughout the test. Including CELH was not look-ahead bias. Its return was still unusually large, so I repeated the full experiment without it to see how dependent the results were on that one stock.

Without CELH, Density returned 12.38% annually and Partitioning returned 12.60%. The comparable eligible equal-weight portfolio returned 12.40%. SPY returned 14.85% over the same period.

The HDBSCAN result therefore changed substantially when CELH was removed. PAM held up slightly better against the eligible baseline, but it still fell behind SPY.

## My conclusion

Before running the experiment, I set rules that each strategy had to pass before I would call it research-promising. Those rules covered overall return, performance across individual years, drawdown, turnover, data coverage, and the test without CELH.

Neither strategy passed every rule.

Density failed because it beat the eligible baseline in only three of six years and fell slightly behind that baseline without CELH. Partitioning was more consistent, but its full-period return was below the eligible baseline and its information ratio against that baseline was negative.

The correct classification for both strategies is feasible but not promising. The selection methods worked as designed, and the results are worth studying, but the evidence is not strong enough to call either method validated.

There are only six annual holding periods in the test. Five stocks are missing from the point-in-time fundamental dataset. The portfolio is a retrospective thematic sample rather than a complete historical index, and the trading-cost model does not include every real-world cost. There is also no live or untouched test period yet.

I think the idea deserves more research. A better next test would use a broader point-in-time universe, a longer history, different rebalance dates, more outlier checks, and a genuinely untouched holdout period.

For now, the main lesson is that an attractive backtest result can change quickly when one influential stock is removed. That is exactly why the less exciting robustness results need to be reported alongside the headline numbers.

*This article describes a historical research experiment and is not investment advice. Backtested results do not guarantee future performance.*
