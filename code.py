import numpy as np
import pandas as pd
import vectorbt as vbt


INITIAL_CASH = 100_000
FEES = 0.001
SLIPPAGE = 0.0005

EMA_PERIODS = np.array([5, 8, 13, 21, 34, 55, 89, 144])

ENTRY_Z = 1.0
EXIT_Z = 0.0
WINDOW = 100

np.random.seed(42)

n = 3000

dates = pd.date_range(start="2024-01-01", periods=n, freq="h")

returns = np.random.normal(loc=0.00005, scale=0.005, size=n)

prices = 100 * np.exp(np.cumsum(returns))

close = pd.Series(prices, index=dates, name="Close")


def ema_field(close, periods):

    field = pd.DataFrame(index=close.index)

    for period in periods:

        field[period] = close.ewm(span=int(period), adjust=False).mean()

    return field


field = ema_field(close, EMA_PERIODS)


def scale_derivative(field, periods):

    log_periods = np.log(periods)

    derivatives = np.gradient(field.to_numpy(), log_periods, axis=1)

    return pd.DataFrame(-derivatives, index=field.index, columns=field.columns)


derivative = scale_derivative(field, EMA_PERIODS)


# Average derivative across EMA timescales

scale_signal = derivative.mean(axis=1)

# Normalize by rolling statistics

rolling_mean = scale_signal.rolling(WINDOW, min_periods=WINDOW).mean()

rolling_std = scale_signal.rolling(WINDOW, min_periods=WINDOW).std()

z_score = ((scale_signal - rolling_mean) / rolling_std.replace(0, np.nan))

# Shift signals to prevent using current-bar
# closing information for same-bar execution.

signal = z_score.shift(1)

entries = ((signal > ENTRY_Z) & (signal.shift(1) <= ENTRY_Z))

exits = ((signal < EXIT_Z) & (signal.shift(1) >= EXIT_Z))

portfolio = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    init_cash=INITIAL_CASH,
    fees=FEES,
    slippage=SLIPPAGE,
    freq="1h",
    direction="longonly"
)

print("\nEMA FIELD BACKTEST\n")

print(portfolio.stats())

print("\nTotal Return:")
print(portfolio.total_return())

print("\nSharpe Ratio:")
print(portfolio.sharpe_ratio())

print("\nMax Drawdown:")
print(portfolio.max_drawdown())

print("\nTrade Count:")
print(portfolio.trades.count())