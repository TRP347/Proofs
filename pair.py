import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

class PairsTradingSimulator:
    def __init__(self, symbol1, symbol2, start_date, end_date):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.start_date = start_date
        self.end_date = end_date
        self.prices1 = self.load_prices(symbol1)
        self.prices2 = self.load_prices(symbol2)
        self.prices1_train = self.prices1[self.start_date:self.end_date]
        self.prices2_train = self.prices2[self.start_date:self.end_date]
        self.returns1 = self.calculate_returns(self.prices1_train)
        self.returns2 = self.calculate_returns(self.prices2_train)
        self.spread = None
        self.entry_threshold = 1.0  # Example threshold for spread deviation
        self.exit_threshold = 0.5   # Example threshold for spread reversion
        self.positions1 = [0]
        self.positions2 = [0]
        self.equity_curve = [1.0]  # Start with initial capital of 1

    def load_prices(self, symbol):
        # Fetch historical price data using yfinance
        data = yf.download(symbol, start=self.start_date, end=self.end_date)
        prices = data['Close']
        return prices

    def calculate_returns(self, prices):
        # Calculate daily returns from price data
        returns = prices.pct_change().fillna(0)
        return returns

    def calculate_spread(self):
        # Calculate spread between two securities
        self.spread = self.prices1_train - self.prices2_train

    def simulate_pairs_trading(self):
        # Simulate pairs trading strategy over historical data
        self.calculate_spread()
        for idx in range(1, len(self.spread)):
            # Entry condition: spread widens beyond entry threshold
            if self.spread[idx] > self.entry_threshold and self.positions1[-1] == 0:
                self.positions1.append(1)
                self.positions2.append(-1)
            # Exit condition: spread reverts below exit threshold
            elif self.spread[idx] < self.exit_threshold and self.positions1[-1] == 1:
                self.positions1.append(0)
                self.positions2.append(0)
            else:
                self.positions1.append(self.positions1[-1])
                self.positions2.append(self.positions2[-1])

            # Calculate equity curve based on position changes
            self.equity_curve.append(self.calculate_equity_curve(idx))

    def calculate_equity_curve(self, idx):
        # Calculate equity curve based on positions and price changes
        if idx == 0:
            return 1.0
        else:
            return self.equity_curve[-1] * (1 + self.positions1[-1] * self.returns1.iloc[idx] +
                                            self.positions2[-1] * self.returns2.iloc[idx])

    def calculate_performance_metrics(self):
        # Calculate performance metrics
        cumulative_return = self.equity_curve[-1] - 1
        daily_returns = pd.Series(self.equity_curve).pct_change().fillna(0)
        annualized_return = (1 + cumulative_return) ** (252 / len(self.prices1_train)) - 1  # Assuming 252 trading days
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / annualized_volatility  # Assuming risk-free rate of 2%
        max_drawdown = np.min(self.equity_curve / np.maximum.accumulate(self.equity_curve)) - 1

        return {
            "Cumulative Return": cumulative_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown
        }

    def plot_results(self):
        # Plot equity curve and spread
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.prices1_train.index, self.prices1_train.values, label=f"{self.symbol1} Prices")
        plt.plot(self.prices2_train.index, self.prices2_train.values, label=f"{self.symbol2} Prices")
        plt.legend(loc="upper left")
        plt.title("Historical Prices")

        plt.subplot(2, 1, 2)
        plt.plot(self.prices1_train.index, self.spread, label="Spread")
        plt.axhline(self.entry_threshold, color="r", linestyle="--", label="Entry Threshold")
        plt.axhline(self.exit_threshold, color="g", linestyle="--", label="Exit Threshold")
        plt.legend(loc="upper left")
        plt.title("Spread and Trading Signals")
        plt.xlabel("Date")
        plt.ylabel("Spread")
        plt.tight_layout()
        plt.show()

    def backtest(self):
        # Conduct backtest of the pairs trading strategy
        self.simulate_pairs_trading()
        performance_metrics = self.calculate_performance_metrics()

        print("\nPerformance Metrics:")
        for metric, value in performance_metrics.items():
            print(f"{metric}: {value}")

        self.plot_results()

if __name__ == "__main__":
    symbol1 = "AAPL"  # Symbol of first security
    symbol2 = "MSFT"  # Symbol of second security
    start_date = "2019-01-01"
    end_date = "2020-01-01"

    simulator = PairsTradingSimulator(symbol1, symbol2, start_date, end_date)
    simulator.backtest()
