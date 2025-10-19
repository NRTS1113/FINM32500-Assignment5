import pandas as pd
import numpy as np

class PriceLoader:
    def __init__(self, start="2020-01-01", end="2020-12-31", seed=42):
        np.random.seed(seed)
        self.dates = pd.date_range(start, end, freq="B")

    def get_price(self, symbol: str) -> pd.Series:
        """
        Returns a pandas.Series of synthetic prices for one symbol.
        Prices follow a simple geometric random walk model.
        """
        n = len(self.dates)
        returns = np.random.normal(0, 0.05, n)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=self.dates, name=symbol)
