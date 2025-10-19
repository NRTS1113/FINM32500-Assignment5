import pandas as pd
import numpy as np


class VolatilityBreakoutStrategy:

    def __init__(self, lookback: int = 20, k = 1.5, hold: bool = True, lag_signal: int = 1, long_only: bool = True):
        self.lookback = int(lookback)
        self.k = float(k)
        self.hold = bool(hold)
        self.lag_signal = int(lag_signal)
        self.long_only = bool(long_only)

    def signals(self, prices: pd.Series) -> pd.Series:
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)

        prices = prices.astype(float).copy()
        prev_close = prices.shift(1)

        rets = prices.pct_change()
        vol = rets.rolling(self.lookback, min_periods=self.lookback).std().shift(1)

        up_band = prev_close * (1 + self.k * vol)
        dn_band = prev_close * (1 - self.k * vol)

        raw = pd.Series(0, index=prices.index, dtype=float)
        raw.loc[prices > up_band] = 1.0
        if not self.long_only:
            raw.loc[prices < dn_band] = -1.0

        if self.hold:
            pos = raw.replace(0, np.nan).ffill().fillna(0.0)
            sig = pos
        else:
            sig = raw 

        if self.lag_signal > 0:
            sig = sig.shift(self.lag_signal).fillna(0.0)

        return sig.rename(getattr(prices, "name", None) or "signal")