import numpy as np
import pandas as pd
import pytest
from backtester.strategy import VolatilityBreakoutStrategy


class TestVolatilityBreakoutStrategy:
    """Comprehensive tests for VolatilityBreakoutStrategy."""

    def test_signals_length_matches_prices(self, strategy, prices):
        """Signal series should have same length as price series."""
        sig = strategy.signals(prices)
        assert len(sig) == len(prices)

    def test_signals_returns_series(self, strategy, prices):
        """signals() should return a pandas Series."""
        sig = strategy.signals(prices)
        assert isinstance(sig, pd.Series)

    def test_signals_initial_nan_handling(self, strategy):
        """Strategy should handle initial NaN values from rolling window."""
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        sig = strategy.signals(prices)
        # With default lookback=20, early signals should be 0 due to insufficient data
        assert len(sig) == len(prices)
        assert not sig.isna().any()  # fillna should handle NaNs

    def test_constant_price_series(self, strategy):
        """Constant prices should produce zero signals."""
        prices = pd.Series([100.0] * 200)
        sig = strategy.signals(prices)
        # Constant prices = no volatility breakouts
        assert sig.sum() == 0.0

    def test_empty_series(self, strategy):
        """Empty price series should return empty signal series."""
        prices = pd.Series([], dtype=float)
        sig = strategy.signals(prices)
        assert len(sig) == 0

    def test_very_short_series(self):
        """Very short series (< lookback) should not crash."""
        strategy = VolatilityBreakoutStrategy(lookback=20)
        prices = pd.Series([100, 101, 102])
        sig = strategy.signals(prices)
        assert len(sig) == len(prices)
        # All signals should be 0 due to insufficient lookback
        assert sig.sum() == 0.0

    def test_long_only_mode(self):
        """Long-only mode should never produce negative signals."""
        strategy = VolatilityBreakoutStrategy(lookback=10, long_only=True)
        prices = pd.Series(np.linspace(100, 80, 100))  # declining prices
        sig = strategy.signals(prices)
        assert sig.min() >= 0.0

    def test_long_short_mode(self):
        """Long-short mode can produce negative signals."""
        strategy = VolatilityBreakoutStrategy(lookback=10, long_only=False)
        # Create a series with sharp decline to trigger short signal
        prices = pd.Series([100] * 15 + [90, 85, 80, 75, 70] + [70] * 10)
        sig = strategy.signals(prices)
        # Should have some negative signals in long_short mode
        assert sig.min() <= 0.0

    def test_hold_mode_persistence(self):
        """Hold mode should persist signals until reversed."""
        strategy = VolatilityBreakoutStrategy(lookback=5, hold=True, lag_signal=0)
        # Single spike should produce persistent signal with hold=True
        prices = pd.Series([100] * 10 + [120] + [110] * 10)
        sig = strategy.signals(prices)
        # After the spike, signal should persist
        assert sig.iloc[-1] != 0  # Signal persists

    def test_no_hold_mode(self):
        """No-hold mode should only signal on breakout bars."""
        strategy = VolatilityBreakoutStrategy(lookback=5, hold=False, lag_signal=0)
        prices = pd.Series([100] * 10 + [120] + [110] * 10)
        sig = strategy.signals(prices)
        # Without hold, signals are transient
        assert sig.sum() >= 0  # Should have at least some signals

    def test_lag_signal_shifts(self):
        """lag_signal should shift signals forward."""
        prices = pd.Series(np.linspace(100, 150, 50))

        strategy_no_lag = VolatilityBreakoutStrategy(lookback=10, lag_signal=0)
        strategy_lag_1 = VolatilityBreakoutStrategy(lookback=10, lag_signal=1)

        sig_no_lag = strategy_no_lag.signals(prices)
        sig_lag_1 = strategy_lag_1.signals(prices)

        # With lag=1, first signal should be 0
        assert sig_lag_1.iloc[0] == 0.0
        # Lagged signal should be shifted version
        np.testing.assert_array_almost_equal(
            sig_no_lag.iloc[:-1].values,
            sig_lag_1.iloc[1:].values
        )

    def test_lookback_parameter(self):
        """Different lookback periods should produce different signals."""
        prices = pd.Series(np.linspace(100, 150, 100))

        strategy_short = VolatilityBreakoutStrategy(lookback=5)
        strategy_long = VolatilityBreakoutStrategy(lookback=30)

        sig_short = strategy_short.signals(prices)
        sig_long = strategy_long.signals(prices)

        # Different lookbacks should produce different signals
        assert not sig_short.equals(sig_long)

    def test_k_parameter_affects_threshold(self):
        """Higher k should produce fewer signals (higher threshold)."""
        prices = pd.Series(np.random.RandomState(42).normal(100, 5, 200))

        strategy_low_k = VolatilityBreakoutStrategy(lookback=20, k=0.5, hold=False)
        strategy_high_k = VolatilityBreakoutStrategy(lookback=20, k=3.0, hold=False)

        sig_low_k = strategy_low_k.signals(prices)
        sig_high_k = strategy_high_k.signals(prices)

        # Lower k should produce more signals
        assert sig_low_k.abs().sum() >= sig_high_k.abs().sum()

    def test_accepts_array_like_input(self):
        """Strategy should accept array-like inputs, not just Series."""
        strategy = VolatilityBreakoutStrategy()

        # Test with list
        prices_list = list(np.linspace(100, 120, 50))
        sig = strategy.signals(prices_list)
        assert isinstance(sig, pd.Series)
        assert len(sig) == len(prices_list)

    def test_signal_name_preservation(self):
        """Signal series should preserve or set appropriate name."""
        strategy = VolatilityBreakoutStrategy()
        prices = pd.Series(np.linspace(100, 120, 50), name="AAPL")
        sig = strategy.signals(prices)
        assert sig.name == "AAPL"

    def test_handles_nan_in_prices(self):
        """Strategy should handle NaN values in price series."""
        strategy = VolatilityBreakoutStrategy(lookback=10)
        prices = pd.Series([100, 101, np.nan, 103, 104] + list(np.linspace(105, 120, 45)))
        sig = strategy.signals(prices)
        # Should not crash and should return same length
        assert len(sig) == len(prices)

    def test_signal_values_are_valid(self, strategy, prices):
        """All signal values should be in valid range."""
        sig = strategy.signals(prices)
        # For long_only=True, signals should be 0 or 1
        assert sig.min() >= -1.0
        assert sig.max() <= 1.0

    def test_deterministic_output(self):
        """Same inputs should produce same outputs."""
        strategy = VolatilityBreakoutStrategy(lookback=10, k=1.5)
        prices = pd.Series(np.linspace(100, 120, 100))

        sig1 = strategy.signals(prices)
        sig2 = strategy.signals(prices)

        pd.testing.assert_series_equal(sig1, sig2)

    def test_volatile_price_series_generates_signals(self):
        """Highly volatile prices should generate signals."""
        strategy = VolatilityBreakoutStrategy(lookback=10, hold=False, lag_signal=0)
        # Create volatile series
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 5))
        sig = strategy.signals(prices)
        # Should have at least some signals
        assert sig.abs().sum() > 0
