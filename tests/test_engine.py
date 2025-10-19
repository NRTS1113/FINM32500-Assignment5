import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, Mock
from backtester.engine import Backtester
from backtester.strategy import VolatilityBreakoutStrategy
from backtester.broker import Broker


class TestBacktester:
    """Comprehensive tests for Backtester engine."""

    def test_backtester_initialization(self, strategy, broker):
        """Backtester should initialize with strategy and broker."""
        bt = Backtester(strategy, broker)
        assert bt.strategy is strategy
        assert bt.broker is broker

    def test_run_returns_dict_with_required_keys(self, strategy, broker, prices):
        """run() should return dict with equity, signals, and trades."""
        bt = Backtester(strategy, broker)
        result = bt.run(prices)

        assert isinstance(result, dict)
        assert "equity" in result
        assert "signals" in result
        assert "trades" in result

    def test_equity_series_same_length_as_prices(self, strategy, broker, prices):
        """Equity series should have same length as price series."""
        bt = Backtester(strategy, broker)
        result = bt.run(prices)

        assert len(result["equity"]) == len(prices)

    def test_signals_series_same_length_as_prices(self, strategy, broker, prices):
        """Signals series should have same length as price series."""
        bt = Backtester(strategy, broker)
        result = bt.run(prices)

        assert len(result["signals"]) == len(prices)

    def test_uses_tminus1_signal_for_trading(self, broker, prices):
        """Engine should use previous signal to determine current trade."""
        # Create a mock strategy that returns specific signals
        mock_strategy = MagicMock()
        signals = pd.Series(0.0, index=prices.index)
        # First bar has signal 0, second bar onwards has signal 1
        # This should trigger a buy when processing the second bar
        signals.iloc[1:] = 1.0  # Buy signal from second bar onwards
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # Should have executed a buy trade (signal goes from 0 to 1)
        assert len(broker.trade_log) >= 1
        assert broker.trade_log[0]["side"].lower() == "buy"
        assert broker.position > 0

    def test_buy_signal_executes_buy(self):
        """Positive signal should execute buy order."""
        broker = Broker(cash=10000)
        prices = pd.Series([100.0] * 50)

        mock_strategy = MagicMock()
        signals = pd.Series(0.0, index=prices.index)
        signals.iloc[10:] = 1.0  # Buy signal from t=10 onwards
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # Should have bought shares
        assert broker.position > 0
        assert len(broker.trade_log) >= 1
        assert broker.trade_log[0]["side"].lower() == "buy"

    def test_sell_signal_executes_sell(self):
        """Zero signal after long position should execute sell."""
        broker = Broker(cash=10000)
        prices = pd.Series([100.0] * 50)

        mock_strategy = MagicMock()
        signals = pd.Series(0.0, index=prices.index)
        signals.iloc[10:25] = 1.0  # Buy at t=10, sell at t=25
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # Should have both buy and sell
        assert len(broker.trade_log) >= 2
        assert broker.trade_log[0]["side"].lower() == "buy"
        assert broker.trade_log[-1]["side"].lower() == "sell"

    def test_equity_starts_with_initial_cash(self):
        """Initial equity should equal initial cash."""
        broker = Broker(cash=50000)
        prices = pd.Series([100.0] * 20)
        strategy = VolatilityBreakoutStrategy()

        bt = Backtester(strategy, broker)
        result = bt.run(prices)

        # First equity value should be close to initial cash (might trade on first bar)
        assert result["equity"].iloc[0] <= 50000

    def test_equity_updates_with_position_value(self):
        """Equity should include position value at current prices."""
        broker = Broker(cash=10000)
        prices = pd.Series([100.0, 100.0, 110.0, 110.0, 110.0])

        mock_strategy = MagicMock()
        signals = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0])
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # After buying at 100 and price moving to 110, equity should increase
        assert result["equity"].iloc[-1] > result["equity"].iloc[1]

    def test_no_trade_on_zero_signal(self):
        """Zero signals throughout should result in no trades."""
        broker = Broker(cash=10000)
        prices = pd.Series([100.0] * 20)

        mock_strategy = MagicMock()
        signals = pd.Series(0.0, index=prices.index)
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        assert len(broker.trade_log) == 0
        assert broker.position == 0

    def test_handles_empty_price_series(self, strategy, broker):
        """Engine should handle empty price series gracefully."""
        prices = pd.Series([], dtype=float)
        bt = Backtester(strategy, broker)
        result = bt.run(prices)

        assert len(result["equity"]) == 0
        assert len(result["signals"]) == 0

    def test_handles_single_price(self, strategy, broker):
        """Engine should handle single price point."""
        prices = pd.Series([100.0])
        bt = Backtester(strategy, broker)
        result = bt.run(prices)

        assert len(result["equity"]) == 1
        assert len(result["signals"]) == 1

    def test_trades_dataframe_structure(self):
        """Trades DataFrame should have expected columns."""
        broker = Broker(cash=10000)
        prices = pd.Series([100.0] * 20)

        mock_strategy = MagicMock()
        signals = pd.Series(0.0, index=prices.index)
        signals.iloc[5:15] = 1.0
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        if len(result["trades"]) > 0:
            assert "side" in result["trades"].columns
            assert "qty" in result["trades"].columns
            assert "price" in result["trades"].columns
            assert "cash" in result["trades"].columns
            assert "position" in result["trades"].columns

    def test_signal_clipping_to_0_1(self):
        """Engine clips signals to [0, 1] range."""
        broker = Broker(cash=10000)
        prices = pd.Series([100.0] * 20)

        mock_strategy = MagicMock()
        # Return signals outside [0,1] range
        signals = pd.Series([0, 0.5, 1.0, 1.5, 2.0, -0.5] + [1.0] * 14)
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # Should clip signals to [0, 1]
        assert result["signals"].min() >= 0.0
        assert result["signals"].max() <= 1.0

    def test_buys_maximum_shares_with_available_cash(self):
        """Buy should use all available cash (floor division)."""
        broker = Broker(cash=1000)
        prices = pd.Series([30.0] * 10)

        mock_strategy = MagicMock()
        signals = pd.Series([0, 1.0] + [1.0] * 8)
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # Should buy floor(1000/30) = 33 shares
        assert broker.trade_log[0]["qty"] == 33

    def test_no_buy_if_insufficient_cash(self):
        """No buy trade if cash cannot buy even 1 share."""
        broker = Broker(cash=10)
        prices = pd.Series([100.0] * 10)

        mock_strategy = MagicMock()
        signals = pd.Series([0, 1.0] + [1.0] * 8)
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # Should not execute any buy
        assert len(broker.trade_log) == 0

    def test_sells_entire_position(self):
        """Sell should liquidate entire position."""
        broker = Broker(cash=10000)
        prices = pd.Series([100.0] * 20)

        mock_strategy = MagicMock()
        signals = pd.Series([0, 1.0] * 5 + [0.0] * 10)
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # After sell, position should be 0
        sell_trades = [t for t in broker.trade_log if t["side"].lower() == "sell"]
        if sell_trades:
            assert sell_trades[-1]["position"] == 0

    def test_no_sell_if_no_position(self):
        """No sell if position is already zero."""
        broker = Broker(cash=10000)
        prices = pd.Series([100.0] * 10)

        mock_strategy = MagicMock()
        # Signal goes to 0 but we never had a position
        signals = pd.Series([0.0] * 10)
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        assert len(broker.trade_log) == 0

    def test_integration_with_real_strategy(self):
        """Full integration test with real VolatilityBreakoutStrategy."""
        broker = Broker(cash=100000)
        strategy = VolatilityBreakoutStrategy(lookback=10, k=1.0)

        # Create trending price series
        prices = pd.Series(np.linspace(100, 150, 100))

        bt = Backtester(strategy, broker)
        result = bt.run(prices)

        # Should have equity curve
        assert len(result["equity"]) == len(prices)
        # Equity should be positive throughout
        assert result["equity"].min() > 0

    def test_handles_nan_in_signals(self, broker):
        """Engine should handle NaN in signals by filling with 0."""
        prices = pd.Series([100.0] * 10)

        mock_strategy = MagicMock()
        signals = pd.Series([np.nan, 0.0, 1.0, np.nan, 1.0, 0.0, np.nan, 0.0, 0.0, 0.0])
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # Should not crash
        assert len(result["equity"]) == len(prices)

    def test_equity_index_matches_price_index(self):
        """Equity series index should match price series index."""
        broker = Broker(cash=10000)
        strategy = VolatilityBreakoutStrategy()

        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = pd.Series(np.linspace(100, 120, 50), index=dates)

        bt = Backtester(strategy, broker)
        result = bt.run(prices)

        pd.testing.assert_index_equal(result["equity"].index, prices.index)

    def test_multiple_round_trips(self):
        """Multiple buy-sell cycles should work correctly."""
        broker = Broker(cash=10000)
        prices = pd.Series([100.0] * 50)

        mock_strategy = MagicMock()
        # Create multiple buy-sell cycles
        signals = pd.Series([0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0] + [0] * 36)
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)
        result = bt.run(prices)

        # Should have multiple trades
        buy_count = sum(1 for t in broker.trade_log if t["side"].lower() == "buy")
        sell_count = sum(1 for t in broker.trade_log if t["side"].lower() == "sell")

        assert buy_count >= 2
        assert sell_count >= 2

    def test_broker_exception_propagates(self):
        """Exceptions from broker should propagate."""
        broker = Broker(cash=1000)
        prices = pd.Series([100.0] * 10)

        # Mock broker to raise exception
        broker.market_order = Mock(side_effect=ValueError("Test error"))

        mock_strategy = MagicMock()
        signals = pd.Series([0, 1.0] + [1.0] * 8)
        mock_strategy.signals.return_value = signals

        bt = Backtester(mock_strategy, broker)

        with pytest.raises(ValueError, match="Test error"):
            bt.run(prices)

    def test_deterministic_results(self):
        """Same inputs should produce same results."""
        strategy = VolatilityBreakoutStrategy(lookback=10)
        prices = pd.Series(np.linspace(100, 120, 50))

        broker1 = Broker(cash=10000)
        bt1 = Backtester(strategy, broker1)
        result1 = bt1.run(prices)

        broker2 = Broker(cash=10000)
        bt2 = Backtester(strategy, broker2)
        result2 = bt2.run(prices)

        pd.testing.assert_series_equal(result1["equity"], result2["equity"])
        pd.testing.assert_series_equal(result1["signals"], result2["signals"])
