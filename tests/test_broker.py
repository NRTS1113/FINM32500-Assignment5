import pytest
from backtester.broker import Broker


class TestBroker:
    """Comprehensive tests for Broker class."""

    def test_initial_state(self):
        """Broker should initialize with correct cash and zero position."""
        broker = Broker(cash=10_000)
        assert broker.cash == 10_000
        assert broker.position == 0
        assert broker.trade_log == []

    def test_default_cash(self):
        """Broker should have default cash of 1,000,000."""
        broker = Broker()
        assert broker.cash == 1_000_000

    def test_buy_updates_cash_and_position(self, broker):
        """Buy order should decrease cash and increase position."""
        initial_cash = broker.cash
        broker.market_order("buy", 10, 50.0)
        assert broker.cash == initial_cash - 500.0
        assert broker.position == 10

    def test_sell_updates_cash_and_position(self, broker):
        """Sell order should increase cash and decrease position."""
        # First buy some shares
        broker.market_order("buy", 10, 50.0)
        cash_after_buy = broker.cash

        # Then sell them
        broker.market_order("sell", 5, 60.0)
        assert broker.cash == cash_after_buy + 300.0
        assert broker.position == 5

    def test_case_insensitive_side(self):
        """Order side should be case-insensitive."""
        broker = Broker(cash=1000)

        broker.market_order("BUY", 1, 10.0)
        assert broker.position == 1

        broker.market_order("SELL", 1, 10.0)
        assert broker.position == 0

        broker.market_order("Buy", 1, 10.0)
        assert broker.position == 1

        broker.market_order("Sell", 1, 10.0)
        assert broker.position == 0

    def test_rejects_zero_quantity(self, broker):
        """Broker should reject orders with zero quantity."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            broker.market_order("buy", 0, 50.0)

    def test_rejects_negative_quantity(self, broker):
        """Broker should reject orders with negative quantity."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            broker.market_order("buy", -5, 50.0)

    def test_rejects_zero_price(self, broker):
        """Broker should reject orders with zero price."""
        with pytest.raises(ValueError, match="Price must be positive"):
            broker.market_order("buy", 10, 0.0)

    def test_rejects_negative_price(self, broker):
        """Broker should reject orders with negative price."""
        with pytest.raises(ValueError, match="Price must be positive"):
            broker.market_order("buy", 10, -50.0)

    def test_allows_negative_cash(self):
        """Broker allows negative cash (no margin check in this simple implementation)."""
        broker = Broker(cash=100)
        # This will make cash negative
        broker.market_order("buy", 10, 50.0)
        assert broker.cash < 0

    def test_allows_negative_position(self, broker):
        """Broker allows short selling (negative position)."""
        # Sell without owning (short sale)
        broker.market_order("sell", 10, 50.0)
        assert broker.position == -10

    def test_trade_log_records_trades(self, broker):
        """Trade log should record all trades."""
        broker.market_order("buy", 5, 100.0)
        broker.market_order("sell", 2, 110.0)

        assert len(broker.trade_log) == 2
        assert broker.trade_log[0]["side"] == "buy"
        assert broker.trade_log[0]["qty"] == 5
        assert broker.trade_log[0]["price"] == 100.0
        assert broker.trade_log[1]["side"] == "sell"
        assert broker.trade_log[1]["qty"] == 2

    def test_trade_log_tracks_state(self, broker):
        """Trade log should track cash and position after each trade."""
        broker.market_order("buy", 10, 50.0)

        assert broker.trade_log[0]["cash"] == 1000 - 500
        assert broker.trade_log[0]["position"] == 10

    def test_total_value_calculation(self, broker):
        """total_value should correctly calculate portfolio value."""
        # Initial: all cash
        assert broker.total_value(100.0) == 1000.0

        # After buying
        broker.market_order("buy", 5, 50.0)
        # cash = 750, position = 5, current_price = 60
        assert broker.total_value(60.0) == 750 + 5 * 60

    def test_total_value_with_negative_position(self):
        """total_value should handle short positions correctly."""
        broker = Broker(cash=1000)
        broker.market_order("sell", 5, 50.0)  # Short 5 shares
        # cash = 1250, position = -5, current_price = 60
        # total = 1250 + (-5 * 60) = 1250 - 300 = 950
        assert broker.total_value(60.0) == 950.0

    def test_multiple_buys(self):
        """Multiple buy orders should accumulate position."""
        broker = Broker(cash=10000)
        broker.market_order("buy", 10, 50.0)
        broker.market_order("buy", 5, 60.0)

        assert broker.position == 15
        assert broker.cash == 10000 - 500 - 300

    def test_multiple_sells(self):
        """Multiple sell orders should reduce position."""
        broker = Broker(cash=1000)
        broker.market_order("buy", 20, 10.0)
        broker.market_order("sell", 5, 15.0)
        broker.market_order("sell", 3, 20.0)

        assert broker.position == 12
        assert broker.cash == 1000 - 200 + 75 + 60

    def test_round_trip_trade(self):
        """Buy and sell same quantity should return to zero position."""
        broker = Broker(cash=1000)
        initial_cash = broker.cash

        broker.market_order("buy", 10, 50.0)
        broker.market_order("sell", 10, 60.0)

        assert broker.position == 0
        # Should have made profit of (60-50)*10 = 100
        assert broker.cash == initial_cash + 100

    def test_fractional_prices(self):
        """Broker should handle fractional prices correctly."""
        broker = Broker(cash=1000)
        broker.market_order("buy", 7, 12.345)

        assert broker.position == 7
        assert abs(broker.cash - (1000 - 7 * 12.345)) < 1e-10

    def test_large_quantities(self):
        """Broker should handle large quantities."""
        broker = Broker(cash=1_000_000_000)
        broker.market_order("buy", 1_000_000, 100.0)

        assert broker.position == 1_000_000
        assert broker.cash == 1_000_000_000 - 100_000_000

    def test_total_value_with_zero_position(self):
        """total_value with zero position should equal cash."""
        broker = Broker(cash=5000)
        assert broker.total_value(99999.0) == 5000.0

    def test_total_value_with_zero_price(self):
        """total_value should handle zero price (edge case)."""
        broker = Broker(cash=1000)
        broker.market_order("buy", 10, 50.0)
        # Even with worthless stock, cash remains
        # cash = 1000 - 500 = 500, position = 10, current_price = 0.0
        # total = 500 + 10 * 0.0 = 500
        assert broker.total_value(0.0) == 500.0

    def test_trade_log_preserves_order(self):
        """Trade log should preserve chronological order."""
        broker = Broker(cash=10000)
        broker.market_order("buy", 1, 100.0)
        broker.market_order("sell", 1, 110.0)
        broker.market_order("buy", 2, 105.0)

        assert broker.trade_log[0]["side"] == "buy"
        assert broker.trade_log[1]["side"] == "sell"
        assert broker.trade_log[2]["side"] == "buy"

    def test_independent_broker_instances(self):
        """Multiple broker instances should be independent."""
        broker1 = Broker(cash=1000)
        broker2 = Broker(cash=2000)

        broker1.market_order("buy", 5, 10.0)

        assert broker1.cash == 950
        assert broker2.cash == 2000
        assert broker1.position == 5
        assert broker2.position == 0
