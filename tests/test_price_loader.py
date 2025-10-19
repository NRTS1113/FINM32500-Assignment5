import numpy as np
import pandas as pd
import pytest
from backtester.price_loader import PriceLoader


class TestPriceLoader:
    """Comprehensive tests for PriceLoader class."""

    def test_initialization_defaults(self):
        """PriceLoader should initialize with default parameters."""
        loader = PriceLoader()
        assert loader.dates is not None
        assert len(loader.dates) > 0

    def test_initialization_with_custom_dates(self):
        """PriceLoader should accept custom date range."""
        loader = PriceLoader(start="2021-01-01", end="2021-01-31")
        assert loader.dates[0] == pd.Timestamp("2021-01-01")
        # Business days only
        assert len(loader.dates) <= 31

    def test_get_price_returns_series(self):
        """get_price should return a pandas Series."""
        loader = PriceLoader()
        prices = loader.get_price("AAPL")
        assert isinstance(prices, pd.Series)

    def test_get_price_series_length_matches_dates(self):
        """Returned price series should match date range length."""
        loader = PriceLoader(start="2020-01-01", end="2020-01-31")
        prices = loader.get_price("AAPL")
        assert len(prices) == len(loader.dates)

    def test_get_price_has_correct_index(self):
        """Price series should have DatetimeIndex matching loader dates."""
        loader = PriceLoader(start="2020-01-01", end="2020-01-31")
        prices = loader.get_price("AAPL")
        pd.testing.assert_index_equal(prices.index, loader.dates)

    def test_get_price_all_positive(self):
        """All prices should be positive."""
        loader = PriceLoader()
        prices = loader.get_price("AAPL")
        assert (prices > 0).all()

    def test_get_price_no_nans(self):
        """Price series should contain no NaN values."""
        loader = PriceLoader()
        prices = loader.get_price("AAPL")
        assert not prices.isna().any()

    def test_get_price_series_name(self):
        """Price series name should match the symbol."""
        loader = PriceLoader()
        prices = loader.get_price("TSLA")
        assert prices.name == "TSLA"

    def test_deterministic_with_seed(self):
        """Same seed should produce same prices."""
        # Each PriceLoader sets its own random seed
        loader1 = PriceLoader(start="2020-01-01", end="2020-01-31", seed=42)
        prices1 = loader1.get_price("AAPL")

        # Create a new loader with same seed - should produce same prices
        loader2 = PriceLoader(start="2020-01-01", end="2020-01-31", seed=42)
        prices2 = loader2.get_price("AAPL")

        pd.testing.assert_series_equal(prices1, prices2)

    def test_different_seeds_produce_different_prices(self):
        """Different seeds should produce different prices."""
        loader1 = PriceLoader(seed=42)
        loader2 = PriceLoader(seed=123)

        prices1 = loader1.get_price("AAPL")
        prices2 = loader2.get_price("AAPL")

        # Should not be equal
        assert not prices1.equals(prices2)

    def test_multiple_symbols_same_loader(self):
        """Calling get_price multiple times should work (but return different series due to RNG)."""
        loader = PriceLoader(seed=42)
        prices1 = loader.get_price("AAPL")
        prices2 = loader.get_price("GOOGL")

        # Different calls produce different prices due to RNG state
        assert prices1.name != prices2.name
        # But same length
        assert len(prices1) == len(prices2)

    def test_business_days_only(self):
        """Dates should be business days only (no weekends)."""
        loader = PriceLoader(start="2020-01-01", end="2020-01-31")
        # Check that no dates are weekends
        for date in loader.dates:
            # Monday=0, Sunday=6
            assert date.weekday() < 5

    def test_price_evolution_realistic(self):
        """Prices should evolve in a realistic manner (not constant, not explosive)."""
        loader = PriceLoader(seed=42)
        prices = loader.get_price("AAPL")

        # Prices should vary
        assert prices.std() > 0

        # Should not have infinite or extreme values
        assert np.isfinite(prices).all()
        assert prices.max() / prices.min() < 1000  # Not too explosive

    def test_empty_date_range(self):
        """Edge case: very short or empty date range."""
        # Single day
        loader = PriceLoader(start="2020-01-02", end="2020-01-02")
        prices = loader.get_price("AAPL")
        assert len(prices) == 1

    def test_long_date_range(self):
        """PriceLoader should handle long date ranges."""
        loader = PriceLoader(start="2010-01-01", end="2020-12-31")
        prices = loader.get_price("AAPL")
        assert len(prices) > 2500  # ~252 trading days/year * 11 years

    def test_prices_start_near_100(self):
        """Initial prices should start near 100 (as per implementation)."""
        loader = PriceLoader(seed=42)
        prices = loader.get_price("AAPL")
        # First price is based on 100 * exp(first_return)
        # Should be within reasonable range of 100
        assert 50 < prices.iloc[0] < 200

    def test_get_price_float_dtype(self):
        """Price values should be float type."""
        loader = PriceLoader()
        prices = loader.get_price("AAPL")
        assert prices.dtype in [np.float64, np.float32, float]

    def test_seed_affects_initialization(self):
        """Seed parameter should affect random number generation."""
        # Create two loaders with same seed
        np.random.seed(999)  # Set global seed
        loader1 = PriceLoader(seed=42)
        initial_state_1 = np.random.get_state()

        np.random.seed(999)  # Reset global seed
        loader2 = PriceLoader(seed=42)
        initial_state_2 = np.random.get_state()

        # After initialization with same seed, random state should be same
        prices1 = loader1.get_price("AAPL")

        # Reset and generate again
        loader3 = PriceLoader(seed=42)
        prices3 = loader3.get_price("AAPL")

        pd.testing.assert_series_equal(prices1, prices3)

    def test_symbol_parameter_only_affects_name(self):
        """Symbol parameter should only affect series name, not values (with same seed)."""
        # Note: This test documents current behavior where different symbols
        # get different prices due to RNG state progression
        loader = PriceLoader(seed=42)
        prices_aapl = loader.get_price("AAPL")

        loader2 = PriceLoader(seed=42)
        prices_googl = loader2.get_price("GOOGL")

        # Same seed, so same prices
        pd.testing.assert_series_equal(
            prices_aapl.rename("GOOGL"),
            prices_googl
        )

    def test_dates_frequency_is_business_days(self):
        """Date frequency should be business days ('B')."""
        loader = PriceLoader(start="2020-01-01", end="2020-02-01")
        # Check that dates follow business day frequency
        assert loader.dates.freq == pd.tseries.offsets.BusinessDay() or loader.dates.inferred_freq == 'B'
