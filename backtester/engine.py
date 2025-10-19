import pandas as pd
from backtester.strategy import VolatilityBreakoutStrategy
from backtester.broker import Broker


class Backtester:
    def __init__(self, strategy, broker):
        self.strategy = strategy
        self.broker = broker

    def run(self, prices: pd.Series):
        sig = self.strategy.signals(prices)
        sig = sig.fillna(0).astype(float)

        sig = sig.clip(lower=0, upper=1)

        prev_sig = 0.0
        equity = []

        for dt, px in prices.items():
            cur_sig = float(sig.loc[dt])

            if prev_sig <= 0 and cur_sig > 0:
                qty = int(self.broker.cash // px)
                if qty > 0:
                    self.broker.market_order("buy", qty=qty, price=float(px))
            elif prev_sig > 0 and cur_sig <= 0:
                pos = int(self.broker.position)
                if pos > 0:
                    self.broker.market_order("sell", qty=pos, price=float(px))

            equity.append((dt, self.broker.total_value(float(px))))

            prev_sig = cur_sig

        equity_series = pd.Series(
            data=[v for _, v in equity],
            index=[d for d, _ in equity],
            name="equity"
        )

        trades_df = pd.DataFrame(self.broker.trade_log) if getattr(self.broker, "trade_log", None) else pd.DataFrame()

        return {
            "equity": equity_series,
            "signals": sig.rename("signal"),
            "trades": trades_df
        }