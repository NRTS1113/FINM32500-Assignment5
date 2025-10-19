class Broker:
    def __init__(self, cash: float = 1_000_000):
        self.cash = cash
        self.position = 0
        self.trade_log = []

    def market_order(self, side: str, qty: int, price: float):
        if qty <= 0:
            raise ValueError("Quantity must be positive.")
        if price <= 0:
            raise ValueError("Price must be positive.")
        
        if side.lower() == "buy":
            self.cash -= qty * price
            self.position += qty
        elif side.lower() == "sell":
            self.cash += qty * price
            self.position -= qty

        self.trade_log.append({
            "side": side,
            "qty": qty,
            "price": price,
            "cash": self.cash,
            "position": self.position
        })

    def total_value(self, current_price: float) -> float:
        return self.cash + self.position * current_price