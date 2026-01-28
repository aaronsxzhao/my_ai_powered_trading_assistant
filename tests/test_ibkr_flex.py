from __future__ import annotations

from datetime import datetime


def test_extract_trade_fills_from_statement_parses_trades():
    from app.data.ibkr_flex import extract_trade_fills_from_statement

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<FlexQueryResponse queryName="Test" type="AF">
  <FlexStatements count="1">
    <FlexStatement accountId="U123" fromDate="20260101" toDate="20260102">
      <Trades>
        <Trade assetCategory="STK" symbol="AAPL" tradeDate="20260102" tradeTime="09:31:00" quantity="100" tradePrice="10.0" buySell="BUY" currency="USD" ibCommission="-1.0"/>
        <Trade assetCategory="STK" symbol="AAPL" tradeDate="20260102" tradeTime="10:00:00" quantity="100" tradePrice="11.0" buySell="SELL" currency="USD" ibCommission="-1.0"/>
      </Trades>
    </FlexStatement>
  </FlexStatements>
</FlexQueryResponse>
"""

    fills = extract_trade_fills_from_statement(xml, allowed_asset_categories={"STK"})
    assert len(fills) == 2
    assert fills[0].symbol == "AAPL"
    assert fills[0].side == "buy"
    assert fills[0].quantity == 100.0
    assert fills[0].price == 10.0
    assert fills[0].currency == "USD"
    assert fills[0].commission == -1.0
    assert isinstance(fills[0].time, datetime)


def test_aggregate_fills_to_round_trips_simple_long():
    from app.data.ibkr_flex import IBKRTradeFill, aggregate_fills_to_round_trips

    fills = [
        IBKRTradeFill(
            symbol="AAPL",
            time=datetime(2026, 1, 2, 9, 31, 0),
            side="buy",
            quantity=100,
            price=10.0,
            currency="USD",
            commission=-1.0,
            asset_category="STK",
        ),
        IBKRTradeFill(
            symbol="AAPL",
            time=datetime(2026, 1, 2, 10, 0, 0),
            side="sell",
            quantity=100,
            price=11.0,
            currency="USD",
            commission=-1.0,
            asset_category="STK",
        ),
    ]

    trades = aggregate_fills_to_round_trips(fills)
    assert len(trades) == 1
    t = trades[0]
    assert t["ticker"] == "AAPL"
    assert t["direction"] == "long"
    assert t["entry_price"] == 10.0
    assert t["exit_price"] == 11.0
    assert t["size"] == 100
    assert t["fees"] == 2.0


def test_aggregate_fills_to_round_trips_reversal_splits_trades():
    from app.data.ibkr_flex import IBKRTradeFill, aggregate_fills_to_round_trips

    fills = [
        # Open long 100
        IBKRTradeFill(
            symbol="AAPL",
            time=datetime(2026, 1, 2, 9, 31, 0),
            side="buy",
            quantity=100,
            price=10.0,
            currency="USD",
        ),
        # Sell 200: closes long 100 and opens short 100
        IBKRTradeFill(
            symbol="AAPL",
            time=datetime(2026, 1, 2, 9, 45, 0),
            side="sell",
            quantity=200,
            price=11.0,
            currency="USD",
        ),
        # Close short 100
        IBKRTradeFill(
            symbol="AAPL",
            time=datetime(2026, 1, 2, 10, 5, 0),
            side="buy",
            quantity=100,
            price=12.0,
            currency="USD",
        ),
    ]

    trades = aggregate_fills_to_round_trips(fills)
    assert len(trades) == 2

    t1, t2 = trades
    assert t1["direction"] == "long"
    assert t1["entry_price"] == 10.0
    assert t1["exit_price"] == 11.0
    assert t1["size"] == 100

    assert t2["direction"] == "short"
    assert t2["entry_price"] == 11.0
    assert t2["exit_price"] == 12.0
    assert t2["size"] == 100

