from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path


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


def test_parse_ibkr_datetime_formats():
    """Test parsing of various IBKR datetime formats."""
    from app.journal.ingest import TradeIngester

    ingester = TradeIngester(use_llm_classification=False)

    # Format: YYYYMMDD;HHMMSS
    dt = ingester._parse_ibkr_datetime("20260106;100229")
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 1
    assert dt.day == 6
    assert dt.hour == 10
    assert dt.minute == 2
    assert dt.second == 29

    # Format: YYYYMMDD alone
    dt = ingester._parse_ibkr_datetime("20260106")
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 1
    assert dt.day == 6
    assert dt.hour == 0

    # Format: date + time separately
    dt = ingester._parse_ibkr_datetime("20260106", "100229")
    assert dt is not None
    assert dt.hour == 10
    assert dt.minute == 2

    # Format: date + time with colons
    dt = ingester._parse_ibkr_datetime("20260106", "10:02:29")
    assert dt is not None
    assert dt.hour == 10
    assert dt.minute == 2

    # ISO format
    dt = ingester._parse_ibkr_datetime("2026-01-06T10:02:29")
    assert dt is not None
    assert dt.hour == 10


def test_import_ibkr_csv_with_msg_row():
    """Test importing IBKR CSV that has MSG row at the top."""
    from app.journal.ingest import TradeIngester

    csv_content = '''"MSG","Realized P/L is not ready and has been disabled for this statement."
"ClientAccountID","AccountAlias","Model","CurrencyPrimary","FXRateToBase","AssetClass","SubCategory","Symbol","Description","Conid","SecurityID","SecurityIDType","CUSIP","ISIN","FIGI","ListingExchange","UnderlyingConid","UnderlyingSymbol","UnderlyingSecurityID","UnderlyingListingExchange","Issuer","IssuerCountryCode","TradeID","Multiplier","RelatedTradeID","Strike","ReportDate","Expiry","DateTime","Put/Call","TradeDate","PrincipalAdjustFactor","SettleDateTarget","TransactionType","Exchange","Quantity","TradePrice","TradeMoney","Proceeds","Taxes","IBCommission","IBCommissionCurrency","NetCash","NetCashInBase","ClosePrice","Open/CloseIndicator","Notes/Codes","CostBasis","FifoPnlRealized","CapitalGainsPnl","FxPnl","MtmPnl","OrigTradePrice","OrigTradeDate","OrigTradeID","OrigOrderID","OrigTransactionID","Buy/Sell","ClearingFirmID","IBOrderID","TransactionID","IBExecID","RelatedTransactionID","RTN","BrokerageOrderID","OrderReference","VolatilityOrderLink","ExchOrderID","ExtExecID","OrderTime","OpenDateTime","HoldingPeriodDateTime","WhenRealized","WhenReopened","LevelOfDetail","ChangeInPrice","ChangeInQuantity","OrderType","TraderID","IsAPIOrder","AccruedInterest","InitialInvestment","PositionActionID","SerialNumber","DeliveryType","CommodityType","Fineness","Weight"
"U16762989","","","USD","1","STK","ETF","TESTX","TEST STOCK","12345","US12345","ISIN","12345","US12345","BBG123","ARCA","","TESTX","","","","US","111111","1","","","20260106","","20260106;093000","","20260106","","20260107","ExchTrade","ARCA","100","50.00","5000","-5000","0","-1","USD","-5001","-5001","50.50","","","5001","0","0","0","50","0","","","0","0","BUY","","123456","111111","exec1","","","order1","","","N/A","ext1","20260106;093000","","","","","EXECUTION","0","0","LMT","","N","0","","","","","","0.0","0.0"
"U16762989","","","USD","1","STK","ETF","TESTX","TEST STOCK","12345","US12345","ISIN","12345","US12345","BBG123","ARCA","","TESTX","","","","US","222222","1","","","20260106","","20260106;100000","","20260106","","20260107","ExchTrade","ARCA","-100","55.00","-5500","5500","0","-1","USD","5499","5499","55.00","","","","0","0","0","","0","","","0","0","SELL","","123457","222222","exec2","","","order2","","","N/A","ext2","20260106;100000","","","","","EXECUTION","0","0","LMT","","N","0","","","","","","0.0","0.0"
'''

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = Path(f.name)

    try:
        ingester = TradeIngester(use_llm_classification=False)
        imported, errors, messages = ingester._import_interactive_brokers_csv(
            tmp_path, skip_errors=True, input_timezone="America/New_York"
        )

        # Should have found fills and created a trade
        assert errors == 0, f"Unexpected errors: {messages}"
        # The fills should create 1 round-trip trade (buy 100 @ 50, sell 100 @ 55)
        assert any("1 round-trip" in m or "fills" in m.lower() for m in messages), messages
    finally:
        tmp_path.unlink()


def test_import_ibkr_csv_skips_summary_rows():
    """Test that IBKR CSV import skips ASSET_SUMMARY and SYMBOL_SUMMARY rows."""
    from app.journal.ingest import TradeIngester

    csv_content = '''"Symbol","DateTime","Quantity","TradePrice","Buy/Sell","AssetClass","CurrencyPrimary","IBCommission","LevelOfDetail"
"TESTX","","1000","","","STK","","","ASSET_SUMMARY"
"TESTX","","500","50.00","","STK","USD","","SYMBOL_SUMMARY"
"TESTX","20260106;093000","100","50.00","BUY","STK","USD","-1","EXECUTION"
"TESTX","20260106;100000","-100","55.00","SELL","STK","USD","-1","EXECUTION"
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = Path(f.name)

    try:
        ingester = TradeIngester(use_llm_classification=False)
        imported, errors, messages = ingester._import_interactive_brokers_csv(
            tmp_path, skip_errors=True, input_timezone="America/New_York"
        )

        # Should skip the summary rows and only process EXECUTION rows
        assert any("summary" in m.lower() or "skipped" in m.lower() for m in messages), messages
        # Should have created fills from the EXECUTION rows
        assert any("fills" in m.lower() for m in messages), messages
    finally:
        tmp_path.unlink()


def test_import_ibkr_csv_handles_multi_leg_trade():
    """Test IBKR CSV import with multiple fills that create a round-trip trade."""
    from app.journal.ingest import TradeIngester

    # Simulates: Buy 100 @ 50, Buy 50 @ 51, Sell 150 @ 55
    csv_content = '''"Symbol","DateTime","Quantity","TradePrice","Buy/Sell","AssetClass","CurrencyPrimary","IBCommission","LevelOfDetail"
"GLDM","20260106;093000","100","50.00","BUY","STK","USD","-1","EXECUTION"
"GLDM","20260106;093500","50","51.00","BUY","STK","USD","-0.5","EXECUTION"
"GLDM","20260106;100000","-150","55.00","SELL","STK","USD","-1.5","EXECUTION"
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = Path(f.name)

    try:
        ingester = TradeIngester(use_llm_classification=False)
        imported, errors, messages = ingester._import_interactive_brokers_csv(
            tmp_path, skip_errors=True, input_timezone="America/New_York"
        )

        # Should have found 3 fills and created 1 round-trip trade
        assert errors == 0, f"Unexpected errors: {messages}"
        assert any("3 fills" in m for m in messages), messages
        assert any("1 round-trip" in m for m in messages), messages
    finally:
        tmp_path.unlink()
