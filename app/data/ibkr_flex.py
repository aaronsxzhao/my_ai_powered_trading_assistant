"""
Interactive Brokers (IBKR) Flex Web Service integration.

This module implements a **read-only** import path using IBKR Flex Web Service v3.
It fetches executions from a user-defined Flex Query and converts them into
"round-trip" trades (position returns to flat).

Docs:
- https://www.ibkrguides.com/clientportal/performanceandstatements/flex3.htm
- https://www.ibkrguides.com/clientportal/performanceandstatements/flex3error.htm
"""

from __future__ import annotations

import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

IBKR_FLEX_BASE_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
IBKR_FLEX_VERSION = "3"
DEFAULT_USER_AGENT = f"Python/{sys.version_info.major}.{sys.version_info.minor} brooks-trading-coach"


class IBKRFlexError(RuntimeError):
    """Raised when IBKR Flex Web Service returns an error."""

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class IBKRTradeFill:
    """Normalized execution fill from IBKR Flex output."""

    symbol: str
    time: datetime
    side: str  # "buy" | "sell"
    quantity: float
    price: float
    currency: str = "USD"
    commission: float | None = None
    asset_category: str | None = None
    exchange: str | None = None
    execution_id: str | None = None  # IBKR IBExecID for deduplication


def get_ibkr_flex_token() -> str | None:
    token = os.getenv("IBKR_FLEX_TOKEN")
    token = token.strip() if token else None
    return token or None


def get_ibkr_flex_query_id() -> str | None:
    query_id = os.getenv("IBKR_FLEX_QUERY_ID")
    query_id = query_id.strip() if query_id else None
    return query_id or None


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find_text(root: ET.Element, child_name: str) -> str | None:
    """Find direct child text by tag name, namespace-agnostic."""
    for child in list(root):
        if _strip_ns(child.tag) == child_name:
            return (child.text or "").strip() or None
    return None


def _parse_ibkr_datetime(date_str: str, time_str: str | None = None) -> datetime | None:
    """
    Parse common IBKR date/time formats from Flex reports.

    Supports:
    - yyyymmdd with optional HH:MM:SS
    - yyyy-mm-dd with optional HH:MM:SS
    - dateTime fields like "yyyymmdd;HH:MM:SS" or ISO strings
    """
    if not date_str:
        return None

    raw = date_str.strip()

    # If combined dateTime provided (e.g., 20250113;09:35:00)
    if ";" in raw and time_str is None:
        parts = raw.split(";")
        if len(parts) == 2:
            raw, time_str = parts[0], parts[1]

    # ISO format
    if "T" in raw or ("-" in raw and ":" in raw):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass

    # Normalize date formats to yyyymmdd
    date_only = raw.replace("-", "")
    if len(date_only) != 8 or not date_only.isdigit():
        return None

    if not time_str:
        try:
            return datetime.strptime(date_only, "%Y%m%d")
        except ValueError:
            return None

    ts = time_str.strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(ts, fmt).time()
            d = datetime.strptime(date_only, "%Y%m%d").date()
            return datetime.combine(d, t)
        except ValueError:
            continue

    return None


class IBKRFlexClient:
    """Minimal IBKR Flex Web Service v3 client."""

    def __init__(
        self,
        *,
        base_url: str = IBKR_FLEX_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def send_request(
        self,
        *,
        token: str,
        query_id: str,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> str:
        """
        Start statement generation and return a reference code.
        """
        url = f"{self.base_url}/SendRequest"
        params: dict[str, str] = {"t": token, "q": str(query_id), "v": IBKR_FLEX_VERSION}
        if from_date and to_date:
            params["fd"] = from_date.strftime("%Y%m%d")
            params["td"] = to_date.strftime("%Y%m%d")

        headers = {"User-Agent": self.user_agent}
        resp = httpx.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
        resp.raise_for_status()

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as e:
            raise IBKRFlexError(f"IBKR Flex SendRequest returned invalid XML: {e}")

        status = _find_text(root, "Status")
        if status != "Success":
            code = _find_text(root, "ErrorCode")
            msg = _find_text(root, "ErrorMessage") or "IBKR Flex request failed"
            raise IBKRFlexError(f"{msg} (ErrorCode {code})", code=code)

        ref = _find_text(root, "ReferenceCode")
        if not ref:
            raise IBKRFlexError("IBKR Flex SendRequest succeeded but no ReferenceCode was returned.")
        return ref

    def get_statement(
        self,
        *,
        token: str,
        reference_code: str,
        max_wait_seconds: float = 30.0,
    ) -> str:
        """
        Retrieve statement XML. Polls briefly if generation is still in progress.
        """
        url = f"{self.base_url}/GetStatement"
        params: dict[str, str] = {"t": token, "q": str(reference_code), "v": IBKR_FLEX_VERSION}
        headers = {"User-Agent": self.user_agent}

        start = time.monotonic()
        delay = 1.0

        while True:
            resp = httpx.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
            resp.raise_for_status()
            text = resp.text

            # If the statement isn't ready, IBKR returns FlexStatementResponse with Fail/ErrorCode.
            try:
                root = ET.fromstring(text)
            except ET.ParseError:
                # If we can't parse it, return raw text (caller can handle).
                return text

            if _strip_ns(root.tag) == "FlexStatementResponse":
                status = _find_text(root, "Status")
                if status == "Fail":
                    code = _find_text(root, "ErrorCode")
                    msg = _find_text(root, "ErrorMessage") or "IBKR Flex GetStatement failed"

                    # 1019: generation in progress. Also treat "try again shortly" codes as retryable.
                    retryable = code in {"1001", "1004", "1005", "1006", "1007", "1008", "1009", "1019", "1021"}
                    elapsed = time.monotonic() - start
                    if retryable and elapsed < max_wait_seconds:
                        time.sleep(delay)
                        delay = min(delay * 2.0, 5.0)
                        continue
                    raise IBKRFlexError(f"{msg} (ErrorCode {code})", code=code)

            # Otherwise, assume this is the actual Flex statement payload.
            return text


def extract_trade_fills_from_statement(
    statement_xml: str,
    *,
    allowed_asset_categories: set[str] | None = None,
) -> list[IBKRTradeFill]:
    """
    Extract normalized trade fills from a Flex statement XML payload.
    """
    try:
        root = ET.fromstring(statement_xml)
    except ET.ParseError as e:
        raise IBKRFlexError(f"Could not parse Flex statement XML: {e}")

    fills: list[IBKRTradeFill] = []

    for el in root.iter():
        if _strip_ns(el.tag) != "Trade":
            continue

        attrs = el.attrib or {}

        symbol = (
            attrs.get("symbol")
            or attrs.get("underlyingSymbol")
            or attrs.get("symbolUnderlying")
            or attrs.get("ticker")
        )
        if not symbol:
            continue
        symbol = str(symbol).strip()
        if not symbol:
            continue

        asset_category = attrs.get("assetCategory") or attrs.get("asset_category")
        if allowed_asset_categories and asset_category and asset_category not in allowed_asset_categories:
            continue

        price_raw = attrs.get("tradePrice") or attrs.get("price") or attrs.get("trade_price")
        qty_raw = attrs.get("quantity") or attrs.get("qty") or attrs.get("tradeQuantity")
        if price_raw is None or qty_raw is None:
            continue

        try:
            price = float(price_raw)
            qty_val = float(qty_raw)
        except (TypeError, ValueError):
            continue

        side_raw = (attrs.get("buySell") or attrs.get("side") or "").strip().lower()
        if side_raw in {"buy", "bot"}:
            side = "buy"
        elif side_raw in {"sell", "sld"}:
            side = "sell"
        else:
            side = "buy" if qty_val >= 0 else "sell"

        quantity = abs(qty_val)
        if quantity <= 0 or price <= 0:
            continue

        # Date/time: prefer dateTime, else tradeDate + tradeTime
        dt = None
        if attrs.get("dateTime"):
            dt = _parse_ibkr_datetime(str(attrs.get("dateTime")))
        if dt is None and attrs.get("tradeDate"):
            dt = _parse_ibkr_datetime(str(attrs.get("tradeDate")), str(attrs.get("tradeTime") or ""))
        if dt is None and attrs.get("tradeDateTime"):
            dt = _parse_ibkr_datetime(str(attrs.get("tradeDateTime")))
        if dt is None:
            # Some reports include just "tradeDate" without time; skip for now.
            continue

        currency = (attrs.get("currency") or "USD").strip().upper()

        commission = None
        for k in ("ibCommission", "commission", "commissions"):
            if k in attrs and attrs[k] is not None and str(attrs[k]).strip() != "":
                try:
                    commission = float(attrs[k])
                except (TypeError, ValueError):
                    commission = None
                break

        fills.append(
            IBKRTradeFill(
                symbol=symbol,
                time=dt,
                side=side,
                quantity=quantity,
                price=price,
                currency=currency,
                commission=commission,
                asset_category=asset_category,
            )
        )

    return fills


def aggregate_fills_to_round_trips(fills: list[IBKRTradeFill], *, include_fills: bool = False) -> list[dict]:
    """
    Convert fills into round-trip trades.

    Strategy:
    - Track position per symbol.
    - When a position returns to 0, emit one aggregated trade record.
    - If a fill *crosses* through 0 (reversal), split it into close + new-open.
    
    Args:
        fills: List of execution fills
        include_fills: If True, include the fills list in each trade dict under "fills" key
    
    Returns:
        List of trade dicts. If include_fills=True, each trade includes a "fills" key
        with the list of IBKRTradeFill objects that make up the trade.
    """

    def vwap(fs: list[IBKRTradeFill]) -> float:
        total_qty = sum(f.quantity for f in fs)
        if total_qty <= 0:
            return 0.0
        return sum(f.quantity * f.price for f in fs) / total_qty

    # Group by symbol
    by_symbol: dict[str, list[IBKRTradeFill]] = {}
    for f in fills:
        by_symbol.setdefault(f.symbol, []).append(f)

    trades: list[dict] = []

    for symbol, fs in by_symbol.items():
        fs_sorted = sorted(fs, key=lambda x: x.time)

        position = 0.0  # +long, -short
        pending: list[IBKRTradeFill] = []

        def emit_trade(pending_fills: list[IBKRTradeFill]) -> None:
            if not pending_fills:
                return

            opening = pending_fills[0]
            direction = "long" if opening.side == "buy" else "short"

            if direction == "long":
                entry_fills = [f for f in pending_fills if f.side == "buy"]
                exit_fills = [f for f in pending_fills if f.side == "sell"]
            else:
                entry_fills = [f for f in pending_fills if f.side == "sell"]
                exit_fills = [f for f in pending_fills if f.side == "buy"]

            entry_qty = sum(f.quantity for f in entry_fills)
            exit_qty = sum(f.quantity for f in exit_fills)
            if entry_qty <= 0 or exit_qty <= 0:
                return

            entry_price = vwap(entry_fills)
            exit_price = vwap(exit_fills)
            if entry_price <= 0 or exit_price <= 0:
                return

            fees = 0.0
            has_fee = False
            for f in pending_fills:
                if f.commission is not None:
                    has_fee = True
                    fees += abs(float(f.commission))

            trade_dict = {
                "ticker": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": entry_qty,
                "entry_time": min(f.time for f in entry_fills),
                "exit_time": max(f.time for f in exit_fills),
                "currency": opening.currency or "USD",
                "fees": fees if has_fee else None,
            }
            
            # Include fills if requested
            if include_fills:
                trade_dict["fills"] = list(pending_fills)
            
            trades.append(trade_dict)

        for fill in fs_sorted:
            signed = fill.quantity if fill.side == "buy" else -fill.quantity

            # If reversal through zero, split the fill into "close" + "new open".
            if pending and position != 0 and (position + signed) * position < 0:
                qty_to_flat = abs(position)
                qty_total = abs(signed)
                qty_remain = max(qty_total - qty_to_flat, 0.0)

                if qty_to_flat > 0:
                    close_fill = IBKRTradeFill(
                        symbol=fill.symbol,
                        time=fill.time,
                        side=fill.side,
                        quantity=qty_to_flat,
                        price=fill.price,
                        currency=fill.currency,
                        commission=fill.commission,
                        asset_category=fill.asset_category,
                        exchange=fill.exchange,
                        execution_id=fill.execution_id,
                    )
                    pending.append(close_fill)
                    position = 0.0
                    emit_trade(pending)
                    pending = []

                if qty_remain > 0:
                    open_fill = IBKRTradeFill(
                        symbol=fill.symbol,
                        time=fill.time,
                        side=fill.side,
                        quantity=qty_remain,
                        price=fill.price,
                        currency=fill.currency,
                        commission=fill.commission,
                        asset_category=fill.asset_category,
                        exchange=fill.exchange,
                        execution_id=fill.execution_id,
                    )
                    pending.append(open_fill)
                    position = open_fill.quantity if open_fill.side == "buy" else -open_fill.quantity
                continue

            position += signed
            pending.append(fill)

            if abs(position) < 1e-6 and pending:
                position = 0.0
                emit_trade(pending)
                pending = []

        # Discard unmatched open position at end
        if pending:
            logger.info(f"IBKR import: discarding {len(pending)} fills for {symbol} (still open position)")

    # Sort trades chronologically (oldest first)
    trades.sort(key=lambda t: t.get("exit_time") or t.get("entry_time") or datetime.min)
    return trades

