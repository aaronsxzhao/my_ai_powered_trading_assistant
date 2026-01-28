from datetime import date, datetime, timezone

import pandas as pd


def test_llm_analyzer_classify_trade_setup_normalizes_keys():
    """Ensure trade classification output maps to strategy_name/strategy_category."""
    from app.llm.analyzer import LLMAnalyzer

    analyzer = LLMAnalyzer()

    # Stub the network call to return a minimal JSON payload using the prompt's key names.
    analyzer._call_llm = (
        lambda *args, **kwargs: """```json
{
  "primary_setup": "second_entry_long",
  "secondary_setup": null,
  "setup_category": "trend",
  "is_second_entry": true,
  "trend_alignment": "with_trend",
  "confidence": "high",
  "signal_bar_quality": "strong",
  "entry_location": "good",
  "reasoning": "2nd entry buy in an uptrend pullback."
}
```"""
    )

    result = analyzer.classify_trade_setup(
        ticker="SPY",
        direction="long",
        entry_price=100.0,
        exit_price=101.0,
        stop_price=99.0,
        entry_reason="Second entry buy",
        notes="",
        timeframe="5m",
    )

    assert result["strategy_name"] == "second_entry_long"
    assert result["strategy_category"] == "trend"
    assert result["confidence"] == "high"


def test_tradecoach_llm_review_maps_setup_quality_and_always_in():
    """Ensure LLM trade review maps letter grades to good/marginal/poor and normalizes always-in."""
    from app.journal.coach import TradeCoach
    from app.journal.models import Trade, TradeDirection, init_db, get_session

    init_db()

    # Create a trade to review.
    with get_session() as session:
        trade = Trade(
            ticker="AI_TEST",
            trade_date=date.today(),
            timeframe="5m",
            direction=TradeDirection.LONG,
            entry_price=100.0,
            exit_price=101.0,
            stop_loss=99.0,
            size=1.0,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
        )
        trade.compute_metrics()
        session.add(trade)
        session.commit()
        trade_id = trade.id

    coach = TradeCoach()

    # Avoid any market-data calls.
    coach._get_ohlcv_context_string = lambda *args, **kwargs: ("", pd.DataFrame())
    coach._get_session_context = lambda *args, **kwargs: (None, None, None, None)
    coach._save_llm_log = lambda *args, **kwargs: None

    class StubAnalyzer:
        is_available = True

        def analyze_trade(self, *args, **kwargs):
            return {
                "context": {
                    "daily_regime": "trend_up",
                    "two_hour_regime": "trend_up",
                    "trading_tf_regime": "trend_up",
                    "always_in_direction": "not_clear",
                    "trade_aligned_with_always_in": True,
                },
                "setup": {
                    "primary_label": "breakout_pullback",
                    "secondary_label": None,
                    "category": "trend",
                    "is_second_entry": False,
                    "with_trend_or_counter": "with_trend",
                },
                "entry_quality": {
                    "signal_bar_quality": "strong",
                    "signal_bar_notes": "Strong bull bar",
                    "entry_location": "right_place",
                    "entry_type": "stop",
                    "entry_quality_score": "A",
                },
                "risk_reward": {
                    "stop_placement": "correct",
                    "stop_notes": "Below swing low",
                    "target_logic": "swing",
                    "target_notes": "Measured move",
                    "traders_equation": "favorable",
                    "probability_estimate": "HIGH_60%+",
                },
                "management": {
                    "exit_quality": "good",
                    "management_notes": "Exited into strength",
                    "scalp_vs_swing_match": True,
                    "got_trapped": False,
                },
                "coaching": {
                    "what_was_good": ["With-trend entry"],
                    "what_was_flawed": ["Entry was slightly late"],
                    "selection_vs_execution": "execution_error",
                    "better_alternative": "Enter earlier on the first pullback",
                    "keep_doing": "Wait for good signal bars",
                    "stop_doing": "Chasing late entries",
                    "rules_for_next_20_trades": ["Only enter on strong signal bars near support"],
                },
                "grade": "B",
                "grade_explanation": "Good trade, minor execution issue",
                "coaching_summary": "Uptrend context; focus on earlier entries.",
            }

    coach._llm_analyzer = StubAnalyzer()

    review = coach.review_trade(trade_id)
    assert review is not None
    assert review.setup_quality == "good"  # "A" -> good
    assert review.always_in == "neutral"  # not_clear -> neutral


def test_llm_analyzer_is_available_requires_api_key(monkeypatch):
    """Analyzer availability is driven by presence of an API key."""
    from app.llm.analyzer import LLMAnalyzer

    # Ensure no key is present.
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    analyzer = LLMAnalyzer()
    assert analyzer.is_available is False

    # Add key and verify availability flips on.
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    analyzer2 = LLMAnalyzer()
    assert analyzer2.is_available is True
