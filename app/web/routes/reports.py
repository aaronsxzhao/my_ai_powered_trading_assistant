"""
Report and batch-analysis API routes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from app.journal.coach import TradeCoach
from app.journal.models import Strategy, Trade, get_session
from app.web.dependencies import require_write_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["reports"])


@router.post("/generate-premarket", dependencies=[require_write_auth])
async def generate_premarket(ticker: str | None = Form(None)):
    """Generate premarket report (non-blocking)."""
    from app.reports.premarket import PremarketReport

    def _generate():
        generator = PremarketReport()
        if ticker:
            reports = [generator.generate_ticker_report(ticker)]
        else:
            reports = generator.generate_all_reports()
        output_dir = generator.save_reports(reports, date.today())
        return len(reports), str(output_dir)

    count, output_dir = await asyncio.to_thread(_generate)
    return JSONResponse({"message": f"Generated {count} reports", "path": output_dir})


@router.post("/generate-eod", dependencies=[require_write_auth])
async def generate_eod():
    """Generate end-of-day report (non-blocking)."""
    from app.reports.eod import EndOfDayReport

    def _generate():
        generator = EndOfDayReport()
        report = generator.generate_report()
        output_path = generator.save_report(report)
        return str(output_path)

    output_path = await asyncio.to_thread(_generate)
    return JSONResponse({"message": "Generated EOD report", "path": output_path})


@router.post("/recalculate-metrics", dependencies=[require_write_auth])
async def recalculate_all_metrics():
    """Recalculate R-multiple, P&L, and other metrics for all trades."""
    with get_session() as session:
        trades = session.query(Trade).all()
        updated = 0
        for trade in trades:
            trade.compute_metrics()
            updated += 1
        session.commit()

    return JSONResponse(
        {"updated": updated, "message": f"Recalculated metrics for {updated} trades"}
    )


@router.post("/analyze-all-trades", dependencies=[require_write_auth])
async def analyze_all_trades(force: bool = False):
    """
    Run AI Coaching Review on all unreviewed trades.

    By default, only analyzes trades that don't have a cached_review.
    Set force=true to re-analyze all trades.
    """
    from concurrent.futures import ThreadPoolExecutor
    import json as json_module

    logger.info(f"🧠 Analyze all trades called with force={force}")

    # Check if LLM is available
    from app.llm.analyzer import LLMAnalyzer

    analyzer = LLMAnalyzer()
    if not analyzer.is_available:
        logger.warning("LLM not available for analysis")
        return JSONResponse(
            {"error": "LLM not available. Check your API key in settings.", "analyzed": 0}
        )

    session = get_session()
    try:
        trades = session.query(Trade).all()
        trade_ids: list[int] = []
        skipped = 0

        for trade in trades:
            if not force and trade.cached_review and trade.review_generated_at:
                skipped += 1
                continue
            trade_ids.append(trade.id)

        logger.info(
            f"🧠 Processing {len(trade_ids)} trades for AI analysis (skipped {skipped} already reviewed)"
        )

        if not trade_ids:
            return JSONResponse(
                {
                    "analyzed": 0,
                    "skipped": skipped,
                    "errors": 0,
                    "message": f"All {skipped} trades already have reviews",
                }
            )

        def analyze_single(trade_id: int):
            try:
                coach = TradeCoach()
                review = coach.review_trade(trade_id)

                if not review:
                    return {"id": trade_id, "success": False, "error": "No review generated"}

                # Open a new session for this thread
                thread_session = get_session()
                try:
                    trade = thread_session.query(Trade).get(trade_id)
                    if not trade:
                        return {"id": trade_id, "success": False, "error": "Trade not found"}

                    ai_setup = review.setup_classification

                    # Store original AI classification permanently (never overwrite if already set)
                    if ai_setup and not trade.ai_setup_classification:
                        trade.ai_setup_classification = ai_setup

                    # Update trade's strategy from AI setup_classification (only if not manually set)
                    if ai_setup and ai_setup.lower() not in [
                        "unknown",
                        "unclassified",
                        "insufficient_information",
                    ]:
                        if not trade.strategy_id:
                            strategy = (
                                thread_session.query(Strategy)
                                .filter(Strategy.name == ai_setup)
                                .first()
                            )
                            if not strategy:
                                strategy = Strategy(
                                    name=ai_setup,
                                    description="Auto-created from AI coaching review",
                                )
                                thread_session.add(strategy)
                                thread_session.flush()
                            trade.strategy_id = strategy.id

                    current_strategy = trade.strategy.name if trade.strategy else ai_setup

                    review_dict = {
                        "grade": review.grade,
                        "grade_explanation": review.grade_explanation,
                        "regime": review.regime,
                        "always_in": review.always_in,
                        "context_description": review.context_description,
                        "setup_classification": current_strategy or ai_setup or "Unknown",
                        "ai_setup_classification": ai_setup,
                        "setup_quality": review.setup_quality,
                        "what_was_good": review.what_was_good or [],
                        "what_was_flawed": review.what_was_flawed or [],
                        "errors_detected": review.errors_detected or [],
                        "rule_for_next_time": review.rule_for_next_time or [],
                    }

                    # Cache the review
                    trade.cached_review = json_module.dumps(review_dict)
                    trade.review_generated_at = datetime.now(timezone.utc)
                    thread_session.commit()

                    return {"id": trade_id, "success": True, "strategy": current_strategy}
                finally:
                    thread_session.close()

            except Exception as e:
                logger.error(f"Error analyzing trade {trade_id}: {e}")
                return {"id": trade_id, "success": False, "error": str(e)}

        # Run with concurrent workers for LLM calls (configurable via LLM_WORKERS env var)
        from app.config import get_llm_workers

        num_workers = get_llm_workers()
        logger.info(f"🧠 Using {num_workers} concurrent LLM workers")

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = await loop.run_in_executor(
                None, lambda: list(executor.map(analyze_single, trade_ids))
            )

        analyzed = sum(1 for r in results if r.get("success"))
        errors = sum(1 for r in results if not r.get("success"))

        logger.info(
            f"✅ Analysis complete: {analyzed} analyzed, {skipped} skipped, {errors} errors"
        )

        return JSONResponse(
            {
                "analyzed": analyzed,
                "skipped": skipped,
                "errors": errors,
                "message": f"Analyzed {analyzed} trades, {errors} errors",
            }
        )
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return JSONResponse({"error": str(e), "analyzed": 0})
    finally:
        session.close()


@router.post("/bulk-analysis", dependencies=[require_write_auth])
async def bulk_analysis(request):
    """Analyze multiple trades using LLM."""
    from datetime import timedelta
    from app.llm.analyzer import LLMAnalyzer

    analyzer = LLMAnalyzer()

    if not analyzer.is_available:
        return JSONResponse(
            {"error": "LLM not available. Check your API key in settings.", "trade_count": 0}
        )

    data = await request.json()
    analysis_type = data.get("type", "count")  # 'count' or 'days'
    value = data.get("value", 10)

    session = get_session()
    try:
        if analysis_type == "days":
            cutoff_date = date.today() - timedelta(days=value)
            trades = (
                session.query(Trade)
                .filter(Trade.trade_date >= cutoff_date)
                .order_by(Trade.trade_number.desc())
                .all()
            )
        else:
            trades = session.query(Trade).order_by(Trade.trade_number.desc()).limit(value).all()

        if not trades:
            return JSONResponse(
                {"error": "No trades found for the selected criteria.", "trade_count": 0}
            )

        wins = sum(1 for t in trades if t.outcome and t.outcome.value == "win")
        losses = sum(1 for t in trades if t.outcome and t.outcome.value == "loss")
        r_values = [t.r_multiple for t in trades if t.r_multiple]
        total_r = sum(r_values) if r_values else 0
        avg_r = total_r / len(r_values) if r_values else 0
        win_rate = (wins / len(trades) * 100) if trades else 0

        # Strategy breakdown
        strategy_stats: dict[str, dict] = {}
        for trade in trades:
            strategy_name = trade.strategy.name if trade.strategy else "Unclassified"
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {"count": 0, "wins": 0, "r_values": []}
            strategy_stats[strategy_name]["count"] += 1
            if trade.outcome and trade.outcome.value == "win":
                strategy_stats[strategy_name]["wins"] += 1
            if trade.r_multiple:
                strategy_stats[strategy_name]["r_values"].append(trade.r_multiple)

        strategy_breakdown: dict[str, dict] = {}
        for name, stats in strategy_stats.items():
            strategy_breakdown[name] = {
                "count": stats["count"],
                "win_rate": (stats["wins"] / stats["count"] * 100) if stats["count"] > 0 else 0,
                "total_r": sum(stats["r_values"]) if stats["r_values"] else 0,
            }

        # Build trade summary for LLM
        trade_summaries = []
        for t in trades[:50]:
            summary = (
                f"#{t.id}: {t.ticker} {t.direction.value.upper()} "
                f"Entry=${t.entry_price:.2f} Exit=${t.exit_price:.2f if t.exit_price else 0:.2f} "
                f"R={t.r_multiple:.2f if t.r_multiple else 0:.2f} "
                f"Result={t.outcome.value.upper() if t.outcome else 'UNKNOWN'} "
                f"Strategy={t.strategy.name if t.strategy else 'unclassified'} "
                f"Duration={t.duration_display}"
            )
            if t.notes:
                summary += f" Notes: {t.notes[:100]}"
            trade_summaries.append(summary)

        system_prompt = """You are an expert Al Brooks price action trading coach analyzing a trader's recent performance.

Analyze the provided trades and identify:
1. PATTERNS: Recurring patterns in wins and losses
2. STRENGTHS: What the trader is doing well
3. WEAKNESSES: Areas that need improvement
4. RECOMMENDATIONS: Specific, actionable advice
5. STRATEGY ANALYSIS: Which strategies are working and which aren't

Be specific, use Brooks terminology, and provide actionable insights.

Respond in JSON format:
{
    "patterns": ["list of patterns identified"],
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "recommendations": ["list of specific recommendations"],
    "full_analysis": "A detailed 2-3 paragraph analysis of the trader's performance"
}"""

        user_prompt = f"""Analyze these {len(trades)} recent trades:

SUMMARY STATS:
- Win Rate: {win_rate:.1f}%
- Total R: {total_r:.2f}R
- Average R per trade: {avg_r:.2f}R
- Wins: {wins}, Losses: {losses}

TRADES:
{chr(10).join(trade_summaries)}

STRATEGY BREAKDOWN:
{chr(10).join(f"- {name}: {stats['count']} trades, {stats['win_rate']:.0f}% win rate, {stats['total_r']:.2f}R" for name, stats in strategy_breakdown.items())}

Provide a comprehensive analysis of this trader's performance with specific patterns, strengths, weaknesses, and recommendations."""

        # Call LLM (non-blocking)
        result_text = await asyncio.to_thread(analyzer._call_llm, system_prompt, user_prompt, 3000)

        if not result_text:
            return JSONResponse(
                {"error": "LLM analysis failed. Please try again.", "trade_count": len(trades)}
            )

        # Parse LLM response
        try:
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                import json

                analysis = json.loads(result_text[json_start:json_end])
            else:
                analysis = {
                    "patterns": [],
                    "strengths": [],
                    "weaknesses": [],
                    "recommendations": [],
                    "full_analysis": result_text,
                }
        except Exception:
            analysis = {
                "patterns": [],
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "full_analysis": result_text,
            }

        analysis["strategy_breakdown"] = strategy_breakdown

        return JSONResponse(
            {
                "trade_count": len(trades),
                "stats": {
                    "win_rate": win_rate,
                    "total_r": total_r,
                    "avg_r": avg_r,
                    "wins": wins,
                    "losses": losses,
                },
                "analysis": analysis,
            }
        )

    except Exception as e:
        logger.error(f"Bulk analysis error: {e}")
        return JSONResponse({"error": str(e), "trade_count": 0})
    finally:
        session.close()
