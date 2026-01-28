"""
Strategy management API routes.

Handles strategy CRUD operations, merging, and categorization.
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.journal.models import Trade, Strategy, get_session
from app.web.dependencies import require_auth, require_write_auth
from app.web.utils import clear_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["strategies"])


DEFAULT_CATEGORIES = [
    {"id": "trend", "name": "Trend"},
    {"id": "trading_range", "name": "Trading Range"},
    {"id": "reversal", "name": "Reversal"},
    {"id": "special", "name": "Special"},
    {"id": "unknown", "name": "Unknown"},
]


def get_categories() -> list:
    """Get categories from settings or return defaults."""
    from app.config_prompts import load_settings

    settings = load_settings()
    return settings.get("categories", DEFAULT_CATEGORIES)


def save_categories(categories: list) -> bool:
    """Save categories to settings."""
    from app.config_prompts import load_settings, save_settings

    settings = load_settings()
    settings["categories"] = categories
    return save_settings(settings)


@router.get("/categories")
async def get_categories_api():
    """Get all categories."""
    return JSONResponse({"categories": get_categories()})


@router.post("/categories", dependencies=[require_write_auth])
async def create_category(request: Request):
    """Create a new category."""
    data = await request.json()
    name = data.get("name", "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="Category name required")

    # Generate ID from name (lowercase, replace spaces with underscores)
    cat_id = name.lower().replace(" ", "_").replace("-", "_")

    categories = get_categories()

    # Check if already exists
    if any(c["id"] == cat_id for c in categories):
        raise HTTPException(status_code=400, detail="Category already exists")

    categories.append({"id": cat_id, "name": name})

    if save_categories(categories):
        return JSONResponse({"message": f"Category '{name}' created", "id": cat_id})
    raise HTTPException(status_code=500, detail="Failed to save category")


@router.delete("/categories/{category_id}", dependencies=[require_write_auth])
async def delete_category(category_id: str):
    """Delete a category. Removes from strategies that use it."""
    categories = get_categories()

    # Find and remove the category
    new_categories = [c for c in categories if c["id"] != category_id]

    if len(new_categories) == len(categories):
        raise HTTPException(status_code=404, detail="Category not found")

    # Remove this category from all strategies that use it
    session = get_session()
    try:
        strategies = session.query(Strategy).all()
        for strategy in strategies:
            if strategy.category:
                cats = [
                    c.strip()
                    for c in strategy.category.split(",")
                    if c.strip() and c.strip() != category_id
                ]
                strategy.category = ",".join(cats) if cats else None
        session.commit()
        clear_cache("active_strategies")
    finally:
        session.close()

    if save_categories(new_categories):
        return JSONResponse({"message": "Category deleted"})
    raise HTTPException(status_code=500, detail="Failed to delete category")


@router.get("/strategies")
async def get_strategies():
    """Get all strategies with trade counts."""
    session = get_session()
    try:
        strategies = session.query(Strategy).order_by(Strategy.category, Strategy.name).all()

        result = []
        for s in strategies:
            trade_count = session.query(Trade).filter(Trade.strategy_id == s.id).count()
            result.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "category": s.category,
                    "description": s.description,
                    "trade_count": trade_count,
                }
            )

        return JSONResponse({"strategies": result})
    finally:
        session.close()


@router.patch("/strategies/{strategy_id}", dependencies=[require_write_auth])
async def update_strategy(strategy_id: int, request: Request):
    """Update a strategy's name, category, or description."""
    data = await request.json()
    session = get_session()
    try:
        strategy = session.query(Strategy).get(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Check for duplicate name before updating
        if "name" in data and data["name"] != strategy.name:
            existing = (
                session.query(Strategy)
                .filter(Strategy.name == data["name"], Strategy.id != strategy_id)
                .first()
            )
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Strategy '{data['name']}' already exists. Use merge instead.",
                )
            strategy.name = data["name"]

        if "category" in data:
            strategy.category = data["category"]
        if "description" in data:
            strategy.description = data["description"]

        session.commit()
        return JSONResponse({"message": f"Strategy '{strategy.name}' updated"})
    finally:
        session.close()


@router.post("/strategies/merge", dependencies=[require_write_auth])
async def merge_strategies(request: Request):
    """Merge multiple strategies into one (reassign all trades)."""
    data = await request.json()
    source_ids = data.get("source_ids", [])
    target_id = data.get("target_id")

    if not source_ids or not target_id:
        raise HTTPException(status_code=400, detail="source_ids and target_id required")

    session = get_session()
    try:
        target = session.query(Strategy).get(target_id)
        if not target:
            raise HTTPException(status_code=404, detail="Target strategy not found")

        merged_count = 0
        for source_id in source_ids:
            if source_id == target_id:
                continue

            # Reassign all trades from source to target
            trades = session.query(Trade).filter(Trade.strategy_id == source_id).all()
            for trade in trades:
                trade.strategy_id = target_id
                merged_count += 1

            # Delete the source strategy
            source = session.query(Strategy).get(source_id)
            if source:
                session.delete(source)

        session.commit()
        clear_cache("active_strategies")  # Invalidate cache

        return JSONResponse(
            {
                "message": f"Merged {merged_count} trades into '{target.name}'",
                "merged_count": merged_count,
            }
        )
    finally:
        session.close()


@router.post("/strategies", dependencies=[require_write_auth])
async def create_strategy(request: Request):
    """Create a new strategy."""
    data = await request.json()
    name = data.get("name")
    category = data.get("category", "unknown")

    if not name:
        raise HTTPException(status_code=400, detail="Strategy name required")

    session = get_session()
    try:
        # Check if already exists
        existing = session.query(Strategy).filter(Strategy.name == name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Strategy already exists")

        strategy = Strategy(
            name=name,
            category=category,
            description=data.get("description", ""),
        )
        session.add(strategy)
        session.commit()
        clear_cache("active_strategies")  # Invalidate cache

        return JSONResponse({"message": f"Strategy '{name}' created", "id": strategy.id})
    finally:
        session.close()


@router.delete("/strategies/{strategy_id}", dependencies=[require_auth])
async def delete_strategy(strategy_id: int):
    """Delete a strategy (only if no trades are assigned)."""
    session = get_session()
    try:
        strategy = session.query(Strategy).get(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Check if any trades use this strategy
        trade_count = session.query(Trade).filter(Trade.strategy_id == strategy_id).count()
        if trade_count > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete: {trade_count} trades use this strategy",
            )

        session.delete(strategy)
        session.commit()
        clear_cache("active_strategies")  # Invalidate cache

        return JSONResponse({"message": "Strategy deleted"})
    finally:
        session.close()
