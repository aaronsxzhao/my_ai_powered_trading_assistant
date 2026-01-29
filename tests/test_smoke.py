"""
Smoke tests for AI Trading Coach.

Quick validation tests that verify:
- All modules import correctly
- FastAPI app starts without errors
- Key routes are registered
- Async wrappers work correctly

To run without dependencies (syntax check only):
    python tests/test_smoke.py --syntax-only

To run with pytest (requires dependencies):
    python -m pytest tests/test_smoke.py -v
"""

import ast
import sys
from pathlib import Path

# Check if running standalone syntax check
if len(sys.argv) > 1 and sys.argv[1] == "--syntax-only":
    print("=== Smoke Test: Syntax Validation ===\n")
    errors = []
    checked = 0

    # Check all Python files for syntax errors
    for py_file in Path("app").rglob("*.py"):
        try:
            source = py_file.read_text()
            ast.parse(source)
            checked += 1
        except SyntaxError as e:
            print(f"✗ {py_file}: line {e.lineno}: {e.msg}")
            errors.append(str(py_file))

    print(f"✓ Checked {checked} Python files")

    # Check rebranding in templates
    template_dir = Path("app/web/templates")
    templates_ok = True

    for html_file in template_dir.rglob("*.html"):
        content = html_file.read_text()
        if "Brooks Trading Coach" in content:
            print(f'✗ {html_file.name}: still has "Brooks Trading Coach"')
            templates_ok = False
            errors.append(str(html_file))

    if templates_ok:
        print("✓ All templates rebranded correctly")

    # Check Python files for branding
    for check_file in ["app/web/server.py", "app/web/__init__.py", "app/web/routes/__init__.py"]:
        content = Path(check_file).read_text()
        if "Brooks Trading Coach" in content:
            print(f'✗ {check_file}: still has "Brooks Trading Coach"')
            errors.append(check_file)
        else:
            print(f"✓ {check_file} rebranded")

    print()
    if errors:
        print(f"=== FAILED: {len(errors)} error(s) ===")
        sys.exit(1)
    else:
        print("=== ALL SMOKE TESTS PASSED ===")
        sys.exit(0)

import pytest
import asyncio
import os

# Set test environment before imports
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["TESTING"] = "true"


class TestImports:
    """Test that all main modules import without errors."""

    def test_config_imports(self):
        """Config modules should import cleanly."""
        from app import config
        from app import config_prompts
        assert hasattr(config, 'MATERIALS_DIR')

    def test_web_imports(self):
        """Web modules should import cleanly."""
        from app.web import server
        from app.web import routes
        assert hasattr(server, 'app')

    def test_auth_imports(self):
        """Auth modules should import cleanly."""
        from app.auth import supabase_auth
        assert hasattr(supabase_auth, 'sign_in')
        assert hasattr(supabase_auth, 'sign_up')

    def test_journal_imports(self):
        """Journal modules should import cleanly."""
        from app.journal import models
        from app.journal import analytics
        assert hasattr(models, 'Trade')
        assert hasattr(analytics, 'TradeAnalytics')

    def test_routes_imports(self):
        """Route modules should import cleanly."""
        from app.web.routes import trades
        from app.web.routes import materials
        from app.web.routes import reports
        assert hasattr(trades, 'router')
        assert hasattr(materials, 'router')


class TestFastAPIApp:
    """Test FastAPI app configuration."""

    def test_app_creates(self):
        """App should create without errors."""
        from app.web.server import app
        assert app is not None
        assert app.title == "AI Trading Coach"

    def test_app_description_updated(self):
        """App description should not mention Brooks."""
        from app.web.server import app
        assert "Brooks" not in app.description
        assert "price action" in app.description.lower()

    def test_routes_registered(self):
        """Key routes should be registered."""
        from app.web.server import app
        
        route_paths = [route.path for route in app.routes]
        
        # Check key routes exist
        assert "/" in route_paths
        assert "/health" in route_paths


class TestAsyncWrappers:
    """Test that async wrappers are properly defined."""

    def test_auth_functions_are_async(self):
        """Auth functions should be async."""
        from app.auth import supabase_auth
        import inspect
        
        assert inspect.iscoroutinefunction(supabase_auth.sign_in)
        assert inspect.iscoroutinefunction(supabase_auth.sign_up)
        assert inspect.iscoroutinefunction(supabase_auth.sign_out)
        assert inspect.iscoroutinefunction(supabase_auth.get_user_from_token)

    def test_trade_routes_are_async(self):
        """Trade route handlers should be async."""
        from app.web.routes import trades
        import inspect
        
        assert inspect.iscoroutinefunction(trades.recalculate_trade_metrics)
        assert inspect.iscoroutinefunction(trades.update_trade_notes)
        assert inspect.iscoroutinefunction(trades.cancel_trade_review)

    def test_material_routes_are_async(self):
        """Material route handlers should be async."""
        from app.web.routes import materials
        import inspect
        
        assert inspect.iscoroutinefunction(materials.list_materials)
        assert inspect.iscoroutinefunction(materials.delete_material)


class TestRebranding:
    """Test that rebranding was applied correctly."""

    def test_server_title(self):
        """Server should use AI Trading Coach title."""
        from app.web.server import app
        assert app.title == "AI Trading Coach"

    def test_web_module_docstring(self):
        """Web module docstring should be updated."""
        from app import web
        assert "AI Trading Coach" in web.__doc__

    def test_routes_module_docstring(self):
        """Routes module docstring should be updated."""
        from app.web import routes
        assert "AI Trading Coach" in routes.__doc__


class TestTemplateFiles:
    """Test that template files exist and are valid."""

    def test_base_template_exists(self):
        """Base template should exist."""
        from app.config import MATERIALS_DIR
        from pathlib import Path
        
        template_dir = Path(__file__).parent.parent / "app" / "web" / "templates"
        assert (template_dir / "base.html").exists()

    def test_base_template_rebranded(self):
        """Base template should use AI Trading Coach branding."""
        from pathlib import Path
        
        template_dir = Path(__file__).parent.parent / "app" / "web" / "templates"
        content = (template_dir / "base.html").read_text()
        
        assert "AI Trading Coach" in content
        assert "Brooks Trading Coach" not in content


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_smoke.py -v
    pytest.main([__file__, "-v"])
