"""
Brooks Price Action Trading Coach

Advisory system for discretionary day traders grounded in Al Brooks price action concepts.
This is a decision-support + journaling + analytics + premarket briefing system.

WARNING: This system does NOT auto-trade. It is advisory only.
"""

# Suppress SWIG deprecation warnings from databento's C++ bindings
# Must be set BEFORE any databento imports happen
import warnings

# Filter by message pattern
warnings.filterwarnings("ignore", message=".*Swig.*")
warnings.filterwarnings("ignore", message=".*swig.*")
warnings.filterwarnings("ignore", message=".*__module__ attribute.*")
# Filter DeprecationWarning from importlib entirely (where SWIG warnings originate)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib.*")

__version__ = "0.1.0"
__author__ = "Trading Coach Team"
