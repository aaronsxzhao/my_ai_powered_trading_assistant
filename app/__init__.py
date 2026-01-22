"""
Brooks Price Action Trading Coach

Advisory system for discretionary day traders grounded in Al Brooks price action concepts.
This is a decision-support + journaling + analytics + premarket briefing system.

WARNING: This system does NOT auto-trade. It is advisory only.
"""

# Suppress SWIG deprecation warnings from databento's C++ bindings
# Must be set BEFORE any databento imports happen
import warnings
warnings.filterwarnings("ignore", message=".*Swig.*has no __module__ attribute")
warnings.filterwarnings("ignore", message=".*swig.*has no __module__ attribute")
warnings.filterwarnings("ignore", message="builtin type .* has no __module__ attribute")

__version__ = "0.1.0"
__author__ = "Trading Coach Team"
