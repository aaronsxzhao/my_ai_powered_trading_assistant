"""Feature extraction for Brooks Trading Coach."""

from app.features.ohlc_features import OHLCFeatures
from app.features.brooks_patterns import BrooksPatternDetector
from app.features.magnets import MagnetDetector

__all__ = ["OHLCFeatures", "BrooksPatternDetector", "MagnetDetector"]
