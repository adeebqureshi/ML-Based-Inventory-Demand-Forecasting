"""
Helper module providing legacy color aliases.

These aliases exist for backward compatibility with older modules
that referenced these color names directly. All colors are sourced
from config/settings.py so that the palette remains centralized.
"""

from ..config.settings import BLUE, TEXT

TEAL = BLUE
MINT = BLUE
BROWN = TEXT

__all__ = ["TEAL", "MINT", "BROWN"]