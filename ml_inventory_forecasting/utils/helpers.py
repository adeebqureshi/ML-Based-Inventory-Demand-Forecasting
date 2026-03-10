from ..config.settings import (
    BLUE, TEAL, DANGER, WARN, SUCCESS,
    TEXT, WHITE, BG, BORDER, TEXT_MID, TEXT_LIGHT,
    PURPLE, BLUE_LIGHT, BLUE_DARK,
)

# Legacy aliases so every file that does `from .helpers import TEAL, MINT` keeps working
TEAL  = BLUE    # primary chart colour → blue
MINT  = TEAL    # secondary chart colour → teal
BROWN = TEXT

__all__ = [
    "TEAL", "MINT", "BROWN", "DANGER", "WARN", "SUCCESS",
    "BLUE", "TEXT", "WHITE", "BG", "BORDER",
    "TEXT_MID", "TEXT_LIGHT", "PURPLE", "BLUE_LIGHT", "BLUE_DARK",
]