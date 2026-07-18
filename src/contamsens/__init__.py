"""contamsens: contamination sensitivity analysis for benchmark claims.

How contaminated would the benchmark have to be for this claim to be wrong?
See THEORY.md for the formal model (CSM), bounds, and Gamma*.
"""

from .bounds import (
    identified_interval,
    identified_interval_twosided,
    max_bias,
    max_bias_grouped,
    max_bias_selection,
    max_bias_simple,
    max_bias_stratified,
    max_inflation,
)
from .csm import CSM, deflation, is_binary_like, item_floor
from .gamma_star import ROBUST, frontier, gamma_star, is_robust, margin
from .inference import bh_fdr, fragility_pvalue, gamma_star_ci, joint_interval
from .leaderboard import audit

__version__ = "0.2.0"

__all__ = [
    "CSM",
    "ROBUST",
    "audit",
    "bh_fdr",
    "deflation",
    "fragility_pvalue",
    "frontier",
    "is_binary_like",
    "gamma_star",
    "gamma_star_ci",
    "identified_interval",
    "identified_interval_twosided",
    "is_robust",
    "item_floor",
    "joint_interval",
    "margin",
    "max_bias",
    "max_bias_grouped",
    "max_bias_selection",
    "max_bias_simple",
    "max_bias_stratified",
    "max_inflation",
]
