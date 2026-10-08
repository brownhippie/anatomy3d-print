"""MAD-style safety margins: require a buffer past a hard limit, not a
bare pass/fail at the edge. Convention carried over from unrelated
research of mine on a different project (a physical safety margin for a
laser display), adapted here for a different kind of limit.

There, higher was dangerous: margin = (T_limit - T_actual) / T_limit,
requiring at least 0.15 (15% of headroom before the limit) to pass.

Here, lower is dangerous — a wall thinner than a printer can reliably
produce — so the margin is the mirror image: how far *above* the minimum
the actual value sits, as a fraction of that minimum.
"""
from dataclasses import dataclass


@dataclass
class MadResult:
    ok: bool
    margin: float  # fraction of headroom past the limit; negative means over the limit
    actual: float
    limit: float


def mad_margin_above_minimum(actual: float, minimum: float, min_margin: float = 0.15) -> MadResult:
    """`actual` must stay at or above `minimum`, with at least `min_margin`
    fraction of headroom. Use for a floor (e.g. wall thickness must clear
    the printer's minimum, not just barely touch it)."""
    if minimum <= 0:
        raise ValueError("minimum must be positive")
    margin = (actual - minimum) / minimum
    return MadResult(ok=margin >= min_margin, margin=margin, actual=actual, limit=minimum)


def mad_margin_below_maximum(actual: float, maximum: float, min_margin: float = 0.15) -> MadResult:
    """`actual` must stay at or below `maximum`, with at least `min_margin`
    fraction of headroom. This is the original convention from the source
    research: a ceiling, not a floor."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    margin = (maximum - actual) / maximum
    return MadResult(ok=margin >= min_margin, margin=margin, actual=actual, limit=maximum)
