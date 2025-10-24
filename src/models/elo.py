from __future__ import annotations
import math

def win_prob_from_elo(elo_a: float, elo_b: float, home_bonus: float = 65.0, home: str | None = "A") -> float:
    aa, bb = float(elo_a), float(elo_b)
    if home == "A":
        aa += home_bonus
    elif home == "B":
        bb += home_bonus
    diff = (aa - bb) / 400.0
    return 1.0 / (1.0 + 10 ** (-diff))
