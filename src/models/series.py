from __future__ import annotations
def series_win_prob(p_game: float, best_of: int = 7) -> float:
    if best_of % 2 == 0 or best_of <= 0:
        raise ValueError("best_of must be a positive odd number")
    need = best_of // 2 + 1
    from math import comb
    return sum(comb(best_of, k) * (p_game ** k) * ((1 - p_game) ** (best_of - k)) for k in range(need, best_of + 1))
