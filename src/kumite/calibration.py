"""確率の較正（温度スケーリング）。

確率 p を softmax(log p / T) に変える。T > 1 で穏やかに、T < 1 で鋭くなる。
T を決める予測と適用する予測は、同じ形（単一モデルか平均か）にそろえる（knowledge/K002）。
"""

import math
from collections.abc import Callable

import numpy as np

EPS = 1e-15
T_BOUNDS = (0.2, 5.0)

Loss = Callable[[np.ndarray, np.ndarray], float]


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(proba, EPS, 1)) / t
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(
    y: np.ndarray, proba: np.ndarray, loss: Loss, bounds: tuple[float, float] = T_BOUNDS
) -> float:
    """loss(y, apply_temperature(proba, T)) を最小にする T。log T 上の黄金分割探索（損失は T に単峰とみなす）。"""
    if len(y) != len(proba):
        raise ValueError(f"y と proba の行数が違います（{len(y)} / {len(proba)}）")

    def f(log_t: float) -> float:
        return loss(y, apply_temperature(proba, math.exp(log_t)))

    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = f(a), f(b)
    while hi - lo > 1e-5:
        if fa <= fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = f(b)
    return math.exp((lo + hi) / 2)
