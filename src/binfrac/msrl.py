"""Análisis preliminar: línea de cresta de la MS (MSRL) por *data binning*, convolución gaussiana y ajuste polinomial."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from .photometry import binary_magnitude, legacy_binary_magnitude


def polynomial_model(x, a, b, c, d, e):
    """m(x) = a·x⁴ + b·x³ + c·x² + d·x + e   (ecuación 4 del informe)."""
    return e + d * x + c * x**2 + b * x**3 + a * x**4


def bin_msrl(color, g, mag_limit: float, bins: np.ndarray, kernel_std: float):
    """Mediana de G en intervalos de color y suavizado con un kernel gaussiano.

    Devuelve (x = borde derecho de cada intervalo, medianas, curva suavizada).
    """
    from astropy.convolution import Gaussian1DKernel, convolve

    color, g = np.asarray(color), np.asarray(g)
    keep = g < mag_limit
    means, edges, _ = stats.binned_statistic(color[keep], g[keep], statistic="median", bins=bins)
    smooth = convolve(means, Gaussian1DKernel(stddev=kernel_std))
    return edges[1:], means, smooth


def fit_polynomial(x, y, p0=(1, 0.8, 4, -1, 2)):
    """Ajuste de ``polynomial_model``. Devuelve (parámetros, errores 1σ)."""
    popt, pcov = curve_fit(polynomial_model, x, y, p0=list(p0))
    return popt, np.sqrt(np.diag(pcov))


def preliminary_fractions(stars: pd.DataFrame, popt, color_range: tuple[float, float], mag_limit: float,
                          q_values: list[float], method: str = "corrected", n_grid: int = 100_000) -> dict[float, float]:
    """Fracción de estrellas más brillantes que la secuencia de binarias para cada ``q``.

    Con ``method="corrected"`` la secuencia es MSRL − 2.5·log10(1+q^3.5); con ``"legacy"`` replica el
    notebook original (signo ``+``) y reproduce la tabla 2 del informe.
    """
    shift = binary_magnitude if method == "corrected" else legacy_binary_magnitude
    lo, hi = color_range
    sub = stars[(stars["g"] < mag_limit) & (stars["color"] < hi) & (stars["color"] > lo)]
    x_model = np.linspace(lo, hi, n_grid)
    y_model = polynomial_model(x_model, *popt)
    out = {}
    for q in q_values:
        curve = np.interp(sub["color"].to_numpy(), x_model, shift(y_model, q))
        is_bin = sub["g"].to_numpy() < curve
        out[q] = float(is_bin.sum() / len(sub))
    return out
