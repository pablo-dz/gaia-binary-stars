"""Identificación de binarias MS-MS sobre el CMD y cálculo de la fracción binaria."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .photometry import ALPHA, binary_magnitude, legacy_binary_magnitude


def select_ms_region(color, g, ridge: pd.DataFrame, g_margin: float = 0.1,
                     color_window: float | None = None) -> np.ndarray:
    """Máscara de las estrellas que se consideran en la región de la secuencia principal.

    - Siempre se acota en magnitud al rango de la MS de la isócrona (``g_margin`` por el extremo débil).
    - Si ``color_window`` no es ``None``, además se exige estar a menos de ``color_window`` mag (en color)
      de la línea de la MS. Excluye gigantes, subgigantes y rama horizontal que caen en el mismo rango de G.
    """
    color, g = np.asarray(color), np.asarray(g)
    mask = (g > ridge["g"].min()) & (g < ridge["g"].max() + g_margin)
    if color_window is not None:
        by_g = ridge.sort_values("g")
        ridge_color = np.interp(g, by_g["g"].to_numpy(), by_g["color"].to_numpy())
        mask &= np.abs(color - ridge_color) <= color_window
    return mask


def classify_binaries(color, g, ridge: pd.DataFrame, q: float, alpha: float = ALPHA) -> np.ndarray:
    """True para las estrellas más brillantes que la secuencia de binarias con razón de masas ``q``.

    La secuencia de binarias es la MS desplazada ``2.5·log10(1+q^α)`` mag hacia el brillo, a igual color.
    Una estrella más brillante que esa línea (menor G) tiene ``q' > q`` y se considera binaria.
    """
    line = np.interp(np.asarray(color), ridge["color"].to_numpy(), binary_magnitude(ridge["g"].to_numpy(), q, alpha))
    return np.asarray(g) < line


def binary_fraction(stars: pd.DataFrame, ridge: pd.DataFrame, q: float, g_margin: float = 0.1,
                    color_window: float | None = None) -> dict:
    """Fracción binaria ``N_bin / (N_bin + N_s)`` en la región de la MS (ecuación 5 del informe)."""
    in_ms = select_ms_region(stars["color"], stars["g"], ridge, g_margin, color_window)
    sub = stars[in_ms]
    is_bin = classify_binaries(sub["color"], sub["g"], ridge, q)
    n_bin, n_single = int(is_bin.sum()), int((~is_bin).sum())
    total = n_bin + n_single
    return {"n_bin": n_bin, "n_single": n_single, "fraction": n_bin / total if total else float("nan"),
            "stars": sub, "is_binary": is_bin}


# ---------------------------------------------------------------------------
# Procedimiento del notebook original ("legacy"), solo para reproducir las cifras del informe
# ---------------------------------------------------------------------------


def legacy_binary_fraction(stars: pd.DataFrame, ms: pd.DataFrame, q: float, g_margin: float = 0.1) -> dict:
    """Replica el cálculo del notebook: signo ``+``, desplazamiento del *color*, ``np.interp`` sin ordenar
    y binaria = estrella **más débil** que la curva (``g > curva``)."""
    mask_mag = (stars["g"] > ms["g"].min()) & (stars["g"] < ms["g"].max() + g_margin)
    sub = stars[mask_mag]
    x_bin = legacy_binary_magnitude(ms["color"].to_numpy(), q)
    y_bin = legacy_binary_magnitude(ms["g"].to_numpy(), q)
    curve = np.interp(sub["color"].to_numpy(), x_bin, y_bin)
    is_bin = sub["g"].to_numpy() > curve
    n_bin, n_single = int(is_bin.sum()), int((~is_bin).sum())
    return {"n_bin": n_bin, "n_single": n_single, "fraction": n_bin / (n_bin + n_single),
            "stars": sub, "is_binary": is_bin}
