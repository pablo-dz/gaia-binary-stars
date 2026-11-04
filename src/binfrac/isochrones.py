"""Isócronas MIST: lectura, conversión a magnitudes aparentes y extracción de la secuencia principal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .photometry import apparent_magnitude

PHASE_MS = 0       # secuencia principal (notación FSPS)
PHASE_SGB_RGB = 2  # subgigantes y rama de gigantes rojas


def load_mist_isochrones(directory: str | Path, pattern: str = "*.iso.cmd") -> dict[float, object]:
    """Lee todos los archivos ``.iso.cmd`` de ``directory``. Devuelve ``{[Fe/H]: ISOCMD}``."""
    from .third_party import read_mist_models

    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No hay archivos {pattern} en {directory}. Ver data/README.md")
    out = {}
    for f in files:
        iso = read_mist_models.ISOCMD(str(f), verbose=False)
        out[iso.abun["[Fe/H]"]] = iso
    return dict(sorted(out.items()))


def apparent_isochrone(iso, log_age: float, distance_pc: float, zero_points: dict[str, float],
                       bands: dict[str, str]) -> pd.DataFrame:
    """Isócrona de edad más cercana a ``log_age`` en magnitudes aparentes.

    ``bands`` mapea ``{"g": "Gaia_G_EDR3", "bp": ..., "rp": ...}`` y ``zero_points`` usa las mismas claves.
    Devuelve columnas ``g``, ``bp``, ``rp``, ``color`` y ``phase``.
    """
    block = iso.isocmds[iso.age_index(log_age)]
    out = pd.DataFrame({
        key: apparent_magnitude(block[col], distance_pc, zero_points[key]) for key, col in bands.items()
    })
    out["color"] = out["bp"] - out["rp"]
    out["phase"] = block["phase"]
    return out


def main_sequence(isochrone: pd.DataFrame) -> pd.DataFrame:
    """Puntos de la isócrona en fase de secuencia principal."""
    return isochrone[isochrone["phase"] == PHASE_MS].reset_index(drop=True)


def ms_ridge(ms: pd.DataFrame) -> pd.DataFrame:
    """Tramo monótono de la secuencia principal, ordenado por color creciente.

    Parte del extremo más débil y llega hasta el punto más azul (cerca del *turn-off*), donde la MS
    deja de ser una función del color. Así se puede interpolar ``g(color)`` sin ambigüedades.
    """
    faint_to_bright = ms.sort_values("g", ascending=False).reset_index(drop=True)
    bluest = int(faint_to_bright["color"].to_numpy().argmin())
    ridge = faint_to_bright.iloc[: bluest + 1]
    return ridge.sort_values("color").reset_index(drop=True)
