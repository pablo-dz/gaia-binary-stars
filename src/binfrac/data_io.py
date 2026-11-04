"""Lectura de las consultas a Gaia DR3 y selección de miembros por movimiento propio."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RENAME = {"phot_g_mean_mag": "g", "phot_bp_mean_mag": "bp", "phot_rp_mean_mag": "rp"}


def load_gaia_csv(path: str | Path, required: list[str]) -> pd.DataFrame:
    """Lee un CSV de Gaia DR3, elimina filas con NaN en ``required`` y añade ``g``, ``bp``, ``rp`` y ``color``."""
    df = pd.read_csv(path).dropna(subset=required).copy()
    df = df.rename(columns=RENAME)
    df["color"] = df["bp"] - df["rp"]
    return df


def pmra_members(df: pd.DataFrame, pmra_range: tuple[float, float]) -> pd.DataFrame:
    """Conserva las estrellas con movimiento propio en ascensión recta dentro de ``pmra_range`` (inclusive).

    Se elige a mano, en el histograma de ``pmra``, la región con más fuentes (la del cúmulo).
    """
    lo, hi = pmra_range
    return df[(df["pmra"] >= lo) & (df["pmra"] <= hi)]
