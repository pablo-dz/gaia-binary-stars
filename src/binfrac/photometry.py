"""Relaciones fotométricas: magnitud de un sistema binario no resuelto y magnitudes aparentes."""
from __future__ import annotations

import numpy as np

ALPHA = 3.5  # exponente de la relación masa-luminosidad L ∝ M^α (Dotter et al. 2007)


def binary_magnitude(m1, q: float, alpha: float = ALPHA):
    """Magnitud de un sistema binario no resuelto con razón de masas ``q = M2/M1``.

    m_bin = m1 − 2.5·log10(1 + q^α).  Es *más brillante* (menor magnitud) que la primaria: para q = 1
    el sistema es ≈ 0.75 mag más brillante.
    """
    return np.asarray(m1, dtype=float) - 2.5 * np.log10(1.0 + q**alpha)


def legacy_binary_magnitude(m1, q: float, alpha: float = ALPHA):
    """Versión del notebook original (``+`` en vez de ``−``): desplaza la secuencia hacia magnitudes *mayores*.

    Se conserva solo para reproducir los números del informe (``method: legacy``); ver README.
    """
    return np.asarray(m1, dtype=float) + 2.5 * np.log10(1.0 + q**alpha)


def apparent_magnitude(abs_mag, distance_pc: float, zero_point: float = 0.0):
    """Magnitud aparente a partir de la absoluta: m = M + 5·log10(d/10 pc) − ZP."""
    return np.asarray(abs_mag, dtype=float) + 5.0 * np.log10(distance_pc / 10.0) - zero_point
