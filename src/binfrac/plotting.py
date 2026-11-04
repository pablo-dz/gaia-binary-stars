"""Figuras: corrección por movimiento propio, MSRL preliminar, ajuste de isócronas, binarias y fracción vs edad."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: str | Path, dpi: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_membership(all_stars: pd.DataFrame, members: pd.DataFrame, pmra_range, name: str,
                    path: str | Path, dpi: int = 200) -> None:
    """Histograma de ``pmra`` y CMD antes (arriba) y después (abajo) de seleccionar la región del cúmulo."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex="col")
    (ax1, ax2), (ax3, ax4) = axes
    ax1.hist(all_stars["pmra"], bins=300, histtype="step", color="black")
    for v in pmra_range:
        ax1.axvline(v, linestyle="dashed", label="Región escogida" if v == pmra_range[0] else None)
    ax1.legend(loc="upper left")
    ax2.scatter(all_stars["color"], all_stars["g"], s=0.01, color="black")
    ax3.hist(members["pmra"], bins=300, histtype="step", color="black")
    ax4.scatter(members["color"], members["g"], s=0.01, color="black")
    for ax in (ax2, ax4):
        ax.invert_yaxis()
        ax.set_ylabel(r"$G$")
    ax4.set_xlabel(r"$BP-RP$")
    ax3.set_xlabel("pmra [mas/yr]")
    for ax in (ax1, ax3):
        ax.set_ylabel("Número")
    fig.suptitle(name)
    _save(fig, path, dpi)


def plot_msrl(stars: pd.DataFrame, x, means, smooth, x_model, y_model, y_binary, q: float, name: str,
              path: str | Path, dpi: int = 200) -> None:
    """MSRL por data binning + convolución, ajuste polinomial y secuencia de binarias para ``q``."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(stars["color"], stars["g"], s=0.1, color="gray", alpha=0.6)
    ax.plot(x, smooth, color="black", label="Data-Binning + Convolución")
    ax.plot(x_model, y_model, "r--", label="Modelo polinomial")
    ax.plot(x_model, y_binary, "b:", label=f"MS binaria con $q={q}$")
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G")
    ax.set_title(name)
    ax.invert_yaxis()
    ax.legend()
    _save(fig, path, dpi)


def plot_isochrone_grid(stars: pd.DataFrame, isochrones: dict[float, pd.DataFrame], best_feh: float, name: str,
                        path: str | Path, xlim=(0.3, 1.6), ylim=(12, 19), dpi: int = 200) -> None:
    """Izquierda: isócronas para toda la grilla de [Fe/H]. Derecha: la isócrona elegida."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(isochrones)))
    for (feh, iso), c in zip(isochrones.items(), colors):
        ax1.plot(iso["color"], iso["g"], color=c, label=f"[Fe/H] = {feh}",
                 linestyle="dashed" if feh == best_feh else "solid")
    best = isochrones[best_feh]
    for ax in (ax1, ax2):
        ax.scatter(stars["color"], stars["g"], s=0.1, color="black")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$BP-RP$")
    ax2.plot(best["color"], best["g"], "r--", label=f"Isócrona ajustada ([Fe/H] = {best_feh})")
    ax1.set_ylim(*ylim[::-1])
    ax1.set_ylabel(r"$G$")
    ax1.legend(fontsize=6, ncol=2)
    ax2.legend()
    fig.suptitle(name)
    _save(fig, path, dpi)


def plot_binary_selection(result: dict, ridge_color, line_g, ms: pd.DataFrame, q: float, name: str,
                          path: str | Path, dpi: int = 200) -> None:
    """CMD de la región de la MS con binarias (verde) y estrellas individuales (negro) y la línea de corte."""
    stars, is_bin = result["stars"], result["is_binary"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(stars["color"][~is_bin], stars["g"][~is_bin], s=0.2, color="black", label="Estrella individual")
    ax.scatter(stars["color"][is_bin], stars["g"][is_bin], s=0.2, color="green", label="Binaria")
    ax.plot(ms["color"], ms["g"], color="orange", linestyle="dashed", label="MS (isócrona)")
    ax.plot(ridge_color, line_g, color="red", label=f"MS binaria $q={q}$")
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G")
    ax.set_ylim(stars["g"].max() + 0.1, stars["g"].min() - 0.1)
    ax.set_title(f"{name}: $f_{{bin}}$ = {100 * result['fraction']:.1f} %")
    ax.legend(markerscale=15)
    _save(fig, path, dpi)


def plot_fraction_vs_age(df: pd.DataFrame, path: str | Path, dpi: int = 200) -> None:
    """Fracción binaria de cada cúmulo en función de su edad (de la literatura)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["age_yr"], df["fraction"], color="black", s=30)
    for _, r in df.iterrows():
        ax.annotate(r["cluster"], (r["age_yr"], r["fraction"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Edad (yr)")
    ax.set_ylabel("Fracción binaria")
    _save(fig, path, dpi)
