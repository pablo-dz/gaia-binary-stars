"""Flujos de trabajo: miembros, MSRL preliminar, ajuste de isócronas, fracción binaria y fracción vs edad."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotting
from .binaries import binary_fraction, legacy_binary_fraction
from .data_io import load_gaia_csv, pmra_members
from .isochrones import apparent_isochrone, load_mist_isochrones, main_sequence, ms_ridge
from .photometry import binary_magnitude, legacy_binary_magnitude
from .msrl import bin_msrl, fit_polynomial, polynomial_model, preliminary_fractions

PHOTOMETRIC_COLUMNS = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]


def _fig(cfg: dict, name: str) -> Path:
    return cfg["paths"]["figures_dir"] / f"{name}.{cfg['figures']['format']}"


def _dpi(cfg: dict) -> int:
    return cfg["figures"]["dpi"]


def _results_dir(cfg: dict) -> Path:
    cfg["paths"]["results_dir"].mkdir(parents=True, exist_ok=True)
    return cfg["paths"]["results_dir"]


def _raw(cfg: dict, filename: str) -> Path:
    path = cfg["paths"]["raw_dir"] / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ver data/README.md")
    return path


def load_members(cfg: dict, name: str) -> pd.DataFrame:
    """Estrellas del cúmulo ``name`` tras la selección por movimiento propio."""
    c = cfg["clusters"][name]
    df = load_gaia_csv(_raw(cfg, c["file"]), cfg["binaries"]["required_columns"])
    return pmra_members(df, tuple(c["pmra"]))


# ---------------------------------------------------------------------------
# 1) Selección de miembros
# ---------------------------------------------------------------------------


def run_membership(cfg: dict, clusters: list[str] | None = None) -> dict[str, int]:
    """Figuras de la corrección por ``pmra`` para cada cúmulo y número de estrellas conservadas."""
    counts = {}
    for name in clusters or cfg["clusters"]:
        c = cfg["clusters"][name]
        full = load_gaia_csv(_raw(cfg, c["file"]), cfg["binaries"]["required_columns"])
        members = pmra_members(full, tuple(c["pmra"]))
        plotting.plot_membership(full, members, c["pmra"], name, _fig(cfg, f"membership/{name}"), _dpi(cfg))
        counts[name] = len(members)
        print(f"{name}: {len(full):,} -> {len(members):,} estrellas")
    return counts


# ---------------------------------------------------------------------------
# 2) MSRL preliminar (NGC 4755, NGC 6694, NGC 2808)
# ---------------------------------------------------------------------------


def run_preliminary(cfg: dict) -> pd.DataFrame:
    """MSRL por data binning + convolución + polinomio y fracción de binarias para varios ``q``."""
    pcfg = cfg["preliminary"]
    method = pcfg["method"]
    shift = binary_magnitude if method == "corrected" else legacy_binary_magnitude
    rows, params = [], {}
    for name, c in pcfg["clusters"].items():
        stars = load_gaia_csv(_raw(cfg, c["file"]), PHOTOMETRIC_COLUMNS)
        lo, hi, step = c["bins"]
        x, means, smooth = bin_msrl(stars["color"], stars["g"], c["mag_limit"], np.arange(lo, hi, step), c["kernel_std"])
        popt, perr = fit_polynomial(x, smooth)
        params[name] = {"coefficients[a,b,c,d,e]": popt.tolist(), "errors": perr.tolist()}

        fractions = preliminary_fractions(stars, popt, tuple(c["color_range"]), c["mag_limit"], pcfg["q_values"], method)
        rows.append({"cluster": name, **{f"f_bin(q>{q})": 100 * f for q, f in fractions.items()}})

        shown = stars[(stars["g"] < c["mag_limit"]) & stars["color"].between(*c["color_range"])]
        x_model = np.linspace(*c["color_range"], 2000)
        y_model = polynomial_model(x_model, *popt)
        q0 = pcfg["q_values"][0]
        plotting.plot_msrl(shown, x, means, smooth, x_model, y_model, shift(y_model, q0), q0, name,
                           _fig(cfg, f"preliminary/{name}_msrl"), _dpi(cfg))
    table = pd.DataFrame(rows)
    out = _results_dir(cfg)
    table.to_csv(out / "preliminary_fractions.csv", index=False)
    with open(out / "preliminary_polynomial_fits.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(table.round(2).to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# 3) Ajuste por isócronas
# ---------------------------------------------------------------------------


def _fitted_clusters(cfg: dict) -> list[str]:
    return [n for n, c in cfg["clusters"].items() if c.get("binary_fraction")]


def _isochrone_for(cfg: dict, isochrones: dict, name: str, feh: float | None = None) -> pd.DataFrame:
    c, icfg = cfg["clusters"][name], cfg["isochrones"]
    return apparent_isochrone(isochrones[c["best_feh"] if feh is None else feh], icfg["log_age"],
                              c["distance_pc"], c["zero_points"], icfg["bands"])


def run_isochrone_fits(cfg: dict, clusters: list[str] | None = None) -> None:
    """Figura de comparación de isócronas MIST de distinta metalicidad con el CMD de cada cúmulo."""
    isochrones = load_mist_isochrones(cfg["isochrones"]["directory"], cfg["isochrones"]["pattern"])
    for name in clusters or _fitted_clusters(cfg):
        c = cfg["clusters"][name]
        stars = load_members(cfg, name)
        grid = {feh: _isochrone_for(cfg, isochrones, name, feh) for feh in isochrones}
        plotting.plot_isochrone_grid(stars, grid, c["best_feh"], name, _fig(cfg, f"isochrones/{name}_adjust"),
                                     dpi=_dpi(cfg))
        print(f"{name}: [Fe/H] elegido = {c['best_feh']}")


# ---------------------------------------------------------------------------
# 4) Fracción binaria
# ---------------------------------------------------------------------------


def run_binary_fraction(cfg: dict, clusters: list[str] | None = None) -> pd.DataFrame:
    """Fracción de binarias MS-MS (razón de masas > q) para cada cúmulo globular ajustado."""
    bcfg = cfg["binaries"]
    q, method = bcfg["q"], bcfg["method"]
    isochrones = load_mist_isochrones(cfg["isochrones"]["directory"], cfg["isochrones"]["pattern"])
    rows = []
    for name in clusters or _fitted_clusters(cfg):
        stars = load_members(cfg, name)
        ms = main_sequence(_isochrone_for(cfg, isochrones, name))
        if method == "corrected":
            ridge = ms_ridge(ms)
            res = binary_fraction(stars, ridge, q, bcfg["g_margin"], bcfg["ms_color_window"])
            line_x, line_y = ridge["color"], binary_magnitude(ridge["g"], q)
        else:
            res = legacy_binary_fraction(stars, ms, q, bcfg["g_margin"])
            line_x, line_y = legacy_binary_magnitude(ms["color"], q), legacy_binary_magnitude(ms["g"], q)
        plotting.plot_binary_selection(res, line_x, line_y, ms, q, name, _fig(cfg, f"binary_fraction/{name}"), _dpi(cfg))
        rows.append({"cluster": name, "q": q, "method": method, "n_bin": res["n_bin"], "n_single": res["n_single"],
                     "fraction": res["fraction"], "age_yr": cfg["clusters"][name].get("age_yr")})
        print(f"{name}: f_bin = {100 * res['fraction']:.1f} %  ({res['n_bin']} / {res['n_bin'] + res['n_single']})")
    table = pd.DataFrame(rows)
    table.to_csv(_results_dir(cfg) / "binary_fractions.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# 5) Fracción binaria vs edad
# ---------------------------------------------------------------------------


def run_fraction_vs_age(cfg: dict) -> pd.DataFrame:
    """Figura de la fracción binaria frente a la edad (lee ``results/binary_fractions.csv``)."""
    path = _results_dir(cfg) / "binary_fractions.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero scripts/04_binary_fraction.py")
    df = pd.read_csv(path)
    plotting.plot_fraction_vs_age(df, _fig(cfg, "fraction_vs_age"), _dpi(cfg))
    corr = df[["age_yr", "fraction"]].corr().iloc[0, 1]
    print(f"Coeficiente de correlación de Pearson (edad, fracción): {corr:.2f} con N = {len(df)} cúmulos")
    return df
