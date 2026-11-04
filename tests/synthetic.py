"""Datos sintéticos (cúmulo con binarias e isócrona falsa con la interfaz de MIST) para las pruebas."""
import numpy as np
import pandas as pd


def make_isochrone_block(n_ms: int = 200, n_post: int = 60) -> np.ndarray:
    """Isócrona sintética en magnitudes absolutas: MS (fase 0) con 'hook' en el turn-off + SGB/RGB (fase 2)."""
    t = np.linspace(0, 1, n_ms)
    g = 9.0 - 5.0 * t                                 # de débil a brillante
    color = 2.0 - 1.2 * t                             # más azul al subir
    hook = t > 0.92                                   # tras el turn-off la MS se vuelve a enrojecer
    color = np.where(hook, color[~hook][-1] + 0.6 * (t - 0.92), color)
    s = np.linspace(0, 1, n_post)
    g_post = np.concatenate([g[-1:] - 0.0 - 3.0 * s])
    c_post = color[-1] + 0.9 * s
    dtype = [("Gaia_G_EDR3", float), ("Gaia_BP_EDR3", float), ("Gaia_RP_EDR3", float), ("phase", float)]
    arr = np.zeros(n_ms + n_post, dtype=dtype)
    colors = np.concatenate([color, c_post])
    gs = np.concatenate([g, g_post])
    arr["Gaia_G_EDR3"] = gs
    arr["Gaia_RP_EDR3"] = gs - 0.5 * colors
    arr["Gaia_BP_EDR3"] = arr["Gaia_RP_EDR3"] + colors
    arr["phase"] = np.concatenate([np.zeros(n_ms), np.full(n_post, 2.0)])
    return arr


class FakeISOCMD:
    """Imita read_mist_models.ISOCMD."""

    def __init__(self, feh: float, shift: float = 0.0):
        block = make_isochrone_block()
        block["Gaia_G_EDR3"] += shift
        block["Gaia_BP_EDR3"] += shift
        block["Gaia_RP_EDR3"] += shift
        self.abun = {"[Fe/H]": feh}
        self.ages = [10.0, 10.3]
        self.isocmds = [block, block]

    def age_index(self, age: float) -> int:
        return int(np.argmin(np.abs(np.array(self.ages) - age)))


def make_cluster(n: int = 6000, binary_fraction: float = 0.3, scatter: float = 0.01, seed: int = 0,
                 block: np.ndarray | None = None):
    """Estrellas sobre la MS (apparent = absolute) con una fracción de binarias de ``q`` uniforme en [0, 1].

    Devuelve (DataFrame con g, bp, rp, color; arreglo q verdadero con 0 para las individuales).
    """
    rng = np.random.default_rng(seed)
    block = make_isochrone_block() if block is None else block
    ms = block[block["phase"] == 0]
    ms_color = ms["Gaia_BP_EDR3"] - ms["Gaia_RP_EDR3"]
    keep = np.arange(len(ms))[: int(0.9 * len(ms))]       # solo el tramo monótono (sin el hook)
    idx = rng.choice(keep[10:-15], size=n)                 # interior de la MS (lejos de los bordes de la región)
    g, color = ms["Gaia_G_EDR3"][idx], ms_color[idx]
    is_bin = rng.random(n) < binary_fraction
    q = np.where(is_bin, rng.random(n), 0.0)
    g = g - 2.5 * np.log10(1 + q**3.5) * is_bin + rng.normal(0, scatter, n)
    color = color + rng.normal(0, scatter, n)
    rp = g - 0.5 * color
    return pd.DataFrame({"g": g, "rp": rp, "bp": rp + color, "color": color}), q
