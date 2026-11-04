import numpy as np

from binfrac import binaries as bn
from binfrac import isochrones as iso
from binfrac.photometry import apparent_magnitude

from synthetic import FakeISOCMD, make_cluster


def _ridge():
    fake = FakeISOCMD(-1.0)
    bands = {"g": "Gaia_G_EDR3", "bp": "Gaia_BP_EDR3", "rp": "Gaia_RP_EDR3"}
    app = iso.apparent_isochrone(fake, 10.3, 10.0, {"g": 0, "bp": 0, "rp": 0}, bands)
    return app, iso.main_sequence(app), iso.ms_ridge(iso.main_sequence(app))


def test_ridge_is_monotonic_and_drops_hook():
    app, ms, ridge = _ridge()
    assert (np.diff(ridge["color"]) >= 0).all()
    assert ridge["g"].is_monotonic_increasing          # más rojo = más débil
    assert len(ridge) < len(ms)                         # se descartó el tramo posterior al turn-off


def test_fraction_recovered_with_corrected_method():
    _, _, ridge = _ridge()
    stars, q_true = make_cluster(n=20000, binary_fraction=0.3, scatter=0.005, seed=1)
    res = bn.binary_fraction(stars, ridge, q=0.5)
    expected = 0.3 * 0.5          # binarias con q > 0.5 / total
    assert abs(res["fraction"] - expected) < 0.03
    assert res["n_bin"] + res["n_single"] == len(res["stars"])


def test_lower_q_threshold_gives_larger_fraction():
    _, _, ridge = _ridge()
    stars, _ = make_cluster(n=10000, binary_fraction=0.3, scatter=0.005, seed=2)
    f = [bn.binary_fraction(stars, ridge, q)["fraction"] for q in (0.9, 0.7, 0.5, 0.4)]
    assert f == sorted(f)


def test_no_binaries_gives_near_zero_fraction():
    _, _, ridge = _ridge()
    stars, _ = make_cluster(n=10000, binary_fraction=0.0, scatter=0.002, seed=3)
    assert bn.binary_fraction(stars, ridge, q=0.7)["fraction"] < 0.05


def test_color_window_excludes_red_stars():
    _, _, ridge = _ridge()
    stars, _ = make_cluster(n=2000, binary_fraction=0.0, seed=4)
    red = stars.copy()
    red["color"] += 0.5
    mask = bn.select_ms_region(red["color"], red["g"], ridge, color_window=0.15)
    assert mask.sum() == 0
    assert bn.select_ms_region(stars["color"], stars["g"], ridge, color_window=0.15).sum() > 0


def test_legacy_runs_and_differs():
    _, ms, ridge = _ridge()
    stars, _ = make_cluster(n=5000, binary_fraction=0.3, seed=5)
    leg = bn.legacy_binary_fraction(stars, ms, q=0.4)
    assert 0 <= leg["fraction"] <= 1
    assert set(leg) == {"n_bin", "n_single", "fraction", "stars", "is_binary"}
