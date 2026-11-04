"""Prueba de humo del flujo completo con un cúmulo sintético e isócronas MIST falsas."""
import numpy as np
import pandas as pd
import pytest

from binfrac import pipeline
from binfrac import isochrones as iso_mod
from binfrac.config import load_config
from binfrac.data_io import load_gaia_csv, pmra_members

from synthetic import FakeISOCMD, make_cluster


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    cfg = load_config()
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg["paths"] = {"raw_dir": raw, "results_dir": tmp_path / "results", "figures_dir": tmp_path / "figures"}
    cfg["figures"] = {"format": "png", "dpi": 40}
    cfg["clusters"] = {
        "M5": {"file": "M5-result.csv", "pmra": [-2, 10], "binary_fraction": True, "distance_pc": 10.0,
               "zero_points": {"g": 0.0, "bp": 0.0, "rp": 0.0}, "best_feh": -1.0, "age_yr": 12.7e9},
        "M3": {"file": "M3-result.csv", "pmra": [-2, 10], "binary_fraction": True, "distance_pc": 10.0,
               "zero_points": {"g": 0.0, "bp": 0.0, "rp": 0.0}, "best_feh": -1.5, "age_yr": 11.4e9},
    }
    for i, name in enumerate(cfg["clusters"]):
        stars, _ = make_cluster(n=4000, binary_fraction=0.25, seed=i)
        stars = stars.rename(columns={"g": "phot_g_mean_mag", "bp": "phot_bp_mean_mag", "rp": "phot_rp_mean_mag"})
        rng = np.random.default_rng(10 + i)
        stars["pmra"] = np.where(rng.random(len(stars)) < 0.8, rng.normal(4, 1, len(stars)), rng.normal(-15, 3, len(stars)))
        stars["distance_gspphot"] = 1000.0
        stars.drop(columns="color").to_csv(raw / cfg["clusters"][name]["file"], index=False)
    monkeypatch.setattr(pipeline, "load_mist_isochrones",
                        lambda *a, **k: {-1.5: FakeISOCMD(-1.5), -1.0: FakeISOCMD(-1.0), -0.5: FakeISOCMD(-0.5, 0.3)})
    return cfg


def test_load_gaia_and_members(cfg):
    df = load_gaia_csv(cfg["paths"]["raw_dir"] / "M5-result.csv", cfg["binaries"]["required_columns"])
    assert {"g", "bp", "rp", "color"}.issubset(df.columns)
    members = pmra_members(df, (-2, 10))
    assert 0 < len(members) < len(df)
    assert members["pmra"].between(-2, 10).all()


def test_membership_and_isochrone_figures(cfg):
    counts = pipeline.run_membership(cfg)
    assert set(counts) == {"M5", "M3"}
    pipeline.run_isochrone_fits(cfg)
    assert (cfg["paths"]["figures_dir"] / "isochrones" / "M5_adjust.png").exists()


@pytest.mark.parametrize("method", ["corrected", "legacy"])
def test_binary_fraction_and_age_plot(cfg, method):
    cfg["binaries"]["method"] = method
    table = pipeline.run_binary_fraction(cfg)
    assert list(table["cluster"]) == ["M5", "M3"]
    assert table["fraction"].between(0, 1).all()
    assert (cfg["paths"]["results_dir"] / "binary_fractions.csv").exists()
    assert (cfg["paths"]["figures_dir"] / "binary_fraction" / "M5.png").exists()
    pipeline.run_fraction_vs_age(cfg)
    assert (cfg["paths"]["figures_dir"] / "fraction_vs_age.png").exists()


def test_corrected_fraction_is_close_to_truth(cfg):
    table = pipeline.run_binary_fraction(cfg)
    # 25 % de binarias con q ~ U(0, 1) -> ~ 25 % * (1 - 0.4) = 15 % con q > 0.4 (la contaminación sube algo el valor)
    assert table["fraction"].between(0.08, 0.30).all()


def test_preliminary_requires_astropy_and_runs(cfg):
    pytest.importorskip("astropy")
    stars, _ = make_cluster(n=5000, binary_fraction=0.3, seed=7)
    stars = stars.rename(columns={"g": "phot_g_mean_mag", "bp": "phot_bp_mean_mag", "rp": "phot_rp_mean_mag"})
    stars.to_csv(cfg["paths"]["raw_dir"] / "ngc4755.csv", index=False)
    cfg["preliminary"] = {"method": "corrected", "q_values": [0.7, 0.4],
                          "clusters": {"ngc4755": {"file": "ngc4755.csv", "mag_limit": 20, "bins": [0.8, 2.0, 0.05],
                                                   "kernel_std": 0.3, "color_range": [0.8, 2.0]}}}
    table = pipeline.run_preliminary(cfg)
    assert table.shape[0] == 1
