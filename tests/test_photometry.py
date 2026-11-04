import numpy as np

from binfrac import photometry as ph


def test_equal_mass_binary_is_075_mag_brighter():
    assert np.isclose(ph.binary_magnitude(15.0, 1.0), 15.0 - 0.7526, atol=1e-3)


def test_binary_magnitude_decreases_with_q():
    q = np.array([0.4, 0.7, 1.0])
    shifts = 15.0 - ph.binary_magnitude(15.0, q)
    assert np.all(np.diff(shifts) > 0)
    assert np.allclose(shifts, [0.043, 0.27, 0.753], atol=0.005)  # 0.27 mag para q = 0.7 (informe)


def test_legacy_has_opposite_sign():
    assert np.isclose(ph.legacy_binary_magnitude(15.0, 0.7) - 15.0, 15.0 - ph.binary_magnitude(15.0, 0.7))


def test_apparent_magnitude():
    assert np.isclose(ph.apparent_magnitude(0.0, 10.0, 0.0), 0.0)
    assert np.isclose(ph.apparent_magnitude(2.0, 1000.0, 0.5), 2.0 + 10.0 - 0.5)
