import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import astro_utils as au

TLE1 = "1 A0690U 26209B   26317.77632001  .00000103  00000-0  60660-4 0  9995"
TLE2 = "2 A0690  98.6300  23.0223 0000842 128.4678 231.6578 14.26733240  8490"


def test_plane_angle_matches_vector_dot_product():
    for i1, o1, i2, o2 in [(98.63, 23.0, 99.2, 24.5), (28.5, 10, 0.0, 0), (51.6, 100, 98.0, 300)]:
        ref = np.degrees(np.arccos(au.h_hat(i1, o1) @ au.h_hat(i2, o2)))
        assert au.plane_angle(i1, o1, i2, o2) == pytest.approx(ref, abs=1e-9)


def test_plane_angle_special_cases():
    assert au.plane_angle(50, 0, 53, 0) == pytest.approx(3.0)                    # inclination only
    i, d = 98.63, 1.0                                                              # node only
    dth = au.plane_angle(i, 0, i, d)
    assert np.sin(np.radians(dth) / 2) == pytest.approx(np.sin(np.radians(i)) * np.sin(np.radians(d) / 2))


def test_general_dv_matches_vector_difference():
    rng = np.random.default_rng(1)
    for _ in range(50):
        v1, v2 = rng.uniform(5, 9, 2)
        g1, g2 = rng.uniform(-10, 10, 2)
        dth = rng.uniform(1, 60)
        r_hat = np.array([1.0, 0, 0])
        t1 = np.array([0, 1.0, 0])
        t2 = np.array([0, np.cos(np.radians(dth)), np.sin(np.radians(dth))])
        V1 = v1 * (np.cos(np.radians(g1)) * t1 + np.sin(np.radians(g1)) * r_hat)
        V2 = v2 * (np.cos(np.radians(g2)) * t2 + np.sin(np.radians(g2)) * r_hat)
        assert au.general_dv(v1, v2, g1, g2, dth) == pytest.approx(np.linalg.norm(V2 - V1), rel=1e-12)


def test_pure_plane_change_special_case():
    v = au.circular_speed(au.R_EARTH + 800)
    assert au.plane_change_dv(v, 1.0) == pytest.approx(0.1301, abs=1e-4)           # km/s
    assert au.general_dv(v, v, 0, 0, 30) == pytest.approx(au.plane_change_dv(v, 30), rel=1e-12)


def test_gauss_equations_against_a_numerical_impulse():
    r = au.R_EARTH + 800
    v = au.circular_speed(r)
    h = r * v
    inc, raan, u, dvn = 98.63, 23.0, 63.0, 0.001
    i, o, uu = np.radians(inc), np.radians(raan), np.radians(u)
    Q = (np.array([[np.cos(o), -np.sin(o), 0], [np.sin(o), np.cos(o), 0], [0, 0, 1]])
         @ np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]]))
    pos = Q @ np.array([np.cos(uu), np.sin(uu), 0]) * r
    tang = Q @ np.array([-np.sin(uu), np.cos(uu), 0])
    vel = v * tang + dvn * (Q @ np.array([0, 0, 1.0]))
    hv = np.cross(pos, vel)
    n = np.cross([0, 0, 1.0], hv)
    inc2 = np.degrees(np.arccos(hv[2] / np.linalg.norm(hv)))
    raan2 = np.degrees(np.arctan2(n[1], n[0])) % 360
    d_inc, d_raan = au.gauss_normal_impulse(r, h, inc, u, dvn)
    assert d_inc == pytest.approx(inc2 - inc, rel=5e-3)
    assert d_raan == pytest.approx(raan2 - raan, rel=5e-3)


def test_hohmann_with_plane_change_numbers():
    r1, r2 = au.R_EARTH + 800, 42164.0
    assert au.hohmann_plane_change(r1, r2, 0.0) == pytest.approx(3.706, abs=2e-3)   # no plane change
    assert au.hohmann_plane_change(r1, r2, 28.5, 0.0) == pytest.approx(4.090, abs=2e-3)   # all at apogee
    beta, total = au.optimal_split(r1, r2, 28.5)
    assert total == pytest.approx(4.063, abs=2e-3)
    assert beta == pytest.approx(2.35, abs=0.05)
    assert total < au.hohmann_plane_change(r1, r2, 28.5, 0.0)


def test_local_time_of_node_for_sentinel3c_tle():
    # TLE RAAN 23.0223 deg; mean Sun right ascension at the TLE epoch ~ 232.9 deg (from GMST)
    ltdn = au.local_time_of_node(23.0223, 232.9, descending=True)
    assert ltdn == pytest.approx(10.0, abs=0.05)                                   # nominal 10:00
    assert au.local_time_of_node(23.0223, 232.9, descending=False) == pytest.approx(22.0, abs=0.05)


def test_tle_parsing_and_checksum():
    assert au.tle_checksum_ok(TLE1) and au.tle_checksum_ok(TLE2)
    bad = TLE2[:-1] + "1"
    assert not au.tle_checksum_ok(bad)
    t = au.parse_tle(TLE1, TLE2)
    assert (t["epoch"].year, t["epoch"].month, t["epoch"].day) == (2026, 11, 13)
    assert (t["epoch"].hour, t["epoch"].minute, t["epoch"].second) == (18, 37, 54)
    assert t["inc"] == 98.63 and t["n_rev_day"] == pytest.approx(14.2673324)
