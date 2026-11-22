"""
orbit_static.py - J2-perturbed orbit report with a real-satellite (TLE) overlay.

Propagates a satellite from its LAUNCH date/time to a FINAL date/time that you type in the box
(DD/MM/YYYY HH:MM:SS, UTC) with the J2 perturbation, and draws, as static plots, two orbits:
  - ORBIT 0      = the first orbit, at launch                       (orange)
  - FINAL ORBIT  = the orbit the satellite is on at the chosen date (green)
and, on top of them, the REAL orbit and position of the satellite taken from a TLE (white).

What is shown
  - 3D view: Earth, equatorial plane, both orbits, perigee (P) / apogee (A), delta-Omega (angle
    between the two ascending nodes), delta-omega, the real orbit and satellite positions (stars).
  - 2D map: ground tracks, ascending-node crossings, real vs simulated track and position.
  - Top right: values of the initial and of the final orbit (inclination, eccentricity, period,
    angular momentum), node drift compared with the motion of the Sun (sun-synchronous check),
    and the exact date/time of the TLE.
  - Plots: altitude and velocities along one orbit at 12 dates between launch and final date;
    e, i, RAAN, argument of perigee and semi-major axis versus days since launch.
  - An A4 PDF with all the 2D graphs and a position table (simulated vs real) is written to
    PDF_FILE every time a date is computed.

Orbit set-up: perigee/apogee altitude, inclination, RAAN (or the local solar time of the
descending node, LTDN), argument of perigee (or a launch site), LAUNCH date/time and the TLE are
parameters at the top of this file. The values given here are an example: Sentinel-3C, fitted to
its real TLE with fit_tle.py.

Reference frames and time
  - Inertial frame (ECI): X towards the vernal equinox, Z towards the north pole.
  - Earth orientation: Greenwich Mean Sidereal Time (GMST) from the launch date/time (UTC),
    advanced at the sidereal rotation rate (Earth texture and ground track).
  - Dates are UTC (UT1 ~ UTC, better than a second).
  - Orbital elements are osculating, taken at each orbit start.
  - The "real" orbit propagates the TLE mean elements with the secular J2 drift (no SGP4
    short-period terms, no drag): good to a few km near the TLE epoch.

Units: km, seconds, km/s
"""

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, TextBox
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from astro_utils import parse_utc, tle_set_epoch

# ----------------------------------------------------------------
# PARAMETERS (edit and re-run)
# ----------------------------------------------------------------
MU = 398600.4418            # Earth gravitational parameter [km^3/s^2]
R_EARTH = 6378.137          # Earth equatorial radius [km]
J2 = 1.08262668e-3          # Earth oblateness coefficient [-]
OMEGA_EARTH = 7.2921158553e-5   # sidereal rotation rate of the Earth [rad/s]

USE_J2 = True               # False = switch J2 off (pure two-body, for comparison)
J2_FACTOR = 1.0             # 1 = real J2; 20 = effect 20 times larger (didactic only)

PERIGEE_ALT = 800       # perigee altitude [km]   (fitted to the real TLE with fit_tle.py)
APOGEE_ALT = 810        # apogee altitude [km]
INCLINATION = 98.63         # [deg]
RAAN = 30.0                 # right ascension of the ascending node [deg]
ARG_PERIGEE = 157.905       # argument of perigee = start argument of latitude [deg]
LTDN = 10.0001              # local solar time of the descending node [h]; if set, it overrides RAAN

LAUNCH = dt.datetime(2026,9,15,1,21)     # Vega-C VV30: 15/09/2026 01:21 UTC
LAUNCH_SITE = None          # (lat, lon) of the launch site, e.g. (5.236, -52.775) for Kourou. If set, it
                            # overrides ARG_PERIGEE (the satellite starts over that latitude). None = off;
                            # keep it off when the orbit is fitted to a TLE (fit_tle.py)
LAUNCH_ASCENDING = True     # True = launch heading north (ascending), False = heading south
DEFAULT_DAYS = 59.7200700   # date proposed at start = launch + this many days (here: the TLE epoch)
MAX_DAYS = 730              # latest accepted date (the further away, the longer the computation)

POINTS_PER_ORBIT = 400      # resolution of curves and trails

STAR_COLORS = ["magenta", "cyan"]      # star marking the satellite at launch / at the chosen date
COLORS = ["orange", "#9dff4d"]    # colour of orbit 0 and of the final orbit (same as 2D map)
SHOW_PERIGEE_APOGEE = True        # P (perigee) and A (apogee) points on the two 3D orbits
SHOW_EQUATORIAL_PLANE = True      # transparent, very large equatorial plane in the 3D view
ORBITS_ON_TOP = True              # orbits always in front, even behind the Earth
SHOW_TLE = True                   # draw the REAL orbit and position of the satellite from the TLE
REAL_COLOR = "white"              # colour of the real orbit / real satellite
# Real TLE of Sentinel-3C (NORAD 100690), epoch 13/11/2026 18:37:54 UTC (source: n2yo.com).
# Paste a newer TLE here to update the real data.
TLE_NAME = "SENTINEL-3C (NORAD 100690)"
TLE_LINE1 = "1 A0690U 26209B   26326.19191917  .00000079  00000-0  50528-4 0  9991"
TLE_LINE2 = "2 A0690  98.6294  31.3203 0001034 126.6768 233.4507 14.26735463  9690"
# Epoch of the TLE as a readable UTC date, DD/MM/YYYY HH:MM:SS(.fff). If set, the program converts it
# to the TLE format (YYDDD.DDDDDDDD) and writes it into TLE_LINE1, checksum included. Note: changing
# the date does NOT change the orbital elements; use it only to correct or set the epoch of the
# elements you paste. None = keep the epoch that is already written in TLE_LINE1.
TLE_EPOCH = None
SHOW_GREENWICH = False            # Greenwich meridian line on the 3D Earth

TEXTURE_FILE = "terra.jpg"        # Earth map (equirectangular, 2:1)
EARTH_RESOLUTION = (360,180)       # (longitude, latitude) points; lower = smoother 3D rotation
SHOW_AXES = True
FAST_X_AXIS = "date"              # x axis of altitude/velocity plots: "date", "longitude", "latitude" or "phase"
FAST_SPAN_ORBITS = 1.0            # length of each curve in orbits
PDF_AUTO = True                   # save the 2D graphs as an A4 PDF at every computation
PDF_FILE = "orbit_graphs.pdf"     # name of the PDF (saved in the same folder as this script)
FAST_INTERMEDIATE = 10            # intermediate dates drawn between launch and chosen date
AXES_LENGTH = 2.2 * R_EARTH



# ----------------------------------------------------------------
# TIME: Julian date and sidereal time
# ----------------------------------------------------------------
def julian_date(t_utc):
    """Julian date of a naive datetime interpreted as UTC."""
    return 2440587.5 + (t_utc - dt.datetime(1970, 1, 1)).total_seconds() / 86400.0


def gmst_deg(t_utc):
    """Greenwich Mean Sidereal Time [deg, 0-360) (Meeus, Astronomical Algorithms, 12.4).
    It is the angle between the vernal equinox (X axis) and the Greenwich meridian."""
    d = julian_date(t_utc) - 2451545.0            # days since J2000.0
    T = d / 36525.0                               # Julian centuries
    return (280.46061837 + 360.98564736629 * d
            + 0.000387933 * T ** 2 - T ** 3 / 38710000.0) % 360.0


def raan_from_ltdn(launch, ltdn):
    """RAAN [deg] that gives the requested mean local solar time at the descending node."""
    ut = launch.hour + launch.minute / 60                 # UT in hours
    sun_ra = (gmst_deg(launch) - 15 * ut + 180) % 360     # right ascension of the mean Sun
    ltan = (ltdn + 12) % 24                               # ascending node = descending + 12 h
    return (sun_ra + 15 * (ltan - 12)) % 360


# ----------------------------------------------------------------
# PHYSICS
# ----------------------------------------------------------------
def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def initial_state(rp, ra, inc, raan, argp):
    """State (r, v) at perigee from perigee/apogee radii and angles."""
    a = (rp + ra) / 2.0
    e = (ra - rp) / (ra + rp)
    vp = np.sqrt(MU * (1 + e) / rp)
    Q = rot_z(np.radians(raan)) @ rot_x(np.radians(inc)) @ rot_z(np.radians(argp))
    return Q @ np.array([rp, 0.0, 0.0]), Q @ np.array([0.0, vp, 0.0]), a, e


def equations(t, s, use_j2):
    """State derivative: gravity of a spherical Earth (+ J2 if use_j2).
        a = -mu r / r^3  -  (3/2) J2 mu R^2 / r^5 * [x(1-5z^2/r^2), y(1-5z^2/r^2), z(3-5z^2/r^2)]
    """
    x, y, z = s[0], s[1], s[2]
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    g = -MU / (r2 * r)
    ax_, ay_, az_ = g * x, g * y, g * z
    if use_j2:
        k = 1.5 * J2 * J2_FACTOR * MU * R_EARTH ** 2 / (r2 * r2 * r)
        f = 5.0 * z * z / r2
        ax_ -= k * x * (1 - f)
        ay_ -= k * y * (1 - f)
        az_ -= k * z * (3 - f)
    # Other perturbations (drag, Moon, Sun...) could be added here
    return [s[3], s[4], s[5], ax_, ay_, az_]


def quantities(S):
    """From a state (6,) or (6,n): altitude, radial velocity, tangential velocity, speed."""
    r_vec, v_vec = S[:3], S[3:]
    r = np.linalg.norm(r_vec, axis=0)
    v = np.linalg.norm(v_vec, axis=0)
    vr = np.sum(r_vec * v_vec, axis=0) / r                            # along r
    vt = np.linalg.norm(np.cross(r_vec, v_vec, axis=0), axis=0) / r   # perpendicular to r
    return r - R_EARTH, vr, vt, v


def elements(S):
    """From a state (6,) or (6,n): eccentricity, inclination, RAAN, argument of perigee
    (angles in degrees). These are INSTANTANEOUS (osculating) values."""
    r_vec, v_vec = S[:3], S[3:]
    r = np.linalg.norm(r_vec, axis=0)
    v2 = np.sum(v_vec * v_vec, axis=0)
    rv = np.sum(r_vec * v_vec, axis=0)
    e_vec = ((v2 - MU / r) * r_vec - rv * v_vec) / MU        # eccentricity vector, points to perigee
    h_vec = np.cross(r_vec, v_vec, axis=0)                   # angular momentum, normal to the plane
    h = np.linalg.norm(h_vec, axis=0)
    ecc = np.linalg.norm(e_vec, axis=0)
    inc = np.arccos(h_vec[2] / h)
    nx, ny = -h_vec[1], h_vec[0]                             # line of nodes = Z x h
    n = np.hypot(nx, ny)
    raan = np.arctan2(ny, nx)
    cosw = np.clip((nx * e_vec[0] + ny * e_vec[1]) / (n * ecc), -1.0, 1.0)
    argp = np.where(e_vec[2] < 0, 2 * np.pi - np.arccos(cosw), np.arccos(cosw))
    return ecc, np.degrees(inc), np.degrees(raan) % 360.0, np.degrees(argp)


def semi_major_axis(S):
    """Osculating semi-major axis [km] from a state (6,) or (6,n) (vis-viva)."""
    r = np.linalg.norm(S[:3], axis=0)
    v2 = np.sum(S[3:] * S[3:], axis=0)
    return 1.0 / (2.0 / r - v2 / MU)


def arc(u, v, radius, n=40):
    """(n,3) points of a circular arc of given radius between directions u and v (smaller angle)."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    c = np.clip(np.dot(u, v), -1.0, 1.0)
    ang = np.arccos(c)
    w = v - c * u
    nw = np.linalg.norm(w)
    if nw < 1e-12:
        return radius * np.tile(u, (n, 1))
    w = w / nw
    t = np.linspace(0.0, ang, n)
    return radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), w))


def lat_lon(P, t, theta0):
    """Latitude and longitude [deg] under the satellite.
    The Earth rotation angle at time t is theta = theta0 + OMEGA_EARTH * t, where theta0 is the
    GMST at launch. The longitude is the inertial angle of the satellite minus theta."""
    lon = np.arctan2(P[1], P[0]) - (theta0 + OMEGA_EARTH * t)
    lon = (lon + np.pi) % (2 * np.pi) - np.pi
    lat = np.arcsin(P[2] / np.linalg.norm(P[:3], axis=0))
    return np.degrees(lon), np.degrees(lat)


def with_breaks(lon, lat):
    """Insert NaN where the track crosses the +-180 edge: no horizontal lines across the map."""
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    return np.insert(lon, jumps, np.nan), np.insert(lat, jumps, np.nan)


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def first_node(P, t, theta0):
    """First south-to-north equator crossing in a series of states P (6,n) at times t.
    Returns (time, longitude) or None."""
    z = P[2]
    k = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
    if len(k) == 0:
        return None
    k = k[0]
    f = -z[k] / (z[k + 1] - z[k])
    pos = P[:3, k] + f * (P[:3, k + 1] - P[:3, k])
    tc = t[k] + f * (t[k + 1] - t[k])
    lon, _ = lat_lon(pos, tc, theta0)
    return tc, float(lon)


# ----------------------------------------------------------------
# REAL SATELLITE: position from the TLE (mean elements + secular J2, Kepler equation)
# ----------------------------------------------------------------
def parse_tle(l1, l2):
    year = 2000 + int(l1[18:20])
    epoch = dt.datetime(year, 1, 1) + dt.timedelta(days=float(l1[20:32]) - 1.0)
    return SimpleNamespace(epoch=epoch, inc=float(l2[8:16]), raan=float(l2[17:25]),
                           ecc=float("0." + l2[26:33].strip()), argp=float(l2[34:42]),
                           M=float(l2[43:51]), n_rev=float(l2[52:63]))


def fix_tle(l1, l2):
    """Rebuilds both lines in the exact TLE columns: copying a TLE from a web page or an editor often
    changes the number of spaces, and the parser reads fixed columns."""
    a, b = l1.split(), l2.split()
    n, tail = b[7][:11], b[7][11:] + "".join(b[8:])
    l1 = (f"1 {a[1]:<6} {a[2]:<8} {a[3]:>14} {a[4]:>10} {a[5]:>8} {a[6]:>8} {a[7]} {a[8]:>5}")
    l2 = (f"2 {b[1]:<5} {b[2]:>8} {b[3]:>8} {b[4]:>7} {b[5]:>8} {b[6]:>8} {n:>11}{tail:>6}")
    return l1, l2


TLE_LINE1, TLE_LINE2 = fix_tle(TLE_LINE1, TLE_LINE2)


if TLE_EPOCH:
    TLE_LINE1 = tle_set_epoch(TLE_LINE1, parse_utc(TLE_EPOCH))
TLE = parse_tle(TLE_LINE1, TLE_LINE2) if SHOW_TLE else None


def tle_positions(t_after_epoch):
    """ECI position [km] (3, n) of the real satellite at t_after_epoch [s] from the TLE epoch.
    Mean elements with the secular J2 drift of RAAN and argument of perigee (no SGP4 short-period
    terms, no drag): good to a few km near the epoch, slowly worse further away."""
    t = np.atleast_1d(np.asarray(t_after_epoch, dtype=float))
    n = TLE.n_rev * 2 * np.pi / 86400.0
    a_t = (MU / n ** 2) ** (1 / 3)
    ec = TLE.ecc
    i = np.radians(TLE.inc)
    pp = a_t * (1 - ec ** 2)
    k = J2 * (R_EARTH / pp) ** 2 * n
    raan = np.radians(TLE.raan) - 1.5 * k * np.cos(i) * t
    argp = np.radians(TLE.argp) + 0.75 * k * (4 - 5 * np.sin(i) ** 2) * t
    Mm = np.radians(TLE.M) + n * t
    E = Mm.copy()
    for _ in range(12):
        E = E - (E - ec * np.sin(E) - Mm) / (1 - ec * np.cos(E))
    xp, yp = a_t * (np.cos(E) - ec), a_t * np.sqrt(1 - ec ** 2) * np.sin(E)
    cw, sw, co, so = np.cos(argp), np.sin(argp), np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    x = (co * cw - so * sw * ci) * xp + (-co * sw - so * cw * ci) * yp
    y = (so * cw + co * sw * ci) * xp + (-so * sw + co * cw * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return np.array([x, y, z])


# ----------------------------------------------------------------
# COMPUTATION: propagation up to the chosen date
# ----------------------------------------------------------------
launch_dt = LAUNCH or dt.datetime.combine(dt.date.today(), dt.time(0, 0))   # launch instant
RAAN_USED = raan_from_ltdn(launch_dt, LTDN) if LTDN is not None else RAAN
if LAUNCH_SITE is not None:
    # the satellite starts at the perigee, so argument of perigee = argument of latitude u,
    # and sin(lat) = sin(i) sin(u)
    _u = np.degrees(np.arcsin(np.sin(np.radians(LAUNCH_SITE[0])) / np.sin(np.radians(INCLINATION))))
    ARG_PERIGEE_USED = _u if LAUNCH_ASCENDING else 180.0 - _u
else:
    ARG_PERIGEE_USED = ARG_PERIGEE
r0, v0, a, e = initial_state(R_EARTH + PERIGEE_ALT, R_EARTH + APOGEE_ALT,
                             INCLINATION, RAAN_USED, ARG_PERIGEE_USED)
T = 2 * np.pi * np.sqrt(a ** 3 / MU)         # Keplerian period (two-body)
s0 = np.concatenate((r0, v0))
TAU = np.linspace(0.0, 1.0, POINTS_PER_ORBIT + 1)


NEAR_CIRCULAR_E = 0.02      # below this the "orbit" is counted from the argument of latitude


def anomalistic_period():
    """Time between two perigee passages with J2 (slightly different from T)."""
    sol = solve_ivp(equations, [0, 1.5 * T], s0, args=(USE_J2,), method="DOP853",
                    dense_output=True, rtol=1e-11, atol=1e-11)
    tg = np.linspace(0.8 * T, 1.2 * T, 8001)
    Pg = sol.sol(tg)
    if e < NEAR_CIRCULAR_E:
        # perigee is not defined for a nearly circular orbit (it jumps around because of the
        # short-period J2 terms): use the return to the same argument of latitude u instead
        def u_of(P):
            h = np.cross(P[:3], P[3:], axis=0)
            hn = h / np.linalg.norm(h, axis=0)
            nn = np.array([-hn[1], hn[0], np.zeros_like(hn[0])])
            nn = nn / np.linalg.norm(nn, axis=0)
            return np.arctan2(np.sum(np.cross(nn, P[:3], axis=0) * hn, axis=0),
                              np.sum(nn * P[:3], axis=0))
        u0 = float(u_of(sol.sol(np.array([0.0])))[0])
        ug = np.unwrap(u_of(Pg))
        target = u0 + 2 * np.pi
        while target - ug[0] > np.pi:                # bring the unwrapped branch next to the target
            ug += 2 * np.pi
        while ug[0] - target > np.pi:
            ug -= 2 * np.pi
        k = np.where((ug[:-1] < target) & (ug[1:] >= target))[0][0]
        return tg[k] + (target - ug[k]) / (ug[k + 1] - ug[k]) * (tg[k + 1] - tg[k])
    vr = np.sum(Pg[:3] * Pg[3:], axis=0)
    k = np.where((vr[:-1] < 0) & (vr[1:] >= 0))[0][0]
    return tg[k] + (-vr[k]) / (vr[k + 1] - vr[k]) * (tg[k + 1] - tg[k])


def compute(days, launch):
    """Propagate (with J2) for `days` days and prepare all the data to display."""
    D_s = days * 86400.0
    Ta = anomalistic_period()
    N = int(D_s // Ta)                    # complete orbits before the chosen date
    t_end = (N + 1 + max(0.0, FAST_SPAN_ORBITS - 1.0)) * Ta * 1.0005
    theta0 = np.radians(gmst_deg(launch))  # Earth rotation angle at launch (GMST)

    sol = solve_ivp(equations, [0, t_end], s0, args=(USE_J2,), method="DOP853",
                    dense_output=True, rtol=1e-10, atol=1e-10)

    # --- slow plots: one sample per orbit (at each perigee passage)
    tk = np.arange(0, N + 2) * Ta
    Pk = sol.sol(tk)
    E, I, OM, W = elements(Pk)
    OM, W = (np.degrees(np.unwrap(np.radians(x))) for x in (OM, W))   # no 359 -> 0 jumps
    SMA = semi_major_axis(Pk)

    # --- the two orbits shown
    windows = []
    for o, k in enumerate((0, N)):
        t_start = k * Ta
        tw = t_start + TAU * Ta
        Pw = sol.sol(tw)
        w = SimpleNamespace(t_start=t_start, tw=tw, P=Pw)
        w.H, w.VR, w.VT, _ = quantities(Pw)
        # longer window for the altitude/velocity plots (FAST_SPAN_ORBITS orbits from the perigee)
        span = max(1.0, FAST_SPAN_ORBITS)
        w.XPH = np.linspace(0.0, span, int(POINTS_PER_ORBIT * span) + 1)      # orbit fraction
        tx = t_start + w.XPH * Ta
        Px = sol.sol(tx)
        w.XH, w.XVR, w.XVT, _ = quantities(Px)
        w.XLON, w.XLAT = lat_lon(Px, tx, theta0)
        w.LON, w.LAT = lat_lon(Pw, tw, theta0)
        # perigee near one end of the window, apogee in the central part
        R = np.linalg.norm(Pw[:3], axis=0)
        m = len(R)
        ends = np.r_[0:int(0.2 * m), int(0.8 * m):m]
        w.idx_p = int(ends[np.argmin(R[ends])])
        w.idx_a = int(0.2 * m + np.argmax(R[int(0.2 * m):int(0.8 * m)]))
        # ascending equator crossing (longitude under the satellite)
        w.node = first_node(Pw, tw, theta0)
        w.node_frac = None if w.node is None else (w.node[0] - t_start) / Ta
        # elements of this orbit
        w.raan = float(OM[k] % 360.0)                     # RAAN from the perigee-passage series
        w.v_peri = Pw[:3, w.idx_p] / np.linalg.norm(Pw[:3, w.idx_p])   # perigee direction
        w.v_node = np.array([np.cos(np.radians(w.raan)), np.sin(np.radians(w.raan)), 0.0])
        # values shown in the top-right panel (osculating, at the perigee passage of this orbit)
        w.k = k
        w.inc = float(I[k])
        w.ecc = float(E[k])
        w.a_osc = float(SMA[k])
        w.period = 2 * np.pi * np.sqrt(w.a_osc ** 3 / MU) / 60.0     # osculating Kepler period [min]
        w.h = float(np.linalg.norm(np.cross(Pk[:3, k], Pk[3:, k])))  # specific ang. momentum [km^2/s]
        w.t_perigee = launch + dt.timedelta(seconds=float(k * Ta))
        windows.append(w)

    # --- where the satellite is at launch and exactly at the chosen date
    tt = np.array([0.0, D_s])
    Pp = sol.sol(tt)
    lon_s, lat_s = lat_lon(Pp, tt, theta0)
    sats = [SimpleNamespace(date=launch + dt.timedelta(seconds=float(t_)),
                            xyz=Pp[:3, q].copy(), lat=float(lat_s[q]), lon=float(lon_s[q]),
                            alt=float(np.linalg.norm(Pp[:3, q]) - R_EARTH))
            for q, t_ in enumerate(tt)]

    # --- real satellite (TLE) at the chosen date: position and one full orbit around it
    real = None
    if TLE is not None:
        t_ep = (TLE.epoch - launch).total_seconds()                 # TLE epoch, s from launch
        T_real = 86400.0 / TLE.n_rev
        tr_ = D_s + np.linspace(0.0, T_real, POINTS_PER_ORBIT + 1)
        Pr = tle_positions(tr_ - t_ep)
        lon_r, lat_r = lat_lon(Pr, tr_, theta0)
        Ps = sol.sol(tr_)                                           # simulated orbit, same time window
        lon_q, lat_q = lat_lon(Ps, tr_, theta0)
        real = SimpleNamespace(simP=Ps, simLON=lon_q, simLAT=lat_q, date=launch + dt.timedelta(seconds=float(D_s)), xyz=Pr[:3, 0].copy(),
                               lat=float(lat_r[0]), lon=float(lon_r[0]),
                               alt=float(np.linalg.norm(Pr[:3, 0]) - R_EARTH),
                               P=Pr, LON=lon_r, LAT=lat_r, days_from_epoch=(D_s - t_ep) / 86400.0)

    # --- altitude/velocity curves: launch, FAST_INTERMEDIATE intermediate dates, final date
    ks = np.unique(np.round(np.linspace(0, N, FAST_INTERMEDIATE + 2)).astype(int))
    epochs = []
    for k in ks:
        t_start = k * Ta
        XPH = np.linspace(0.0, max(1.0, FAST_SPAN_ORBITS), int(POINTS_PER_ORBIT * max(1.0, FAST_SPAN_ORBITS)) + 1)
        tx = t_start + XPH * Ta
        Px = sol.sol(tx)
        ep = SimpleNamespace(k=int(k), date=launch + dt.timedelta(seconds=float(t_start)))
        ep.XH, ep.XVR, ep.XVT, _ = quantities(Px)
        ep.XLON, ep.XLAT = lat_lon(Px, tx, theta0)
        ep.XPH = XPH
        epochs.append(ep)

    # --- FINAL DATA: orbit 0 vs final orbit
    wa, wb = windows
    d_raan_acc = float(OM[N] - OM[0])                     # accumulated (with the revolutions)
    d_raan = float(wrap180(d_raan_acc))                   # angle between the nodes
    ang_peri = float(np.degrees(np.arccos(np.clip(np.dot(wa.v_peri, wb.v_peri), -1, 1))))
    d_argp_acc = float(W[N] - W[0])

    S = SimpleNamespace(days=days, D_s=D_s, Ta=Ta, N=N, launch=launch,
                        end_date=launch + dt.timedelta(days=days), theta0=theta0,
                        sol=sol, x_days=tk / 86400.0, win=windows, epochs=epochs, sats=sats, real=real,
                        E=E, I=I, OM=OM, W=W, SMA=SMA,
                        d_raan=d_raan, d_raan_acc=d_raan_acc, ang_peri=ang_peri,
                        d_argp_acc=d_argp_acc)
    # check against theory (mean secular rates per orbit)
    p = a * (1 - e ** 2)
    n_mean = np.sqrt(MU / a ** 3)
    fk = J2 * J2_FACTOR * (R_EARTH / p) ** 2
    inc_r = np.radians(INCLINATION)
    th_node = np.degrees(-1.5 * n_mean * fk * np.cos(inc_r)) * Ta
    th_peri = np.degrees(0.75 * n_mean * fk * (4 - 5 * np.sin(inc_r) ** 2)) * Ta
    print(f"Launch {launch:%d/%m/%Y %H:%M} UTC | GMST at launch = {np.degrees(theta0):.4f} deg "
          f"({np.degrees(theta0) / 15:.4f} sidereal hours)")
    print(f"Keplerian period T = {T / 60:.2f} min | period with J2 = {Ta / 60:.2f} min")
    print(f"Complete orbits up to the chosen date: {N}")
    print(f"Angle between nodes (RAAN): {d_raan:+.3f} deg (accumulated {d_raan_acc:+.3f}) | "
          f"angle between perigees: {ang_peri:.3f} deg")
    if USE_J2 and N > 0:
        print(f"Node regression:      theory {th_node:+.5f} deg/orbit | "
              f"simulated {d_raan_acc / N:+.5f}")
        print(f"Perigee precession:   theory {th_peri:+.5f} deg/orbit | "
              f"simulated {d_argp_acc / N:+.5f}")
    return S



# ----------------------------------------------------------------
# EARTH TEXTURE
# ----------------------------------------------------------------
def load_texture(file_name):
    path = Path(__file__).parent / file_name
    img = None
    if path.exists():
        img = plt.imread(path)
    else:
        try:
            import pyvista as pv
            img = pv.examples.load_globe_texture().to_array()
            plt.imsave(path, img)
            print(f"Texture downloaded and saved as {file_name}")
        except Exception as err:
            print(f"Texture not available ({err}). Using a blue sphere.")
            return None
    img = img[..., :3]
    if img.dtype == np.uint8:
        img = img / 255.0
    return img


# ----------------------------------------------------------------
# FIGURE (containers are created once; the content is redrawn for every date)
# ----------------------------------------------------------------
BG = "black"
FG = "white"

fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.canvas.manager.set_window_title("Orbit with J2 - launch vs chosen date")
gs = GridSpec(5, 3, width_ratios=[1.3, 1, 1], height_ratios=[2.15, 1, 1, 1, 1],
              left=0.045, right=0.985, top=0.95, bottom=0.12,
              hspace=0.85, wspace=0.17)
ax = fig.add_subplot(gs[0:3, 0], projection="3d", facecolor=BG,
                     computed_zorder=not ORBITS_ON_TOP)
ax_map = fig.add_subplot(gs[3:5, 0], facecolor=BG)
ax_txt = fig.add_subplot(gs[0, 1:3], facecolor=BG)
ax_h = fig.add_subplot(gs[1, 1])
ax_vr = fig.add_subplot(gs[2, 1])
ax_vt = fig.add_subplot(gs[3, 1])
ax_e = fig.add_subplot(gs[4, 1])
ax_i = fig.add_subplot(gs[1, 2])
ax_om = fig.add_subplot(gs[2, 2])
ax_w = fig.add_subplot(gs[3, 2])
ax_d = fig.add_subplot(gs[4, 2])
ALL_AXES = (ax, ax_map, ax_h, ax_vr, ax_vt, ax_e, ax_i, ax_om, ax_w, ax_d)

texture = load_texture(TEXTURE_FILE)

ax_txt.set_axis_off()
info_text = ax_txt.text(0.0, 1.0, "", va="top", ha="left", color=FG,
                        family="monospace", fontsize=9, transform=ax_txt.transAxes)

S = None      # data of the current propagation


def orbit_label(o):
    return "Orbit 0 (launch)" if o == 0 else f"Final orbit (no. {S.N + 1})"


# ----------------------------------------------------------------
# 3D SCENE (static)
# ----------------------------------------------------------------
def build_3d(axx):
    """Draw the 3D scene (Earth, plane, axes, orbits, P/A, delta Omega, delta omega) on `axx`.
    The Earth is shown with its orientation at launch (GMST at launch)."""
    w0, w1 = S.win
    n_lon, n_lat = EARTH_RESOLUTION
    lon = np.linspace(-np.pi, np.pi, n_lon)
    colat = np.linspace(0, np.pi, n_lat)
    xt = R_EARTH * np.outer(np.cos(lon), np.sin(colat))
    yt = R_EARTH * np.outer(np.sin(lon), np.sin(colat))
    zt = R_EARTH * np.outer(np.ones_like(lon), np.cos(colat))
    if texture is not None:
        h_img, w_img = texture.shape[:2]
        rows = texture[(colat / np.pi * (h_img - 1)).astype(int)]          # (n_lat, w_img, 3)
        geo_lon = (lon - S.theta0 + np.pi) % (2 * np.pi)                    # geographic lon + pi
        cols = (geo_lon / (2 * np.pi) * (w_img - 1)).astype(int)
        colors = rows[:, cols].transpose(1, 0, 2)                           # (n_lon, n_lat, 3)
        axx.plot_surface(xt, yt, zt, facecolors=colors, shade=False, zorder=1,
                         linewidth=0, antialiased=False, rstride=1, cstride=1)
    else:
        axx.plot_surface(xt, yt, zt, color="royalblue", alpha=0.6, linewidth=0, zorder=1)

    lim = max(np.abs(w1.P[:3]).max(), R_EARTH) * 1.05
    if SHOW_EQUATORIAL_PLANE:
        # plane z = 0 much larger than the scene: at the view edges it looks infinite
        half = 4.0 * lim
        gx = np.array([-half, half])
        PX, PY = np.meshgrid(gx, gx)
        axx.plot_surface(PX, PY, np.zeros_like(PX), color="#4dd2ff", alpha=0.13,
                         shade=False, linewidth=0, antialiased=False, zorder=1.5)
        c = np.arange(-half, half + 1e-6, lim / 2)
        nan = np.array([np.nan])
        gxs = np.concatenate([np.r_[[ci, ci], nan] for ci in c] +
                             [np.r_[[-half, half], nan] for _ in c])
        gys = np.concatenate([np.r_[[-half, half], nan] for _ in c] +
                             [np.r_[[ci, ci], nan] for ci in c])
        axx.plot(gxs, gys, np.zeros_like(gxs), color="#4dd2ff", alpha=0.22, lw=0.5, zorder=1.6)

    if SHOW_AXES:
        La = min(AXES_LENGTH, 0.9 * lim)      # axes never longer than the view
        for vec, color, name in [((La, 0, 0), "red", "X (γ)"),
                                 ((0, La, 0), "lime", "Y"),
                                 ((0, 0, La), "cyan", "Z (north pole)")]:
            axx.quiver(0, 0, 0, *vec, color=color, lw=2, arrow_length_ratio=0.08)
            axx.text(*(1.08 * np.array(vec)), name, color=color, fontsize=10)

    handles = []
    if SHOW_GREENWICH:     # Greenwich meridian at launch: half great circle from pole to pole
        c = np.linspace(0.0, np.pi, 60)
        rr = 1.004 * R_EARTH
        g, = axx.plot(rr * np.sin(c) * np.cos(S.theta0), rr * np.sin(c) * np.sin(S.theta0),
                      rr * np.cos(c), color="#ffd24d", lw=2.4, zorder=4,
                      label="Greenwich meridian (at launch)")
        handles.append(g)

    # the two orbits at the chosen date: simulated (green) and real from the TLE (white, dashed)
    sim_P = S.real.simP if S.real is not None else w1.P
    handles += axx.plot(*sim_P[:3], color=COLORS[1], lw=3.0, zorder=6,
                        label="Simulated orbit (chosen date)")
    sat_sim = S.sats[1]
    axx.plot([sat_sim.xyz[0]], [sat_sim.xyz[1]], [sat_sim.xyz[2]], marker="*", ls="none",
             markersize=15, color=STAR_COLORS[1], markeredgecolor="white", markeredgewidth=0.6,
             zorder=30)
    handles.append(Line2D([], [], marker="*", ls="none", markersize=11, color=STAR_COLORS[1],
                          markeredgecolor="white", label="Simulated satellite"))
    if S.real is not None:
        handles += axx.plot(*S.real.P[:3], color=REAL_COLOR, lw=1.8, ls="--", zorder=7,
                            label="Real orbit (TLE)")
        axx.plot([S.real.xyz[0]], [S.real.xyz[1]], [S.real.xyz[2]], marker="*", ls="none",
                 markersize=17, color=REAL_COLOR, markeredgecolor="red", markeredgewidth=1.0,
                 zorder=31)
        handles.append(Line2D([], [], marker="*", ls="none", markersize=12, color=REAL_COLOR,
                              markeredgecolor="red", label="Real satellite"))
    leg = axx.legend(handles=handles, loc="upper left", frameon=False, labelcolor=FG,
                     fontsize=9)
    for h in leg.legend_handles:
        if h.get_linestyle() not in ("None", "--", ":"):
            h.set_linewidth(2.5)
    axx.set_xlim(-lim, lim)
    axx.set_ylim(-lim, lim)
    axx.set_zlim(-lim, lim)
    axx.set_box_aspect((1, 1, 1), zoom=1.12)
    axx.set_axis_off()
    axx.view_init(elev=25, azim=300)     # a view from which the angles read well


# ----------------------------------------------------------------
# DRAW: 3D view, map and plots
# ----------------------------------------------------------------
def draw():
    """Clear and redraw everything from the data S."""
    for a_ in ALL_AXES:
        a_.clear()
    build_3d(ax)
    draw_2d()


def draw_2d():
    """Ground track and the 8 plots, drawn on the axes ax_map, ax_h, ax_vr, ... (globals)."""
    w0, w1 = S.win

    # ---------------- 2D map: ground tracks and ascending-node crossings
    ax_map.set_facecolor(BG)
    ax_map.set_title("Ground track at the chosen date: simulated vs real",
                     color=FG, fontsize=10, loc="left", pad=4)
    if texture is not None:
        ax_map.imshow(texture, extent=[-180, 180, -90, 90], aspect="auto", alpha=0.55)
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-90, 90)
    ax_map.set_xticks(range(-180, 181, 60))
    ax_map.set_yticks(range(-90, 91, 30))
    ax_map.grid(color="gray", alpha=0.35, lw=0.5)
    ax_map.axhline(0, color="white", alpha=0.5, lw=0.8)
    ax_map.tick_params(colors=FG, labelsize=8)
    ax_map.set_xlabel("longitude [deg]", color=FG, fontsize=8, labelpad=1)
    ax_map.set_ylabel("latitude [deg]", color=FG, fontsize=8, labelpad=1)
    for s in ax_map.spines.values():
        s.set_color("gray")
    sat_sim = S.sats[1]
    if S.real is not None:
        sim_lon, sim_lat = S.real.simLON, S.real.simLAT
    else:
        sim_lon, sim_lat = w1.LON, w1.LAT
    ax_map.plot(*with_breaks(sim_lon, sim_lat), color=COLORS[1], lw=2.4, label="Simulated orbit")
    ax_map.plot([sat_sim.lon], [sat_sim.lat], marker="*", ls="none", markersize=15,
                color=STAR_COLORS[1], markeredgecolor="white", markeredgewidth=0.6, zorder=8,
                label=f"Simulated satellite: {sat_sim.lat:+.1f}°, {sat_sim.lon:+.1f}°")
    if S.real is not None:
        ax_map.plot(*with_breaks(S.real.LON, S.real.LAT), color=REAL_COLOR, lw=1.4, ls="--",
                    label="Real orbit (TLE)")
        ax_map.plot([S.real.lon], [S.real.lat], marker="*", ls="none", markersize=15,
                    color=REAL_COLOR, markeredgecolor="red", markeredgewidth=1.0, zorder=9,
                    label=f"Real satellite: {S.real.lat:+.1f}°, {S.real.lon:+.1f}°")
    ax_map.legend(loc="lower left", fontsize=7, labelcolor=FG, facecolor=BG,
                  edgecolor="gray", framealpha=0.7)

    # ---------------- plots
    def common(axg, title, unit):
        axg.set_facecolor(BG)
        axg.set_ylabel(unit, color=FG, fontsize=9)
        axg.set_title(title, color=FG, fontsize=10, loc="left", pad=3)
        axg.tick_params(colors=FG, labelsize=8)
        axg.ticklabel_format(axis="y", useOffset=False, style="plain")
        for s in axg.spines.values():
            s.set_color("gray")

    def fast(axg, key, title, unit):
        """Along one orbit (perigee to perigee): orbit 0 and final orbit overlaid."""
        common(axg, title, unit)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("ep", [COLORS[0], COLORS[1]])
        ne = len(S.epochs)
        for j, ep in enumerate(S.epochs):
            c = cmap(j / max(1, ne - 1))
            x, y = ep.XPH, getattr(ep, "X" + key)
            if FAST_X_AXIS == "latitude":
                x = ep.XLAT
            elif FAST_X_AXIS == "longitude":
                x, y = with_breaks(ep.XLON, y)
            if FAST_X_AXIS == "date":
                x = j + 0.5 * (1 - 0.8) + 0.8 * (ep.XPH / ep.XPH[-1])   # one slot per date
            axg.plot(x, y, color=c, lw=1.4, label=f"{ep.date:%d/%m/%Y}")
            axg.plot([x[0]], [y[0]], "o", color=c, markersize=4, markeredgecolor="white",
                     markeredgewidth=0.4, zorder=5)
        if FAST_X_AXIS == "date":
            axg.set_xlim(0, ne)
            axg.set_xticks([j + 0.5 for j in range(ne)])
            axg.set_xticklabels([f"{ep.date:%d/%m}" for ep in S.epochs], rotation=45,
                                ha="right", fontsize=6)
            for j in range(1, ne):
                axg.axvline(j, color="gray", lw=0.4, ls=":")
        if FAST_X_AXIS == "longitude":
            axg.set_xlim(-180, 180)
            axg.set_xticks(range(-180, 181, 60))
        elif FAST_X_AXIS == "phase":
            axg.set_xlim(0, max(1.0, FAST_SPAN_ORBITS))
        lo, hi = axg.get_ylim()
        if lo < 0 < hi:
            axg.axhline(0, color="gray", lw=0.5)

    def slow(axg, y, title, color, unit, span_min=None):
        """x = days since launch, one point per orbit (at the perigee passage)."""
        common(axg, title, unit)
        axg.plot(S.x_days, y, color=color, lw=1.6)
        lo, hi = y.min(), y.max()
        if span_min is not None:
            centre, span = (lo + hi) / 2, max(hi - lo, span_min)
            axg.set_ylim(centre - span * 0.6, centre + span * 0.6)
        if lo < 0 < hi:
            axg.axhline(0, color="gray", lw=0.5)
        axg.set_xlim(0, S.x_days[-1])
        axg.axvline(S.days, color="white", lw=0.9, ls=":")        # chosen date
        axg.plot([S.x_days[0]], [y[0]], "o", color=COLORS[0], markersize=8,
                 markeredgecolor="white", zorder=5)
        axg.plot([S.x_days[S.N]], [y[S.N]], "o", color=COLORS[1], markersize=7,
                 markeredgecolor="white", zorder=5)

    fast(ax_h, "H", "Altitude", "km")
    fast(ax_vr, "VR", "Radial velocity", "km/s")
    fast(ax_vt, "VT", "Tangential velocity", "km/s")
    slow(ax_e, S.E, "Eccentricity", "violet", "e [-]", span_min=1e-4)
    slow(ax_i, S.I, "Inclination", "gold", "deg", span_min=0.005)
    slow(ax_om, S.OM, "Right ascension of ascending node (RAAN)", "tomato", "deg",
         span_min=0.05)
    slow(ax_w, S.W, "Argument of perigee", "mediumspringgreen", "deg", span_min=0.05)
    slow(ax_d, S.SMA, "Semi-major axis (osculating)", "white", "km", span_min=10.0)
    if FAST_X_AXIS != "date":
        ax_h.legend(loc="upper center", ncol=4, fontsize=5.5, labelcolor=FG, facecolor=BG,
                    edgecolor="gray", framealpha=0.7)
    ax_vt.set_xlabel({"date": "date of the orbit shown (one orbit per date, perigee to perigee)",
                      "latitude": "latitude under the satellite [deg] (dot = perigee)",
                      "longitude": "longitude under the satellite [deg] (dot = perigee)",
                      "phase": "fraction of the orbit (perigee to perigee)"}[FAST_X_AXIS],
                     color=FG, fontsize=8)
    for axg in (ax_e, ax_i, ax_om, ax_w, ax_d):
        axg.set_xlabel(f"days since launch ({S.launch:%d/%m/%Y} -> {S.end_date:%d/%m/%Y})",
                       color=FG, fontsize=8)


# ----------------------------------------------------------------
# TOP-RIGHT PANEL: final values of the initial and of the final orbit
# ----------------------------------------------------------------
SUN_RATE = 0.98564736629      # deg/day: advance of the right ascension of the mean Sun
SUN_SYNC_TOL = 0.02           # deg/day: tolerance to call the orbit sun-synchronous


def draw_position_table(axt, x0, y0, fs):
    """Satellite position (geocentric ECI and lat/lon/alt): simulation at launch and at the chosen
    date, and the real satellite (TLE) at the chosen date. Column headings in the star colours."""
    from matplotlib.transforms import offset_copy
    cw = 0.6 * fs                                   # width of one monospace character [pt]
    fg = axt.figure
    cols = [("Simulated", S.sats[1], STAR_COLORS[1])]
    if S.real is not None:
        cols.append(("Real (TLE)", S.real, REAL_COLOR))
    art = [axt.text(x0, y0, "Satellite position", va="top", ha="left",
                    color=FG, family="monospace", fontsize=fs, fontweight="bold",
                    transform=axt.transAxes)]
    for q, (name, _, colr) in enumerate(cols):
        tr = offset_copy(axt.transAxes, fig=fg, x=cw * (10 + 15 * (q + 1)), y=-1.4 * fs,
                         units="points")
        art.append(axt.text(x0, y0, "* " + name, va="top", ha="right", color=colr,
                            family="monospace", fontsize=fs, fontweight="bold", transform=tr))
    fmts = [("Date", lambda z: f"{z.date:%d/%m/%Y}"),
            ("Time UTC", lambda z: f"{z.date:%H:%M:%S}"),
            ("X ECI [km]", lambda z: f"{z.xyz[0]:.1f}"),
            ("Y ECI [km]", lambda z: f"{z.xyz[1]:.1f}"),
            ("Z ECI [km]", lambda z: f"{z.xyz[2]:.1f}"),
            ("Lat [deg]", lambda z: f"{z.lat:+.3f}"),
            ("Lon [deg]", lambda z: f"{z.lon:+.3f}"),
            ("Alt [km]", lambda z: f"{z.alt:.1f}")]
    body = [f"{lbl:10}" + "".join(f"{f(z):>15}" for _, z, _ in cols) for lbl, f in fmts]
    tr = offset_copy(axt.transAxes, fig=fg, y=-2.8 * fs, units="points")
    art.append(axt.text(x0, y0, "\n".join(body), va="top", ha="left", color=FG,
                        family="monospace", fontsize=fs, transform=tr))
    extra, ecol = [], []
    if S.real is not None:
        d = float(np.linalg.norm(S.sats[1].xyz - S.real.xyz))
        extra += [(f"TLE {TLE_NAME}", REAL_COLOR),
                  ("  epoch " + tle_epoch_text(), REAL_COLOR),
                  (f"Sim - real (chosen date): {d:.0f} km", REAL_COLOR),
                  (f"  (date is {S.real.days_from_epoch:+.1f} d from the TLE epoch)", REAL_COLOR)]
    for q, (txt, colr) in enumerate(extra):
        tr2 = offset_copy(axt.transAxes, fig=fg, y=-(2.8 + (len(fmts) + 1 + q) * 1.25) * fs,
                          units="points")
        art.append(axt.text(x0, y0, txt, va="top", ha="left", color=colr,
                            family="monospace", fontsize=fs, transform=tr2))
    return art


pos_artists = []


def tle_epoch_text():
    """Exact date and time of the TLE, e.g. '13/11/2026 18:37:54.049 UTC (day 26317.77632001)'."""
    e = TLE.epoch + dt.timedelta(microseconds=500)           # round to the millisecond
    return f"{e:%d/%m/%Y %H:%M:%S}.{e.microsecond // 1000:03d} UTC (day {TLE_LINE1[18:32].strip()})"


def draw_info():
    w0, w1 = S.win
    rows = [("Inclination i [deg]", "inc", "{:.4f}", "{:+.4f}"),
            ("Eccentricity e [-]", "ecc", "{:.6f}", "{:+.6f}"),
            ("Period T [min]", "period", "{:.3f}", "{:+.3f}"),
            ("Angular momentum h [km^2/s]", "h", "{:.2f}", "{:+.2f}")]
    lines = [f"Launch {S.launch:%d/%m/%Y %H:%M} UTC  ->  chosen date {S.end_date:%d/%m/%Y %H:%M:%S} UTC"
             f"   ({S.days:.2f} d, {S.N + 1} orbits)",
             (f"TLE {TLE_NAME}: epoch {tle_epoch_text()}" if TLE is not None else ""),
             f"{'':30}{'Orbit 0':>15}{'Final orbit':>15}{'Change':>15}",
             f"{'Perigee passage (UTC)':30}{w0.t_perigee.strftime('%d/%m/%y %H:%M'):>15}"
             f"{w1.t_perigee.strftime('%d/%m/%y %H:%M'):>15}"]
    for name, key, f1, f2 in rows:
        a_, b_ = getattr(w0, key), getattr(w1, key)
        lines.append(f"{name:30}{f1.format(a_):>15}{f1.format(b_):>15}{f2.format(b_ - a_):>15}")
    # node shift versus the apparent motion of the mean Sun (sun-synchronous check)
    t_days = S.N * S.Ta / 86400.0                  # time between orbit 0 and the final orbit
    rate = S.d_raan_acc / t_days                   # node rate [deg/day]
    sun_adv = SUN_RATE * t_days                    # advance of the mean Sun in the same time
    drift_min = (S.d_raan_acc - sun_adv) / 15.0 * 60.0     # change of the node local time [min]
    sync = abs(rate - SUN_RATE) <= SUN_SYNC_TOL
    lines += ["",
              f"{'Node shift  dOmega  [deg]':30}{S.d_raan_acc:>+15.3f}"
              f"   (angle {S.d_raan:+.3f})",
              f"{'Mean Sun advance  [deg]':30}{sun_adv:>+15.3f}",
              f"{'Node rate  [deg/day]':30}{rate:>+15.5f}   (Sun: {SUN_RATE:+.5f})",
              f"{'Node local time drift [min]':30}{drift_min:>+15.1f}",
              ("=> SUN-SYNCHRONOUS: the node follows the Sun"
               if sync else
               f"=> NOT sun-synchronous: the node moves {rate - SUN_RATE:+.3f} deg/day "
               f"relative to the Sun")]
    info_text.set_text("\n".join(lines))


# ----------------------------------------------------------------
# CONTROLS: date box, Compute, Final 3D
# ----------------------------------------------------------------
proposed_date = (launch_dt + dt.timedelta(days=DEFAULT_DAYS)).strftime("%d/%m/%Y %H:%M:%S")

ax_tb = fig.add_axes([0.215, 0.045, 0.115, 0.035])
date_box = TextBox(ax_tb, "Final date UTC (DD/MM/YYYY HH:MM:SS)  ", initial=proposed_date,
                   color="#222222", hovercolor="#333333")
date_box.label.set_color(FG)
date_box.label.set_fontsize(9)
date_box.text_disp.set_color(FG)

ax_calc = fig.add_axes([0.34, 0.045, 0.05, 0.035])
btn_calc = Button(ax_calc, "Compute", color="#333333", hovercolor="#555555")
btn_calc.label.set_color(FG)

ax_3df = fig.add_axes([0.405, 0.045, 0.09, 0.035])
btn_3df = Button(ax_3df, "Final 3D", color="#333333", hovercolor="#555555")
btn_3df.label.set_color(FG)

status_msg = fig.text(0.51, 0.058, "", color="yellow", fontsize=9)


def open_final_3d(_event=None):
    """Open a NEW window with the 3D scene: rotate it with the mouse as much as you like,
    zoom with the wheel and save with the disk icon."""
    if S is None:
        return
    f3 = plt.figure(figsize=(9, 9), facecolor=BG)
    f3.canvas.manager.set_window_title("Orbit - 3D (rotate with the mouse)")
    a3 = f3.add_subplot(111, projection="3d", facecolor=BG, computed_zorder=not ORBITS_ON_TOP)
    build_3d(a3)
    a3.set_box_aspect((1, 1, 1), zoom=1.15)
    f3.suptitle(f"Simulated and real orbit, chosen date {S.end_date:%d/%m/%Y %H:%M:%S} UTC",
                color=FG, fontsize=11)
    f3.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
    f3.show()


def export_pdf(path=None):
    """A4 PDF with the 2D graphs: ground track over the full width, the 8 plots below."""
    global ax_map, ax_h, ax_vr, ax_vt, ax_e, ax_i, ax_om, ax_w, ax_d
    path = Path(path or PDF_FILE)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    saved = (ax_map, ax_h, ax_vr, ax_vt, ax_e, ax_i, ax_om, ax_w, ax_d)
    fp = plt.figure(figsize=(8.27, 11.69), facecolor=BG)          # A4 portrait
    gs = GridSpec(5, 2, figure=fp, height_ratios=[2.6, 1, 1, 1, 1], left=0.09, right=0.97,
                  top=0.735, bottom=0.06, hspace=0.85, wspace=0.28)
    axt = fp.add_axes([0.09, 0.775, 0.88, 0.18], facecolor=BG)
    axt.set_axis_off()
    ax_map = fp.add_subplot(gs[0, :])
    ax_h, ax_i = fp.add_subplot(gs[1, 0]), fp.add_subplot(gs[1, 1])
    ax_vr, ax_om = fp.add_subplot(gs[2, 0]), fp.add_subplot(gs[2, 1])
    ax_vt, ax_w = fp.add_subplot(gs[3, 0]), fp.add_subplot(gs[3, 1])
    ax_e, ax_d = fp.add_subplot(gs[4, 0]), fp.add_subplot(gs[4, 1])
    try:
        draw_2d()
        draw_position_table(axt, 0.0, 1.0, 7.5)
        fp.suptitle(f"J2 orbit: launch {S.launch:%d/%m/%Y} -> {S.end_date:%d/%m/%Y %H:%M:%S} UTC "
                    f"({S.days:.1f} days, {S.N + 1} orbits)", color=FG, fontsize=11, y=0.975)
        try:
            fp.savefig(path, facecolor=BG)
        except PermissionError:                       # file open in a PDF viewer
            path = path.with_name(f"{path.stem}_{dt.datetime.now():%H%M%S}{path.suffix}")
            fp.savefig(path, facecolor=BG)
    finally:
        ax_map, ax_h, ax_vr, ax_vt, ax_e, ax_i, ax_om, ax_w, ax_d = saved
        plt.close(fp)
    return path


def recompute(_=None):
    """Read the date from the box, re-propagate and redraw."""
    global S
    date_text = date_box.text.strip()
    try:
        date = None
        for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
            try:
                date = dt.datetime.strptime(date_text, fmt)
                break
            except ValueError:
                pass
        if date is None:
            raise ValueError
    except ValueError:
        status_msg.set_text("Invalid format: write DD/MM/YYYY HH:MM:SS (e.g. 25/12/2026 10:30:00)")
        fig.canvas.draw_idle()
        return
    days = (date - launch_dt).total_seconds() / 86400.0
    if days < 1:
        status_msg.set_text(f"The date must be at least 1 day after launch "
                            f"({launch_dt:%d/%m/%Y})")
        fig.canvas.draw_idle()
        return
    if days > MAX_DAYS:
        status_msg.set_text(f"Date too far: maximum {MAX_DAYS} days after launch "
                            f"(change MAX_DAYS in the code)")
        fig.canvas.draw_idle()
        return
    status_msg.set_text(f"Computing up to {date_text} ...")
    fig.canvas.draw()
    fig.canvas.flush_events()
    S = compute(days, launch_dt)
    draw()
    draw_info()
    msg = f"Done: {S.N + 1} orbits from launch to {date_text}"
    if PDF_AUTO:
        msg += f"  |  PDF saved: {export_pdf().name}"
    status_msg.set_text(msg)
    fig.canvas.draw_idle()


btn_3df.on_clicked(open_final_3d)
btn_calc.on_clicked(recompute)
date_box.on_submit(recompute)

recompute()

if __name__ == "__main__":
    plt.show()
