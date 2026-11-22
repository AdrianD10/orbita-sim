"""
astro_utils.py - small analytical toolkit used to cross-check the simulation.

Pure functions (no plotting, no global state) for impulsive manoeuvres, plane geometry,
local solar time of the node and TLE parsing. Units: km, s, km/s, degrees at the interface.
Every formula is verified against an independent vector computation in tests/.
"""
import datetime as dt
import numpy as np

MU = 398600.4418        # km^3/s^2
R_EARTH = 6378.137      # km


# ---------------------------------------------------------------- orbit geometry
def h_hat(inc_deg, raan_deg):
    """Unit normal of an orbital plane (direction of the angular momentum), ECI."""
    i, o = np.radians(inc_deg), np.radians(raan_deg)
    return np.array([np.sin(i) * np.sin(o), -np.sin(i) * np.cos(o), np.cos(i)])


def plane_angle(i1, raan1, i2, raan2):
    """Angle between two orbital planes [deg]: cos = cos i1 cos i2 + sin i1 sin i2 cos(dRAAN)."""
    i1, i2, d = np.radians(i1), np.radians(i2), np.radians(raan2 - raan1)
    c = np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(d)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def circular_speed(r):
    """Circular speed [km/s] at radius r [km]."""
    return float(np.sqrt(MU / r))


# ---------------------------------------------------------------- impulsive manoeuvres
def plane_change_dv(v, dtheta_deg):
    """Pure plane change at constant speed v: dv = 2 v sin(dtheta/2)."""
    return 2.0 * v * np.sin(np.radians(dtheta_deg) / 2.0)


def general_dv(v1, v2, gamma1_deg, gamma2_deg, dtheta_deg):
    """Delta-v between two velocities at the same point on the line of nodes of the two planes.

    gamma = flight path angle of each orbit, dtheta = angle between the planes.
    dv^2 = v1^2 + v2^2 - 2 v1 v2 (cos g1 cos g2 cos dtheta + sin g1 sin g2)
    """
    g1, g2, dth = np.radians(gamma1_deg), np.radians(gamma2_deg), np.radians(dtheta_deg)
    cos_alpha = np.cos(g1) * np.cos(g2) * np.cos(dth) + np.sin(g1) * np.sin(g2)
    return float(np.sqrt(max(v1 ** 2 + v2 ** 2 - 2 * v1 * v2 * cos_alpha, 0.0)))


def gauss_normal_impulse(r, h, inc_deg, u_deg, dvn):
    """Small normal impulse dvn [km/s] on a circular orbit -> (d_inc, d_raan) in degrees (Gauss)."""
    u, i = np.radians(u_deg), np.radians(inc_deg)
    d_inc = r * np.cos(u) / h * dvn
    d_raan = r * np.sin(u) / (h * np.sin(i)) * dvn
    return float(np.degrees(d_inc)), float(np.degrees(d_raan))


def hohmann_plane_change(r1, r2, d_inc_deg, beta1_deg=0.0):
    """Hohmann transfer between circular orbits with a plane change d_inc split in two burns.

    beta1 at the first burn (perigee), the rest at the second (apogee). Returns total dv [km/s].
    """
    a_t = 0.5 * (r1 + r2)
    v1, v2 = circular_speed(r1), circular_speed(r2)
    vp = np.sqrt(MU * (2 / r1 - 1 / a_t))
    va = np.sqrt(MU * (2 / r2 - 1 / a_t))
    b1 = np.radians(beta1_deg)
    b2 = np.radians(d_inc_deg) - b1
    return float(np.sqrt(v1 ** 2 + vp ** 2 - 2 * v1 * vp * np.cos(b1))
                 + np.sqrt(va ** 2 + v2 ** 2 - 2 * va * v2 * np.cos(b2)))


def optimal_split(r1, r2, d_inc_deg, n=4001):
    """Plane-change split (deg at the first burn) that minimises the total dv."""
    betas = np.linspace(0.0, d_inc_deg, n)
    tot = [hohmann_plane_change(r1, r2, d_inc_deg, b) for b in betas]
    k = int(np.argmin(tot))
    return float(betas[k]), float(tot[k])


# ---------------------------------------------------------------- local solar time
def local_time_of_node(raan_deg, sun_ra_deg, descending=True):
    """Local mean solar time [h] of the node: LTAN = 12 + (RAAN - RA_sun)/15, LTDN = LTAN + 12."""
    ltan = (12.0 + (raan_deg - sun_ra_deg) / 15.0) % 24.0
    return (ltan + 12.0) % 24.0 if descending else ltan


# ---------------------------------------------------------------- TLE
def tle_checksum_ok(line):
    """Modulo-10 checksum of a TLE line (digits count, '-' counts 1)."""
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return total % 10 == int(line[68])


def parse_tle(line1, line2):
    """Epoch and mean elements of a TLE as a dict (angles in degrees, n in rev/day)."""
    year = 2000 + int(line1[18:20])
    epoch = dt.datetime(year, 1, 1) + dt.timedelta(days=float(line1[20:32]) - 1.0)
    return dict(epoch=epoch, inc=float(line2[8:16]), raan=float(line2[17:25]),
                ecc=float("0." + line2[26:33].strip()), argp=float(line2[34:42]),
                mean_anomaly=float(line2[43:51]), n_rev_day=float(line2[52:63]))


def parse_utc(text):
    """'DD/MM/YYYY HH:MM:SS' (optionally with decimals of a second, or without the time) -> datetime."""
    text = text.strip()
    for fmt in ("%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"date not understood: {text!r} (use DD/MM/YYYY HH:MM:SS)")


def tle_epoch_field(epoch):
    """datetime (UTC) -> TLE epoch field 'YYDDD.DDDDDDDD' (day of the year, 1 January = 1.0)."""
    day = (epoch - dt.datetime(epoch.year, 1, 1)).total_seconds() / 86400.0 + 1.0
    return f"{epoch.year % 100:02d}{day:012.8f}"


def tle_set_epoch(line1, epoch):
    """Return line 1 of a TLE with the epoch replaced by `epoch` (datetime) and the checksum redone."""
    body = line1[:18] + tle_epoch_field(epoch) + line1[32:68]
    check = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in body) % 10
    return body + str(check)
