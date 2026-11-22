# Fit the simulated orbit to the real TLE of orbit_static.py (TLE_LINE1 / TLE_LINE2).
# It finds the three parameters that make the simulation pass through the real satellite
# at the TLE epoch:  altitude (-> orbital period), start argument of latitude (-> phase along
# the orbit) and RAAN at launch (-> orbital plane).   Run:  python fit_tle.py
# Then copy the printed block into the parameters of orbit_static.py.
import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.integrate import solve_ivp
import orbit_static as m

MU, R, J2 = m.MU, m.R_EARTH, m.J2
T = m.TLE
nK = T.n_rev * 2 * np.pi / 86400.0
a_K = (MU / nK ** 2) ** (1 / 3)
p = a_K * (1 - T.ecc ** 2)
wdot = 0.75 * nK * J2 * (R / p) ** 2 * (4 - 5 * np.sin(np.radians(T.inc)) ** 2)
rate_t = nK + wdot                                  # mean rate of the argument of latitude [rad/s]
u_target = (T.argp + T.M) % 360
t_ep = (T.epoch - m.launch_dt).total_seconds()
tt = np.linspace(0, t_ep, 60000)


def u_of(P):
    h = np.cross(P[:3], P[3:], axis=0)
    hn = h / np.linalg.norm(h, axis=0)
    nn = np.array([-hn[1], hn[0], np.zeros_like(hn[0])])
    nn = nn / np.linalg.norm(nn, axis=0)
    return np.arctan2(np.sum(np.cross(nn, P[:3], axis=0) * hn, axis=0), np.sum(nn * P[:3], axis=0))


def setup(alt, u0, raan):
    r0, v0, a, e = m.initial_state(R + alt - 0.5, R + alt + 0.5, T.inc, raan, u0)
    m.s0 = np.concatenate((r0, v0)); m.a = a; m.e = e; m.T = 2 * np.pi * np.sqrt(a ** 3 / MU)


def measure(alt, u0, raan):
    setup(alt, u0, raan)
    sol = solve_ivp(m.equations, [0, t_ep], m.s0, args=(m.USE_J2,), method="DOP853",
                    dense_output=True, rtol=1e-10, atol=1e-10)
    uu = np.unwrap(u_of(sol.sol(tt)))
    return (uu[-1] - uu[0]) / t_ep, np.degrees(uu[-1]) % 360


alt, u0 = a_K - R + 6.0, 224.5
raan = m.raan_from_ltdn(m.launch_dt, 10.0)
for outer in range(5):
    x0, x1 = alt, alt + 1.0
    f0 = measure(x0, u0, raan)[0] - rate_t
    for _ in range(4):
        f1 = measure(x1, u0, raan)[0] - rate_t
        if abs(f1) < 1e-12:
            break
        x0, f0, x1 = x1, f1, x1 - f1 * (x1 - x0) / (f1 - f0)
    alt = x1
    _, ue = measure(alt, u0, raan)
    du = (ue - u_target + 180) % 360 - 180
    u0 = (u0 - du) % 360
    print(f"iteration {outer}: altitude {alt:.3f} km, phase error {du:+.4f} deg")
for _ in range(2):                                   # RAAN: compare with the simulated node at the epoch
    setup(alt, u0, raan)
    S = m.compute(t_ep / 86400.0, m.launch_dt)
    raan = (raan - ((S.win[1].raan - T.raan + 180) % 360 - 180)) % 360
sun = (m.gmst_deg(m.launch_dt) - 15 * (m.launch_dt.hour + m.launch_dt.minute / 60) + 180) % 360
ltdn = (((12 + (raan - sun) / 15) % 24) + 12) % 24
print("\nCopy into orbit_static.py:\n")
print(f"PERIGEE_ALT = {alt - 0.5:.3f}")
print(f"APOGEE_ALT = {alt + 0.5:.3f}")
print(f"INCLINATION = {T.inc}")
print(f"LTDN = {ltdn:.4f}")
print(f"ARG_PERIGEE = {u0:.3f}")
print("LAUNCH_SITE = None")
print(f"DEFAULT_DAYS = {t_ep / 86400:.7f}    # = TLE epoch")
