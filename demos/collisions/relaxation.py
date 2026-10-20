"""Collision relaxation benchmarks against theory: Maxwellianization, thermalization,
temperature equilibration (Takizuka-Abe binary collisions), and relaxation to a
neutral bath (background MCC). All particles share one cell (0-D plasma)."""
import os
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from PyPIC3D.collisions import apply_collisions
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles

jax.config.update("jax_enable_x64", True)

# ----------------------------- input parameters -----------------------------
N = 20_000                      # macro-particles per species
DENSITY = 1.0e26                # physical density of each species [m^-3]
COULOMB_LOG = 10.0
SCATTER = 0.01                  # binary-collision dt: scattering parameter s = <1 - cos> at thermal speed
MCC_STEPS = 400                 # MCC steps
T_HOT, T_COLD = 120.0, 40.0     # thermalization: electron / positron temperature [eV]
T_E, T_I, MASS_RATIO = 100.0, 20.0, 25.0  # equilibration: electron / ion temperature [eV], m_i / m_e
T_CUBE = 50.0                   # Maxwellianization: temperature of the uniform velocity cube [eV]
T_GAS, T_ION, NU_MCC = 300.0, 3000.0, 1.0e6  # MCC: bath [K], initial argon ions [K], collision rate [1/s]
OUTPUT_DIR = "outputs/collisions"
SEED = 0

E, ME, KB, EPS, M_AR = 1.602176634e-19, 9.1093837e-31, 1.380649e-23, 8.8541878128e-12, 6.6335209e-26


def spitzer(qa, qb, ma, mb, ta, tb, nb):
    """Energy exchange rate nu_ab in dT_a/dt = nu_ab (T_b - T_a); temperatures in J."""
    return 2 * qa**2 * qb**2 * nb * COULOMB_LOG / (3 * (2 * np.pi) ** 1.5 * EPS**2 * ma * mb * (ta / ma + tb / mb) ** 1.5)


def two_temperature_theory(q, m, t0, time_axis):
    """Integrate the coupled Spitzer equations for two species."""
    t, out = np.array(t0, float), []
    dt = np.diff(time_axis, prepend=0.0)
    for h in dt:
        for _ in range(20):
            rate = spitzer(q[0], q[1], m[0], m[1], t[0], t[1], DENSITY)
            t = t + h / 20 * rate * np.array([t[1] - t[0], t[0] - t[1]])
        out.append(t.copy())
    return np.array(out)


def collision_dt(q, m, t):
    """Largest dt keeping every species pair's scattering parameter at thermal speed below SCATTER."""
    pairs = [(a, b) for a in range(len(m)) for b in range(a, len(m))]
    mu = lambda a, b: m[a] * m[b] / (m[a] + m[b])
    g3 = lambda a, b: (3 * (t[a] / m[a] + t[b] / m[b])) ** 1.5
    return min(SCATTER * 4 * np.pi * EPS**2 * mu(a, b) ** 2 * g3(a, b) / (q[a] ** 2 * q[b] ** 2 * DENSITY * COULOMB_LOG) for a, b in pairs)


def run(name, mass, charge, temperature, specs, t_end, dt, cube=False):
    """Advance a 0-D plasma with the given collision operators and record moments."""
    n_s, steps = len(mass), int(np.ceil(t_end / dt))
    keys = jax.random.split(jax.random.PRNGKey(SEED), 2 * n_s + 1)
    if cube:
        u = [jnp.sqrt(3 * t / m) * jax.random.uniform(k, (N, 3), minval=-1, maxval=1) for k, t, m in zip(keys, temperature, mass)]
    else:
        u = [jnp.sqrt(t / m) * jax.random.normal(k, (N, 3)) for k, t, m in zip(keys, temperature, mass)]
    particles = TiledParticles(
        x=jax.random.uniform(keys[-1], (1, 1, 1, n_s, N, 3)), u=jnp.stack(u)[None, None, None],
        active=jnp.ones((1, 1, 1, n_s, N), bool), key=keys[n_s])
    config = SpeciesConfig(charge=jnp.asarray(charge), mass=jnp.asarray(mass), weight=jnp.full(n_s, DENSITY / N),
                           update_x=jnp.zeros((n_s, 3), bool))
    world = SimpleNamespace(dx=1.0, dy=1.0, dz=1.0, dt=dt, eps=EPS, kb=KB)
    step = jax.jit(lambda p: apply_collisions(p, config, SimpleNamespace(collisions=specs), world)[0])
    temps, kurt, start = [], [], time.perf_counter()
    for i in range(steps + 1):
        v = np.asarray(particles.u[0, 0, 0])
        v = v - v.mean(axis=1, keepdims=True)
        temps.append(np.asarray(mass) * np.mean(np.sum(v**2, -1), -1) / 3)
        kurt.append(np.mean(v[..., 0] ** 4, -1) / np.mean(v[..., 0] ** 2, -1) ** 2 - 3)
        if i < steps:
            particles = step(particles)
        if i % max(steps // 5, 1) == 0:
            print(f"  {name}: step {i:5d}/{steps}  T = {np.round(temps[-1] / E, 2)} eV  ({time.perf_counter() - start:.1f} s)", flush=True)
    return np.arange(steps + 1) * dt, np.array(temps), np.array(kurt), particles


# ----------------------------- run the simulations -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"JAX devices: {jax.devices()}")
results = {}

q, m = (-E, E), (ME, ME)
t0 = (T_HOT * E, T_COLD * E)
nu = 2 * spitzer(q[0], q[1], m[0], m[1], *t0, DENSITY)
print("Thermalization (electron/positron, all binary pairs)")
results["thermalization"] = run("thermalization", m, q, t0, (("binary", 0, 0, COULOMB_LOG), ("binary", 1, 1, COULOMB_LOG),
                                ("binary", 0, 1, COULOMB_LOG)), 2.0 / nu, collision_dt(q, m, t0))

q, m = (-E, E), (ME, MASS_RATIO * ME)
t0 = (T_E * E, T_I * E)
nu = 2 * spitzer(q[0], q[1], m[0], m[1], *t0, DENSITY)
print(f"Equilibration (electron/ion, m_i/m_e = {MASS_RATIO:g})")
results["equilibration"] = run("equilibration", m, q, t0, (("binary", 0, 0, COULOMB_LOG), ("binary", 1, 1, COULOMB_LOG),
                               ("binary", 0, 1, COULOMB_LOG)), 2.0 / nu, collision_dt(q, m, t0))

nu_ee = spitzer(-E, -E, ME, ME, T_CUBE * E, T_CUBE * E, DENSITY)
print("Maxwellianization (electron self-collisions from a uniform velocity cube)")
results["maxwellianization"] = run("maxwellianization", (ME,), (-E,), (T_CUBE * E,), (("binary", 0, 0, COULOMB_LOG),),
                                   4.0 / nu_ee, collision_dt((-E,), (ME,), (T_CUBE * E,)), cube=True)

n_gas = 1.0e20
energy = np.logspace(-6, 3, 400)
sigma = NU_MCC / (n_gas * np.sqrt(2 * energy * E / (M_AR / 2)))
# sigma ~ 1/g gives a velocity-independent collision rate (Maxwell molecules)
mcc = (("mcc", 0, n_gas, T_GAS, M_AR, (("elastic", 0.0, tuple(energy), tuple(sigma), -1, -1),)),)
print("Background MCC (argon ions relaxing in a 300 K argon bath)")
results["mcc"] = run("mcc", (M_AR,), (E,), (KB * T_ION,), mcc, 10.0 / NU_MCC, 10.0 / NU_MCC / MCC_STEPS, cube=True)

# ----------------------------- plot, save, and print -----------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
summary = []
for panel, (case, labels, mass) in zip(ax.flat[:2], [("thermalization", ("electrons", "positrons"), (ME, ME)),
                                                       ("equilibration", ("electrons", "ions"), (ME, MASS_RATIO * ME))]):
    t, temps, _, _ = results[case]
    theory = two_temperature_theory((-E, E), mass, temps[0], t)
    for s, label in enumerate(labels):
        panel.plot(t * 1e12, temps[:, s] / E, lw=2, label=f"{label} (PyPIC3D)")
        panel.plot(t * 1e12, theory[:, s] / E, "k--", lw=1.2, label="Spitzer (Maxwellian) theory" if s == 0 else None)
    panel.set(title=case.capitalize(), xlabel="time [ps]", ylabel="temperature [eV]")
    panel.legend()
    error = np.max(np.abs(temps - theory)) / np.abs(temps[0, 0] - temps[0, 1])
    summary.append(f"{case:18s} max |T - T_Spitzer| / dT0 = {100 * error:.2f}%")

t, _, kurt, particles = results["maxwellianization"]
ax[1, 0].plot(t * nu_ee, kurt[:, 0], lw=2, label="PyPIC3D")
ax[1, 0].axhline(0.0, color="k", ls="--", label="Maxwellian")
ax[1, 0].set(title="Maxwellianization", xlabel=r"$\nu_{ee} t$", ylabel=r"excess kurtosis of $v_x$")
inset = ax[1, 0].inset_axes([0.45, 0.15, 0.5, 0.45])
v = np.asarray(particles.u[0, 0, 0, 0, :, 0]) / np.sqrt(T_CUBE * E / ME)
inset.hist(v, bins=80, density=True, alpha=0.6)
grid = np.linspace(-4, 4, 200)
inset.plot(grid, np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi), "k--")
inset.set_xticks([]), inset.set_yticks([])
ax[1, 0].legend(loc="upper left")
summary.append(f"{'maxwellianization':18s} final excess kurtosis = {kurt[-1, 0]:+.3f} (Maxwellian: 0, initial: {kurt[0, 0]:+.2f})")

t, temps, _, _ = results["mcc"]
theory = T_GAS + (T_ION - T_GAS) * np.exp(-0.5 * NU_MCC * t)
ax[1, 1].plot(t * 1e6, temps[:, 0] / KB, lw=2, label="PyPIC3D")
ax[1, 1].plot(t * 1e6, theory, "k--", label=r"$T_g + (T_0 - T_g)e^{-\nu t/2}$")
ax[1, 1].set(title="MCC relaxation to neutral bath", xlabel=r"time [$\mu$s]", ylabel="ion temperature [K]")
ax[1, 1].legend()
summary.append(f"{'mcc':18s} max |T - T_theory| / dT0 = {100 * np.max(np.abs(temps[:, 0] / KB - theory)) / (T_ION - T_GAS):.2f}%")

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/relaxation.png", dpi=200)
np.savez(f"{OUTPUT_DIR}/relaxation.npz", **{f"{k}_{n}": v for k, r in results.items() for n, v in zip(("t", "T", "kurtosis"), r[:3])})
print("\n".join(summary))
print(f"Saved {OUTPUT_DIR}/relaxation.png and relaxation.npz")
