"""Uniform plasma relaxing to a Maxwellian through background-gas Monte Carlo collisions.
Argon ions start with a flat (box-shaped) velocity distribution and collide elastically with a
300 K argon gas. With sigma ~ 1/g the collision rate nu is speed independent, so the temperature
relaxes exactly as T(t) = T_gas + (T_0 - T_gas) exp(-nu t / 2) while the distribution becomes Maxwellian.
Every curve is the mean over SEEDS independent runs, with a shaded +/- 1 standard deviation band."""
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from PyPIC3D.collisions import apply_collisions
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles

jax.config.update("jax_enable_x64", True)

# ----------------------------- input parameters -----------------------------
N = 100_000                     # macro-ions
T_GAS, T_ION = 300.0, 1000.0    # neutral gas and initial ion temperature [K]
N_GAS, NU = 1.0e20, 1.0e6       # gas density [m^-3], ion-neutral collision rate [1/s]
STEPS, T_END = 400, 10.0        # steps, end time in units of 1 / nu
SNAPSHOTS = (0.0, 0.5, 2.0, 10.0)  # times (1 / nu) for the distribution panel
SEEDS = range(5)                # independent random seeds
BINS = np.linspace(-4.5, 4.5, 121)
OUTPUT_DIR = "outputs/mcc_maxwellian"

E, KB, M_AR = 1.602176634e-19, 1.380649e-23, 6.6335209e-26
VTH = np.sqrt(KB * T_GAS / M_AR)


def run(seed):
    """One relaxation run; returns temperature, excess kurtosis and v_x histograms at SNAPSHOTS."""
    energy = np.logspace(-6, 3, 400)
    sigma = NU / (N_GAS * np.sqrt(2 * energy * E / (M_AR / 2)))  # sigma ~ 1/g: constant collision rate
    specs = (("mcc", 0, N_GAS, T_GAS, M_AR, (("elastic", 0.0, tuple(energy), tuple(sigma), -1, -1),)),)
    dt = T_END / NU / STEPS
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    u0 = np.sqrt(3 * KB * T_ION / M_AR) * jax.random.uniform(k1, (N, 3), minval=-1, maxval=1)
    particles = TiledParticles(x=jax.random.uniform(k2, (1, 1, 1, 1, N, 3)), u=u0[None, None, None, None],
                               active=jnp.ones((1, 1, 1, 1, N), bool), key=k3)
    config = SpeciesConfig(charge=jnp.array([E]), mass=jnp.array([M_AR]), weight=jnp.array([1.0]), update_x=jnp.zeros((1, 3), bool))
    world = SimpleNamespace(dx=1.0, dy=1.0, dz=1.0, dt=dt, eps=8.8541878128e-12, kb=KB)
    step = jax.jit(lambda p: apply_collisions(p, config, SimpleNamespace(collisions=specs), world)[0])
    temps, kurt, hists = [], [], {}
    for i in range(STEPS + 1):
        v = np.asarray(particles.u[0, 0, 0, 0])
        v = v - v.mean(0)
        temps.append(M_AR * np.mean(v**2) / KB)
        kurt.append(np.mean(v[:, 0] ** 4) / np.mean(v[:, 0] ** 2) ** 2 - 3)
        for s in SNAPSHOTS:
            if abs(i * dt * NU - s) < 0.5 * dt * NU:
                hists[s] = np.histogram(v[:, 0] / VTH, bins=BINS, density=True)[0]
        if i % (STEPS // 5) == 0:
            print(f"  seed {seed} step {i:4d}/{STEPS}: T = {temps[-1]:7.1f} K, excess kurtosis = {kurt[-1]:+.3f}", flush=True)
        if i < STEPS:
            particles = step(particles)
    return np.array(temps), np.array(kurt), np.array([hists[s] for s in SNAPSHOTS])


# ----------------------------- run the simulations -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Relaxing {N:,} argon ions from a {T_ION:g} K box distribution in {T_GAS:g} K argon, {len(SEEDS)} seeds")
temps, kurt, hists = map(np.array, zip(*(run(seed) for seed in SEEDS)))

# ----------------------------- plot, save, and print -----------------------------
times = np.arange(STEPS + 1) * T_END / STEPS
theory = T_GAS + (T_ION - T_GAS) * np.exp(-0.5 * times)
centers = 0.5 * (BINS[1:] + BINS[:-1])
fig, ax = plt.subplots(3, 1, figsize=(8, 10))
for color, s, h in zip(plt.cm.Blues(np.linspace(0.4, 0.95, len(SNAPSHOTS))), SNAPSHOTS, hists.transpose(1, 0, 2)):
    ax[0].fill_between(centers, h.mean(0) - h.std(0), h.mean(0) + h.std(0), step="mid", color=color, alpha=0.3, lw=0)
    ax[0].step(centers, h.mean(0), where="mid", lw=2, color=color, label=rf"$\nu t$ = {s:g}")
grid = np.linspace(-4.5, 4.5, 300)
ax[0].plot(grid, np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.5, label=f"Maxwellian at {T_GAS:g} K")
ax[0].set(xlabel=r"$v_x / \sqrt{k_B T_{gas} / m}$", ylabel="probability density",
          title=rf"Relaxation to a Maxwellian (background MCC, mean $\pm 1\sigma$ of {len(SEEDS)} seeds)")
for a, data, ref, ylabel in ((ax[1], kurt, np.zeros_like(times), r"excess kurtosis of $v_x$"), (ax[2], temps, theory, "ion temperature [K]")):
    a.fill_between(times, data.mean(0) - data.std(0), data.mean(0) + data.std(0), color="tab:blue", alpha=0.25, lw=0)
    a.plot(times, data.mean(0), color="tab:blue", lw=2, label="PyPIC3D")
    a.plot(times, ref, "k--", lw=1.5, label="Maxwellian" if a is ax[1] else r"theory $T_g + (T_0 - T_g)\,e^{-\nu t/2}$")
    a.set(ylabel=ylabel)
ax[2].set(xlabel=r"$\nu t$")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "mcc_maxwellian.png"), dpi=300)
err = np.abs(temps - theory).max(1)
print(f"Temperature vs theory: max |T - T_theory| = {err.mean():.1f} +/- {err.std():.1f} K; final T = {temps[:, -1].mean():.1f} "
      f"+/- {temps[:, -1].std():.1f} K (theory {theory[-1]:.1f} K); final excess kurtosis = {kurt[:, -1].mean():+.3f} +/- {kurt[:, -1].std():.3f}")
print(f"Saved {OUTPUT_DIR}/mcc_maxwellian.png")
