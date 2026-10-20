"""Weakly ionized two-stream instability: full PyPIC3D runs with and without
electron-neutral Monte Carlo collisions (elastic + excitation) in a cold argon gas."""
import os
import shutil

import jax
import matplotlib.pyplot as plt
import numpy as np

from PyPIC3D.__main__ import run_PyPIC3D
import toml

jax.config.update("jax_enable_x64", True)

# ----------------------------- input parameters -----------------------------
NX = 100                        # grid cells
N_BEAM = 1500                   # macro-electrons per beam
V_BEAM = 0.5e8                  # beam drift speeds +/- V_BEAM [m/s]
VTH_E = 1.5e7                   # electron thermal speed [m/s]
T_END = 5.413e-8                # simulated time [s]
GAS_DENSITY = 1.5e20            # argon density [m^-3]
GAS_TEMPERATURE = 300.0         # [K]
SIGMA_ELASTIC = 1.0e-19         # elastic cross section [m^2] (nu ~ n sigma v ~ 7.5e8 1/s)
SIGMA_EXCITATION = 2.0e-20      # excitation cross section above threshold [m^2]
EXCITATION_ENERGY = 11.5        # argon excitation threshold [eV]
OUTPUT_DIR = "outputs/two_stream_mcc"

M_AR = 6.6335209e-26
BEAM = """
[particle{i}]
name = "{key}"
N_particles = {n}
number_density = 1.0e15
charge = -1.602e-19
mass = 9.1093837e-31
vth = {vth}
initial_vx = {v}
"""
COLLISIONS = """
[collisions.{key}_argon]
type = "background_mcc"
species = "{key}"
background_density = {n_gas}
background_temperature = {t_gas}
background_mass = {m_gas}
[collisions.{key}_argon.elastic]
cross_section = {sigma}
[collisions.{key}_argon.excitation]
energy = {e_exc}
cross_section = [[0.0, 0.0], [{e_exc}, 0.0], [{e_exc_plus}, {sigma_exc}], [1.0e4, {sigma_exc}]]
"""


def write_config(name, collisions):
    out = os.path.join(OUTPUT_DIR, name)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    text = f"""[simulation_parameters]
name = "{name}"
t_wind = {T_END}
solver = "electrodynamic_yee"
Nx = {NX}
Ny = 1
Nz = 1
x_wind = 1
y_wind = 1
z_wind = 1
output_dir = "{out}"
cfl = 0.99
shape_factor = 2
current_calculation = "j_from_rhov"
relativistic = true

[plotting]
plotting_interval = 4
plot_openpmd_particles = false
plot_openpmd_fields = false
"""
    for i, (key, v) in enumerate((("electron1", V_BEAM), ("electron2", -V_BEAM)), start=1):
        text += BEAM.format(i=i, key=key, n=N_BEAM, vth=VTH_E, v=v)
        if collisions:
            text += COLLISIONS.format(key=key, n_gas=GAS_DENSITY, t_gas=GAS_TEMPERATURE, m_gas=M_AR, sigma=SIGMA_ELASTIC,
                                      e_exc=EXCITATION_ENERGY, e_exc_plus=EXCITATION_ENERGY * 1.001, sigma_exc=SIGMA_EXCITATION)
    text += f"""
[particle3]
name = "ions"
N_particles = {2 * N_BEAM}
charge = 1.602e-19
mass = 1.67e-27
vth = 3.5e4
"""
    path = os.path.join(out, f"{name}.toml")
    with open(path, "w") as handle:
        handle.write(text)
    return path, out


# ----------------------------- run the simulations -----------------------------
runs = {}
for name, collisions in (("collisionless", False), ("mcc", True)):
    print(f"\n=== Two-stream run: {name} ===", flush=True)
    path, out = write_config(name, collisions)
    run_PyPIC3D(toml.load(path))
    load = lambda f: np.loadtxt(os.path.join(out, "data", f), delimiter=",", ndmin=2)
    runs[name] = load("electric_field_energy.txt"), load("kinetic_energy.txt")

# ----------------------------- plot, save, and print -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, (field, kinetic) in runs.items():
    ax[0].semilogy(field[:, 0] * 1e9, field[:, 1], lw=2, label=name)
    ax[1].plot(kinetic[:, 0] * 1e9, kinetic[:, 1] / kinetic[0, 1], lw=2, label=name)
    peak = np.argmax(field[:, 1])
    growth = np.polyfit(field[peak // 4: peak, 0], np.log(field[peak // 4: peak, 1]), 1)[0] / 2
    print(f"{name:14s} field-energy growth rate = {growth:.3e} 1/s, peak E-field energy = {field[peak, 1]:.3e} J, "
          f"final kinetic / initial = {kinetic[-1, 1] / kinetic[0, 1]:.4f}")
ax[0].set(xlabel="time [ns]", ylabel="electric field energy [J]", title="Two-stream growth")
ax[1].set(xlabel="time [ns]", ylabel="kinetic energy / initial", title="Electron energy loss to argon")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "two_stream_mcc.png"), dpi=200)
print(f"Saved {OUTPUT_DIR}/two_stream_mcc.png")
