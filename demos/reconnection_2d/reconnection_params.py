import argparse
import shutil
import sys
from pathlib import Path

import dill
import numpy as np

from pywarpx import callbacks, fields, libwarpx, picmi

constants = picmi.constants

cfl = 0.99 # CFL number for stability

B0 = 0.1  # Initial magnetic field strength (T)

# Physical parameters
m_ion = 400.0  # Ion mass (electron masses)

beta_e = 0.1
Bg = 0.3  # times B0 - guiding field
dB = 0.01  # times B0 - initial perturbation to seed reconnection

T_ratio = 5.0  # T_i / T_e

# Domain parameters
LX = 40  # ion skin depths
LZ = 20  # ion skin depths

LT = 50  # ion cyclotron periods
# DT = 1e-3  # ion cyclotron periods

# Resolution parameters
NX = 300
NZ = 300

# Ion mass (kg)
M = m_ion * constants.m_e

# Cyclotron angular frequency (rad/s) and period (s)
w_ce = constants.q_e * abs(B0) / constants.m_e
w_ci = constants.q_e * abs(B0) / M
t_ci = 2.0 * np.pi / w_ci

# Electron plasma frequency: w_pe / omega_ce = 2 is given
w_pe = 2.0 * w_ce

# calculate plasma density based on electron plasma frequency
n_plasma = w_pe**2 * constants.m_e * constants.ep0 / constants.q_e**2

# Ion plasma frequency (Hz)
w_pi = np.sqrt(constants.q_e**2 * n_plasma / (M * constants.ep0))

# Ion skin depth (m)
l_i = constants.c / w_pi

# # Alfven speed (m/s): vA = B / sqrt(mu0 * n * (M + m)) = c * omega_ci / w_pi
vA = abs(B0) / np.sqrt(
    constants.mu0 * n_plasma * (constants.m_e + M)
)

# calculate Te based on beta
Te = (
    beta_e
    * B0**2
    / (2.0 * constants.mu0 * n_plasma)
    / constants.q_e
)
Ti = Te * T_ratio

# calculate thermal speeds
ve_th = np.sqrt(Te * constants.q_e / constants.m_e)
vi_th = np.sqrt(Ti * constants.q_e / M)

# Ion Larmor radius (m)
rho_i = vi_th / w_ci

# Reference resistivity (Malakit et al.)
eta0 = l_i * vA / (constants.ep0 * constants.c**2)

Lx = LX * l_i
Lz = LZ * l_i

dx = Lx / NX
dz = Lz / NZ

dt = cfl / constants.c / (1/dx + 1/dz)
# compute the time step in seconds based on the cfl condition and the speed of light, then convert to ion cyclotron periods

# dt = DT * t_ci

# run very low resolution as a CI test
total_steps = int(LT * t_ci / dt)
# compute the total number of steps based on the total simulation time in seconds and the time step

# total_steps = int(LT / DT)
diag_steps = total_steps // 200

# # Initial magnetic field
# Bg *= B0
# dB *= B0
# Bx = (
#     f"{B0}*tanh(z*{1.0 / l_i})"
#     f"+{-dB * Lx / (2.0 * Lz)}*cos({2.0 * np.pi / Lx}*x)"
#     f"*sin({np.pi / Lz}*z)"
# )
# By = (
#     f"sqrt({Bg**2 + B0**2}-({B0}*tanh(z*{1.0 / l_i}))**2)"
# )
# Bz = f"{dB}*sin({2.0 * np.pi / Lx}*x)*cos({np.pi / Lz}*z)"

# J0 = B0 / constants.mu0 / l_i
# # Build the initial magnetic field on a (NX, NY, NZ) grid (NY=1 for this 2D x-z setup)
NY = 1

x = np.linspace(-Lx / 2.0, Lx / 2.0, NX, endpoint=False)
y = np.zeros(NY)
z = np.linspace(-Lz / 2.0, Lz / 2.0, NZ, endpoint=False)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

Bx_np = B0 * np.tanh(Z / l_i) + (-dB * Lx / (2.0 * Lz)) * np.cos(
    (2.0 * np.pi / Lx) * X
) * np.sin((np.pi / Lz) * Z)

By_np = np.sqrt(Bg**2 + B0**2 - (B0 * np.tanh(Z / l_i)) ** 2)

Bz_np = dB * np.sin((2.0 * np.pi / Lx) * X) * np.cos((np.pi / Lz) * Z)

np.save("Bx_initial.npy", Bx_np)
np.save("By_initial.npy", By_np)
np.save("Bz_initial.npy", Bz_np)
# save the parameters to a file for later use

print(f"Parameters:")
print(f" Nx, Nz = {NX}, {NZ}")
print(f" Lx, Lz = {Lx:.2e} m, {Lz:.2e} m")
print(f" dt = {dt:.2e} s")
print(f" Total steps = {total_steps}")
print(f" Particles per cell = { 200 }")
print(f" Number of Particles = {200 * NX * NZ}")
print(f" B0 = {B0:.2e} T")
print(f" VA = {vA:.2e} m/s")
print(f" Ion Cyclotron Period: {t_ci} s ")
print(f"Te = {Te:.2e} eV")
print(f"Ti = {Ti:.2e} eV")
print(f"mi = {M:.2e} kg")
print(f"ve_th = {ve_th:.2e} m/s")
print(f"vi_th = {vi_th:.2e} m/s")
