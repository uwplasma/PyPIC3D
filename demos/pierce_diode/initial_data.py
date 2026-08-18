import jax
import jax.numpy as jnp

c = 2.998e8  # speed of light in m/s
PPC = 50  # particles per cell
Nx  = 200  # number of cells in the x direction

# Christopher Woolford Aug 17th 2026
# This script generates the initial data for a Pierce diode 
# instability simulation.

pierce_parameter = 1.2 * jnp.pi  # Pierce parameter
# 0 to pi is stable, pi to 2pi is unstable
# pierce_parameter = w_p * L / v0
# w_p is the plasma frequency, L is the length of the diode, v0 is the initial velocity of the beam

# we are going to do a cold beam simulation, so we will set the initial velocity of the beam to be 0.2c

x_wind = 0.1 # length of the 1D diode
beam_velocity = 0.2 * c # initial velocity of the electron beam
vth           = 10e-12 * c # thermal velocity of the electron beam

effective_timescale = x_wind / beam_velocity  # effective timescale of the simulation


w_p = pierce_parameter * beam_velocity / x_wind  # plasma frequency
# w_p = sqrt(n * e^2 / (m * epsilon_0)) where n is the electron density, e is the electron charge, m is the electron mass, and epsilon_0 is the permittivity of free space

ne = (w_p**2 * 8.854e-12 * 9.109e-31) / (1.602e-19**2)  # electron density in m^-3
# define the plasma density in terms of the plasma frequency

debye_length = vth / w_p  # Debye length
# define the Debye length in terms of the thermal velocity and plasma frequency

y_wind = 1 # debye_length
z_wind = 1 #debye_length
# define the simulation domain size in the y and z directions to be equal to the Debye length

volume = x_wind * y_wind * z_wind  # volume of the simulation domain

N = int(PPC * Nx)  # total number of particles in the simulation domain
weight = (ne * volume) / N  # weight of each particle


print(f"Pierce parameter: {pierce_parameter}")
print(f"Electron density: {ne:.2e} m^-3")
print(f"Debye length: {debye_length:.2e} m")
print(f"Number of macroparticles: {N}")
print(f"Macroparticle weight: {weight:.2e} particles per macroparticle")
print(f"Thermal velocity: {vth:.2e} m/s")
print(f"x points per debye length: {Nx / (x_wind / debye_length):.2f}")
print(f"Beam velocity: {beam_velocity:.2e} m/s")
print(f"10x effective timescale: {10 * effective_timescale:.2e} s")