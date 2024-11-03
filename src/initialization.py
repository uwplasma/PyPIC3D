import jax
import jax.numpy as jnp
from jax import random
from src.utils import debye_length, load_particles_from_toml
from src.fields import initialize_fields

def setup_simulation(simulation_parameters, dx, dy, dz, Nx, Ny, Nz, solver, cold_start=False):
    theoretical_freq = simulation_parameters["theoretical_freq"]
    dt = simulation_parameters["dt"]
    me = simulation_parameters["me"]
    eps = simulation_parameters["eps"]
    q_e = simulation_parameters["q_e"]
    Te = simulation_parameters["Te"]
    N_electrons = simulation_parameters["N_electrons"]
    x_wind = simulation_parameters["x_wind"]
    y_wind = simulation_parameters["y_wind"]
    z_wind = simulation_parameters["z_wind"]
    kb = simulation_parameters["kb"]
    t_wind = simulation_parameters["t_wind"]

    if theoretical_freq * dt > 2.0:
        print(f"# of Electrons is Low and may introduce numerical stability")
        print(f"In order to correct this, # of Electrons needs to be at least { (2/dt)**2 * (me*eps/q_e**2) } for this spatial resolution")

    debye = debye_length(eps, Te, N_electrons, x_wind, y_wind, z_wind, q_e, kb)
    if debye < dx:
        print(f"Debye Length is less than the spatial resolution, this may introduce numerical instability")

    Nt = int(t_wind / dt)

    print(f'time window: {t_wind}')
    print(f'x window: {x_wind}')
    print(f'y window: {y_wind}')
    print(f'z window: {z_wind}')
    print(f"\nResolution")
    print(f'dx: {dx}')
    print(f'dy: {dy}')
    print(f'dz: {dz}')
    print(f'dt:          {dt}')
    print(f'Nt:          {Nt}\n')

    particles = load_particles_from_toml("config.toml", simulation_parameters, dx, dy, dz)

    if solver == 'autodiff':
        Ex, Ey, Ez, Bx, By, Bz = initialize_fields(particles, (1/(4*jnp.pi*eps)))
        rho = jnp.zeros((Nx, Ny, Nz))
        phi = jnp.zeros((Nx, Ny, Nz))
    else:
        Ex, Ey, Ez, Bx, By, Bz, phi, rho = initialize_fields(Nx, Ny, Nz)

    return particles, Ex, Ey, Ez, Bx, By, Bz, phi, rho