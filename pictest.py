# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities


import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
# Importing relevant libraries


def initial_particles(N_particles, x_wind, y_wind, z_wind, key):
# this method initializes the velocties and the positions of the particles
    x = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=x_wind)
    y = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=y_wind)
    z = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=z_wind)
    # initialize the positions of the particles
    v_x        = jax.numpy.zeros(shape = (N_particles) )
    v_y        = jax.numpy.zeros(shape = (N_particles) )
    v_z        = jax.numpy.zeros(shape = (N_particles) )
    # initialize the velocities of the particles
    return x, y, z, v_x, v_y, v_z

@jit
def laplacian(field):
    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field)/(dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field)/(dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field)/(dz*dz)
    return x_comp + y_comp + z_comp


@jit
def compute_rho(electron_x, electron_y, electron_z, ion_x, ion_y, ion_z):
    rho = jax.numpy.zeros(shape = (Nx, Ny, Nz))

    for electron in range(N_electrons):
        x = (electron_x.at[electron].get() / dx).astype(int)
        y = (electron_y.at[electron].get() / dy).astype(int)
        z = (electron_z.at[electron].get() / dz).astype(int)
        # I am starting by just rounding for now
        # Ideally, I would like to partition the charge across array spacings.
        rho = rho.at[x,y,z].add(q_e)

    for ion in range(N_ions):
        x = (ion_x.at[ion].get() / dx).astype(int)
        y = (ion_y.at[ion].get() / dy).astype(int)
        z = (ion_z.at[ion].get() / dz).astype(int)
        # I am starting by just rounding for now
        # Ideally, I would like to partition the charge across array spacings.
        rho = rho.at[x,y,z].add(q_i)

    return rho

@jit
def solve_poisson(rho):
    return jax.scipy.sparse.linalg.cg(laplacian, rho, rho, maxiter=500)[0]

def compute_Eforce(q, Ex, Ey, Ez, x, y, z):
    # This method computes the force from the electric field using interpolation
    Fx = q*jax.scipy.ndimage.map_coordinates(Ex, [x, y, z], order=1)
    Fy = q*jax.scipy.ndimage.map_coordinates(Ey, [x, y, z], order=1)
    Fz = q*jax.scipy.ndimage.map_coordinates(Ez, [x, y, z], order=1)
    # interpolate the electric field component arrays and calculate the force
    return Fx, Fy, Fz


def update_velocity(vx, vy, vz, Fx, Fy, Fz, dt, m):
    # update the velocity of the particles
    vx = vx + (Fx*dt/m)
    vy = vy + (Fy*dt/m)
    vz = vz + (Fz*dt/m)
    return vx, vy, vz

def update_position(x, y, z, vx, vy, vz):
    # update the position of the particles
    x = x + vx*dt
    y = y + vy*dt
    z = z + vz*dt
    return x, y, z


def plot(x, y, z, t):
    # makes a 3d plot of the positions of the particles
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(electron_x, electron_y, electron_z)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title("Particle Positions")
    plt.savefig(f"plots/particles.{t}.png", dpi=200)
    ax.cla()

############## DEBUG THIS ###########################################################################
def delx(field, dx):
    return (jnp.roll(field.vals, shift=1, axis=0) - jnp.roll(field.vals, shift=-1, axis=0))/(2*dx)

def dely(field, dy):
    return (jnp.roll(field.vals, shift=1, axis=1) - jnp.roll(field.vals, shift=-1, axis=1))/(2*dy)

def delz(field, dz):
    return (jnp.roll(field.vals, shift=1, axis=2) - jnp.roll(field.vals, shift=-1, axis=2))/(2*dz)

def curl(field_x, field_y, field_z, dx, dy, dz):
    x_comp =  dely(field_z, dy) - delz(field_y, dz)
    y_comp =  delz(field_z, dz) - delx(field_z, dz)
    z_comp =  delx(field_y, dx) - dely(field_x, dz)
    return x_comp, y_comp, z_comp

def update_b(Bx, By, Bz, Ex, Ey, Ez, dt, dx, dy, dz):
    dBx, dBy, dBz = -curl(Ex, Ey, Ez, dx, dy, dz) * dt
    return Bx + dBx, By + dBy, Bz + dBz

##########################################################################################################











############################ INITIALIZE EVERYTHING #######################################################
# I am starting by simulating a hydrogen plasma
print("Initializing Simulation...")

me = 9.1093837e-31
# mass of the electron
mi = 1.67e-23
# mass of the ion
q_e = 1.602e-19
# charge of electron
q_i = -1.602e-19
# charge of ion

N_electrons = 500
N_ions      = 500
# specify the number of electrons and ions in the plasma
t_wind = 10e-9
Nt     = 10
dt     = t_wind / Nt
print(f'time window: {t_wind}')
print(f'Nt:          {Nt}')
print(f'dt:          {dt}')

Nx = 1000
Ny = 1000
Nz = 1000
# specify the number of array spacings in x, y, and z
x_wind = 0.5
y_wind = 0.5
z_wind = 0.5
# specify the size of the spatial window in meters

dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
# compute the spatial resolution
print(f'Dx: {dx}')
print(f'Dy: {dy}')
print(f'Dz: {dz}')

Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Bx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
By = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Bz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
# initialize the electric and magnetic field arrays as 0

key = random.key(0)
electron_x, electron_y, electron_z, ev_x, ev_y, ev_z  = initial_particles(N_electrons, x_wind, y_wind, z_wind, key)
ion_x, ion_y, ion_z, iv_x, iv_y, iv_z                 = initial_particles(N_ions, x_wind, y_wind, z_wind, key)
# initialize the positions and velocities of the electrons and ions in the plasma.
# eventually, I need to update the initialization to use a more accurate position and velocity distribution.

for t in range(Nt):
    ############### SOLVE E FIELD ######################################################################################
    print(f'Time: {t*dt} s')
    print("Solving Electric Field...")
    rho    = compute_rho(electron_x, electron_y, electron_z, ion_x, ion_y, ion_z)
    # compute the charge density of the plasma
    #phi = jax.scipy.sparse.linalg.cg(laplacian, rho, rho, maxiter=1000)[0]
    phi = solve_poisson(rho)
    # Use conjugated gradients to calculate the electric potential from the charge density
    E_fields = jnp.gradient(phi)
    Ex       = -1 * E_fields[0]
    Ey       = -1 * E_fields[1]
    Ez       = -1 * E_fields[2]
    # Calculate the E field using the gradient of the potential

    ############### UPDATE ELECTRONS ##########################################################################################
    print("Updating Electrons...")
    Fx, Fy, Fz = compute_Eforce(q_e, Ex, Ey, Ez, electron_x, electron_y, electron_z)
    # compute the force on the electrons from the electric field
    ev_x, ev_y, ev_z = update_velocity(ev_x, ev_y, ev_z, Fx, Fy, Fz, dt, me)
    # Update the velocties from the electric field
    electron_x, electron_y, electron_z = update_position(electron_x, electron_y, electron_z, ev_x, ev_y, ev_z)
    # Update the positions of the particles


    ############### UPDATE IONS ################################################################################################
    print("Updating Ions...")
    Fx, Fy, Fz = compute_Eforce(q_i, Ex, Ey, Ez, ion_x, ion_y, ion_z)
    # compute the force on the ions from the electric field
    iv_x, iv_y, iv_z = update_velocity(iv_x, iv_y, iv_z, Fx, Fy, Fz, dt, mi)
    # Update the velocities from the electric field
    ion_x, ion_y, ion_z  = update_position(ion_x, ion_y, ion_z, iv_x, iv_y, iv_z)
    # Update the positions of the particles
    x = jnp.concatenate([electron_x, ion_x])
    y = jnp.concatenate([electron_y, ion_y])
    z = jnp.concatenate([electron_z, ion_z])
    plot( x, y, z, t)
    # plot the particles and save as png file