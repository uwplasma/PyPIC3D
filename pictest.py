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
def compute_rho(electron_x, electron_y, electron_z, ion_x, ion_y, ion_z, dx, dy, dz):
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

    rho = rho / (dx*dy*dz)
    # divide by cell volume
    return rho

@jit
def solve_poisson(rho):
    return jax.scipy.sparse.linalg.cg(laplacian, rho, rho, maxiter=500)[0]

@jit
def compute_Eforce(q, Ex, Ey, Ez, x, y, z):
    # This method computes the force from the electric field using interpolation
    Fx = q*jax.scipy.ndimage.map_coordinates(Ex, [x, y, z], order=1)
    Fy = q*jax.scipy.ndimage.map_coordinates(Ey, [x, y, z], order=1)
    Fz = q*jax.scipy.ndimage.map_coordinates(Ez, [x, y, z], order=1)
    # interpolate the electric field component arrays and calculate the force
    return Fx, Fy, Fz


@jit
def update_velocity(vx, vy, vz, Fx, Fy, Fz, dt, m):
    # update the velocity of the particles
    vx = vx + (Fx*dt/m)
    vy = vy + (Fy*dt/m)
    vz = vz + (Fz*dt/m)
    return vx, vy, vz

@jit
def update_position(x, y, z, vx, vy, vz):
    # update the position of the particles
    x = x + vx*dt
    y = y + vy*dt
    z = z + vz*dt
    return x, y, z


def plot(x, y, z, t, x_wind, y_wind, z_wind):
    # makes a 3d plot of the positions of the particles
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim( -x_wind, x_wind )
    ax.set_ylim( -y_wind, y_wind )
    ax.set_zlim( -z_wind, z_wind )
    ax.scatter(electron_x, electron_y, electron_z)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title("Particle Positions")
    plt.savefig(f"plots/particles.{t:09}.png", dpi=300)
    ax.cla()






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

t_wind = 100e-9
Nt     = 3#0000
# Nt for resolution
dt     = t_wind / Nt
# Actual number of steps to loop over
print(f'time window: {t_wind}')
print(f'Nt:          {Nt}')
print(f'dt:          {dt}')

Nx = 1000
Ny = 1000
Nz = 1000
# specify the number of array spacings in x, y, and z
x_wind = 0.5e-2
y_wind = 0.5e-2
z_wind = 0.5e-2
# specify the size of the spatial window in meters

dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
# compute the spatial resolution
print(f'Dx: {dx}')
print(f'Dy: {dy}')
print(f'Dz: {dz}')

Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
# initialize the electric field arrays as 0

key1 = random.key(4353)
key2 = random.key(1043)
electron_x, electron_y, electron_z, ev_x, ev_y, ev_z  = initial_particles(N_electrons, x_wind, y_wind, z_wind, key1)
ion_x, ion_y, ion_z, iv_x, iv_y, iv_z                 = initial_particles(N_ions, x_wind, y_wind, z_wind, key2)
# initialize the positions and velocities of the electrons and ions in the plasma.
# eventually, I need to update the initialization to use a more accurate position and velocity distribution.

average_rho            = []
average_poisson        = []
average_E              = []
average_electron_Force = []
average_ion_Force      = []
average_e_update       = []
average_ion_update     = []
average_plot           = []
# create lists for average times

for t in range(Nt):
    ############### SOLVE E FIELD ######################################################################################
    print(f'Time: {t*dt} s')
    print("Solving Electric Field...")
    start = time.time()
    rho    = compute_rho(electron_x, electron_y, electron_z, ion_x, ion_y, ion_z, dx, dy, dz)
    end   = time.time()
    print(f"Time Spent on Rho: {end-start} s")
    average_rho.append(end-start)
    print( f'Max Value of Rho: {jnp.max(rho)}' )
    # compute the charge density of the plasma
    start = time.time()
    phi = solve_poisson(rho)
    end   = time.time()
    print(f"Time Spent on Phi: {end-start} s")
    average_poisson.append(end-start)
    print( f'Max Value of Phi: {jnp.max(phi)}' )
    # Use conjugated gradients to calculate the electric potential from the charge density
    start = time.time()
    E_fields = jnp.gradient(phi)
    Ex       = -1 * E_fields[0]
    Ey       = -1 * E_fields[1]
    Ez       = -1 * E_fields[2]
    end = time.time()
    print(f'Time Spent Calculating E: {end-start} s')
    average_E.append(end-start)
    # Calculate the E field using the gradient of the potential
    print( f'Max Value of Ex: {jnp.max(Ex)}' )
    print( f'Max Value of Ey: {jnp.max(Ey)}' )
    print( f'Max Value of Ez: {jnp.max(Ez)}' )
    ############### UPDATE ELECTRONS ##########################################################################################
    print("Updating Electrons...")
    start = time.time()
    Fx, Fy, Fz = compute_Eforce(q_e, Ex, Ey, Ez, electron_x, electron_y, electron_z)
    end = time.time()
    print(f'Time Spent on Calculating Electrons Force: {end-start} s')
    average_electron_Force.append(end-start)
    # compute the force on the electrons from the electric field
    start = time.time()
    ev_x, ev_y, ev_z = update_velocity(ev_x, ev_y, ev_z, Fx, Fy, Fz, dt, me)
    # Update the velocties from the electric field
    electron_x, electron_y, electron_z = update_position(electron_x, electron_y, electron_z, ev_x, ev_y, ev_z)
    # Update the positions of the particles
    end   = time.time()
    print(f'Time Spent on Updating Electrons: {end-start} s')
    average_e_update.append(end-start)

    ############### UPDATE IONS ################################################################################################
    print("Updating Ions...")
    start = time.time()
    Fx, Fy, Fz = compute_Eforce(q_i, Ex, Ey, Ez, ion_x, ion_y, ion_z)
    end   = time.time()
    print(f"Time Spent on Calculating Ions Force: {end-start} s")
    average_ion_Force.append(end-start)
    # compute the force on the ions from the electric field
    start = time.time()
    iv_x, iv_y, iv_z = update_velocity(iv_x, iv_y, iv_z, Fx, Fy, Fz, dt, mi)
    # Update the velocities from the electric field
    ion_x, ion_y, ion_z  = update_position(ion_x, ion_y, ion_z, iv_x, iv_y, iv_z)
    end   = time.time()
    print(f"Time Spent on Updating Ions: {end-start} s")
    average_ion_update.append(end-start)
    # Update the positions of the particles
    start = time.time()
    x = jnp.concatenate([electron_x, ion_x])
    y = jnp.concatenate([electron_y, ion_y])
    z = jnp.concatenate([electron_z, ion_z])
    plot( x, y, z, t, x_wind, y_wind, z_wind)
    end  = time.time()
    print(f'Time Spent on Plotting: {end-start} s')
    average_plot.append(end-start)
    # plot the particles and save as png file


    print(f"Average Rho: {np.mean(average_rho[1:])} s")
    print(f"Average Poisson: {np.mean(average_poisson[1:])} s")
    print(f"Average E: {np.mean(average_E[1:])} s")
    print(f"Average Electron Force: {np.mean(average_electron_Force[1:])} s")
    print(f"Average Ion Force: {np.mean(average_ion_Force[1:])} s")
    print(f"Average Electron Update: {np.mean(average_e_update[1:])} s")
    print(f"Average Ion Update: {np.mean(average_ion_update[1:])} s")
    print(f"Average Plotting: {np.mean(average_plot[1:])} s")