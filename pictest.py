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
import math
from pyevtk.hl import gridToVTK
# Importing relevant libraries

def initial_particles(N_particles, x_wind, y_wind, z_wind, mass, T, kb, key):
# this method initializes the velocties and the positions of the particles
    x = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=0.05 * x_wind)
    y = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=0.05 * y_wind)
    z = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=0.05 * z_wind)
    # initialize the positions of the particles
    std = kb * T / mass
    v_x = np.random.normal(0, std, N_particles)
    v_y = np.random.normal(0, std, N_particles)
    v_z = np.random.normal(0, std, N_particles)
    # initialize the particles with a maxwell boltzmann distribution.
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
def solve_poisson(rho, eps, phi=None, M = None):
    phi, exitcode = jax.scipy.sparse.linalg.cg(laplacian, rho, rho, maxiter=8000, M=M)
    #print(exitcode)
    return phi

    # b = rho/eps
    # x = 0
    # # initalize
    # if phi is None:
    #     r = b
    # else:
    #     r = b - laplacian(phi)
    
    # M = jnp.zeros((100,100,100))
    # for i in range(1, rho.shape[0]):
    #     M = M.at[i, i, i].add(1)
    # print("Initialize Poisson Solver")
    # print(f"M: {jnp.max(M)}")
    # for i in range(10):
    #     rt = jnp.transpose(r)
    #     # compute the intial approximation for r and its transpose

    #     Minv = M
    #     # compute the inverse of the preconditioner
    #     print(f"Minv: {jnp.max(Minv)}")
    #     d = r  #jnp.multiply( M, r)
    #     alpha = jnp.multiply(rt, laplacian(r) )
    #     print(f"alpha: {alpha.shape}")
    #     print(f"d: {d.shape}")
    #     x = x + jnp.multiply( alpha, d )
    #     print(f"x: {jnp.max(x)}")
    #     rnew = r - jnp.multiply( alpha, laplacian(d) )
    #     print(f"rnew: {jnp.max(x)}")
    #     # need to work through all this
    #     beta =jnp.multiply( jnp.multiply( jnp.transpose(rnew), laplacian(rnew) ),  1 / jnp.multiply(rt, r) )
    #     print(f"beta: {jnp.max(beta)}")
    #     d = jnp.dot( M, rnew ) + beta * d    
    #     r = rnew
    #     # reassign new value of r
    #     print("Update")
    #     print(f"Error: {jnp.max( b - laplacian(x) )}")

    
    # return x

@jit
def boris(q, Ex, Ey, Ez, Bx, By, Bz, x, y, z, vx, vy, vz, dt, m):
    efield_atx = jax.scipy.ndimage.map_coordinates(Ex, [x, y, z], order=1)
    efield_aty = jax.scipy.ndimage.map_coordinates(Ey, [x, y, z], order=1)
    efield_atz = jax.scipy.ndimage.map_coordinates(Ez, [x, y, z], order=1)
    # interpolate the electric field component arrays and calculate the e field at the particle positions
    bfield_atx = jax.scipy.ndimage.map_coordinates(Bx, [x, y, z], order=1)
    bfield_aty = jax.scipy.ndimage.map_coordinates(By, [x, y, z], order=1)
    bfield_atz = jax.scipy.ndimage.map_coordinates(Bz, [x, y, z], order=1)
    # interpolate the magnetic field component arrays and calculate the b field at the particle positions


    vxminus = vx + q*dt/(2*m)*efield_atx
    vyminus = vy + q*dt/(2*m)*efield_aty
    vzminus = vz + q*dt/(2*m)*efield_atz
    # calculate the v minus vector used in the boris push algorithm
    tx = q*dt/(2*m)*bfield_atx
    ty = q*dt/(2*m)*bfield_aty
    tz = q*dt/(2*m)*bfield_atz

    vprimex = vxminus + (vyminus*tz - vzminus*ty)
    vprimey = vyminus + (vzminus*tx - vxminus*tz)
    vprimez = vzminus + (vxminus*ty - vyminus*tx)
    # vprime = vminus + vminus cross t

    smag = 2 / (1 + tx*tx + ty*ty + tz*tz)
    sx = smag * tx
    sy = smag * ty
    sz = smag * tz
    # calculate the scaled rotation vector

    vxplus = vxminus + (vprimey*sz - vprimez*sy)
    vyplus = vyminus + (vprimez*sx - vprimex*sz)
    vzplus = vzminus + (vprimex*sy - vprimey*sx)

    newvx = vxplus + q*dt/(2*m)*efield_atx
    newvy = vyplus + q*dt/(2*m)*efield_aty
    newvz = vzplus + q*dt/(2*m)*efield_atz
    # calculate the new velocity

    return newvx, newvy, newvz


# @jit
# def compute_Eforce(q, Ex, Ey, Ez, x, y, z):
#     # This method computes the force from the electric field using interpolation
#     Fx = q*jax.scipy.ndimage.map_coordinates(Ex, [x, y, z], order=1)
#     Fy = q*jax.scipy.ndimage.map_coordinates(Ey, [x, y, z], order=1)
#     Fz = q*jax.scipy.ndimage.map_coordinates(Ez, [x, y, z], order=1)
#     # interpolate the electric field component arrays and calculate the force
#     return Fx, Fy, Fz

@jit
def curlx(yfield, zfield, dy, dz):
    delZdely = (jnp.roll(zfield, shift=1, axis=1) + jnp.roll(zfield, shift=-1, axis=1) - 2*zfield)/(dy*dy)
    delYdelz = (jnp.roll(yfield, shift=1, axis=2) + jnp.roll(yfield, shift=-1, axis=2) - 2*yfield)/(dz*dz)
    return delZdely - delYdelz

@jit
def curly(xfield, zfield, dx, dz):
    delXdelz = (jnp.roll(xfield, shift=1, axis=2) + jnp.roll(xfield, shift=-1, axis=2) - 2*xfield)/(dz*dz)
    delZdelx = (jnp.roll(zfield, shift=1, axis=0) + jnp.roll(zfield, shift=-1, axis=0) - 2*zfield)/(dx*dx)
    return delXdelz - delZdelx

@jit
def curlz(yfield, xfield, dx, dy):
    delYdelx = (jnp.roll(yfield, shift=1, axis=0) + jnp.roll(yfield, shift=-1, axis=0) - 2*yfield)/(dx*dx)
    delXdely = (jnp.roll(xfield, shift=1, axis=1) + jnp.roll(xfield, shift=-1, axis=1) - 2*xfield)/(dy*dy)
    return delYdelx - delXdely

@jit
def update_B(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt):
    Bx = Bx - dt*curlx(Ey, Ez, dy, dz)
    By = By - dt*curly(Ex, Ez, dx, dz)
    Bz = Bz - dt*curlz(Ex, Ey, dx, dy)
    return Bx, By, Bz



# @jit
# def update_velocity(vx, vy, vz, Fx, Fy, Fz, dt, m):
#     # update the velocity of the particles
#     vx = vx + (Fx*dt/2/m)
#     vy = vy + (Fy*dt/2/m)
#     vz = vz + (Fz*dt/2/m)
#     return vx, vy, vz

@jit
def update_position(x, y, z, vx, vy, vz):
    # update the position of the particles
    x = x + vx*dt/2
    y = y + vy*dt/2
    z = z + vz*dt/2
    return x, y, z

def plot_fields(fieldx, fieldy, fieldz, t, name, dx, dy, dz):
    Nx = Ex.shape[0]
    Ny = Ex.shape[1]
    Nz = Ex.shape[2]
    x = np.linspace(0, Nx, Nx) * dx
    y = np.linspace(0, Ny, Ny) * dy
    z = np.linspace(0, Nz, Nz) * dz
    gridToVTK(f"./plots/fields/{name}_{t:09}", x, y, z,   \
            cellData = {f"{name}_x" : np.asarray(fieldx), \
             f"{name}_y" : np.asarray(fieldy), f"{name}_z" : np.asarray(fieldz)}) 
# plot the electric fields in the vtk file format

def plot_positions(x, y, z, t, x_wind, y_wind, z_wind):
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
    plt.savefig(f"plots/positions/particles.{t:09}.png", dpi=300)
    plt.close()






############################ INITIALIZE EVERYTHING #######################################################
# I am starting by simulating a hydrogen plasma
print("Initializing Simulation...")

eps = 8.854e-12
# permitivity of freespace
C = 3e8 # m/s
# Speed of light
eps = 8.854e-12
# permitivity of freespace
C = 3e8 # m/s
# Speed of light
kb = 1.380649e-23 # J/K
# Boltzmann's constant
me = 9.1093837e-31 # Kg
# mass of the electron
mi = 1.67e-23 # Kg
# mass of the ion
q_e = 1.602e-19
# charge of electron
q_i = -1.602e-19
# charge of ion
Te = 100 # K
# electron temperature
Ti = 100 # K
# ion temperature
# assuming an isothermal plasma for now

N_electrons = 100
N_ions      = 100
# specify the number of electrons and ions in the plasma

Nx = 50
Ny = 50
Nz = 50
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

################ Courant Condition #############################################################################
courant_number = 1
dt = courant_number / (  C * ( (1/dx) + (1/dy) + (1/dz) )   )
# calculate spatial resolution using courant condition

t_wind = 1e-9
# time window for simultion
Nt     = int( t_wind / dt )
# Nt for resolution


print(f'time window: {t_wind}')
print(f'Nt:          {Nt}')
print(f'dt:          {dt}')

plot_freq = 3
# plot data every 5 timesteps

Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
# initialize the electric field arrays as 0
Bx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
By = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Bz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
# initialize the magnetic field arrays as 0

key1 = random.key(4353)
key2 = random.key(1043)
electron_x, electron_y, electron_z, ev_x, ev_y, ev_z  = initial_particles(N_electrons, x_wind, y_wind, z_wind, me, Te, kb, key1)
ion_x, ion_y, ion_z, iv_x, iv_y, iv_z                 = initial_particles(N_ions, x_wind, y_wind, z_wind, mi, Ti, kb, key2)
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

M = None
# set poisson solver precondition to None


for t in range(10):
    if t % plot_freq == 0:
    ############## PLOTTING ###################################################################   
        start = time.time()
        particlesx = jnp.concatenate([electron_x, ion_x])
        particlesy = jnp.concatenate([electron_y, ion_y])
        particlesz = jnp.concatenate([electron_z, ion_z])
        plot_positions( particlesx, particlesy, particlesz, t, x_wind, y_wind, z_wind)
        plot_fields(Ex, Ey, Ez, t, "E", dx, dy, dz)
        plot_fields(Bx, By, Bz, t, "B", dx, dy, dz)
        end  = time.time()
        print(f'Time Spent on Plotting: {end-start} s')
        average_plot.append(end-start)
    # plot the particles and save as png file

    ############### SOLVE E FIELD ######################################################################################
    print(f'Time: {t*dt} s')
    print("Solving Electric Field...")
    start = time.time()
    rho    = compute_rho(electron_x, electron_y, electron_z, ion_x, ion_y, ion_z, dx, dy, dz)
    end   = time.time()
    print(f"Time Spent on Rho: {end-start} s")
    average_rho.append(end-start)
    #print( f'Max Value of Rho: {jnp.max(rho)}' )
    # compute the charge density of the plasma
    start = time.time()
    phi = solve_poisson(rho, eps, M)
    end   = time.time()
    #print(f"Time Spent on Phi: {end-start} s")
    average_poisson.append(end-start)
    #print( f'Max Value of Phi: {jnp.max(phi)}' )
    print( f'Max Laplacian of Phi: {jnp.max(laplacian(phi))}')
    print( f'Max Charge Density: {jnp.max(rho)}' )
    print( f'Poisson Error: {jnp.max( laplacian(phi) - (1/eps)*rho )}' )
    # Use conjugated gradients to calculate the electric potential from the charge density
    #M = (eps * phi * jnp.linalg.inv(rho))[:,:,0]
    #M = (eps * phi)[:,:,0]
    #M = (1/eps) * jnp.linalg.inv(rho[:,:,0])
    # testing a new approximation for M because it appears that rho does not always have an inverse matrix    


    #print( f"Max Value of M: {jnp.max( M ) }" )
    # Using the assumption that the timestep is small, precondition poisson solver with
    # previous values for phi and rho

    # exit()

    start = time.time()
    E_fields = jnp.gradient(phi)
    Ex       = -1 * E_fields[0]
    Ey       = -1 * E_fields[1]
    Ez       = -1 * E_fields[2]
    end = time.time()
    #print(f'Time Spent Calculating E: {end-start} s')
    average_E.append(end-start)
    # Calculate the E field using the gradient of the potential
    #print( f'Max Value of Ex: {jnp.max(Ex)}' )
    #print( f'Max Value of Ey: {jnp.max(Ey)}' )
    #print( f'Max Value of Ez: {jnp.max(Ez)}' )

    ################ MAGNETIC FIELD UPDATE #######################################################################
    Bx, By, Bz = update_B(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt)

    ############### UPDATE ELECTRONS ##########################################################################################
    print("Updating Electrons...")
    start = time.time()
    ev_x, ev_y, ev_z = boris(q_e, Ex, Ey, Ez, Bx, By, Bz, electron_x, \
                             electron_y, electron_z, ev_x, ev_y, ev_z, dt, me)
    # implement the boris push algorithm to solve for new particle velocities



    # start = time.time()
    # Fx, Fy, Fz = compute_Eforce(q_e, Ex, Ey, Ez, electron_x, electron_y, electron_z)
    # end = time.time()
    # #print(f'Time Spent on Calculating Electrons Force: {end-start} s')
    # average_electron_Force.append(end-start)
    # # compute the force on the electrons from the electric field
    # start = time.time()
    # ev_x, ev_y, ev_z = update_velocity(ev_x, ev_y, ev_z, Fx, Fy, Fz, dt, me)
    # # Update the velocties from the electric field
    electron_x, electron_y, electron_z = update_position(electron_x, electron_y, electron_z, ev_x, ev_y, ev_z)
    # Update the positions of the particles
    end   = time.time()
    #print(f'Time Spent on Updating Electrons: {end-start} s')
    average_e_update.append(end-start)

    ############### UPDATE IONS ################################################################################################
    print("Updating Ions...")

    start = time.time()
    iv_x, iv_y, iv_z = boris(q_i, Ex, Ey, Ez, Bx, By, Bz, ion_x, \
                             ion_y, ion_z, iv_x, iv_y, iv_z, dt, mi)

    # start = time.time()
    # Fx, Fy, Fz = compute_Eforce(q_i, Ex, Ey, Ez, ion_x, ion_y, ion_z)
    # end   = time.time()
    # #print(f"Time Spent on Calculating Ions Force: {end-start} s")
    # average_ion_Force.append(end-start)
    # # compute the force on the ions from the electric field
    # start = time.time()
    # iv_x, iv_y, iv_z = update_velocity(iv_x, iv_y, iv_z, Fx, Fy, Fz, dt, mi)
    # # Update the velocities from the electric field
    ion_x, ion_y, ion_z  = update_position(ion_x, ion_y, ion_z, iv_x, iv_y, iv_z)
    end   = time.time()
    #print(f"Time Spent on Updating Ions: {end-start} s")
    average_ion_update.append(end-start)
    # Update the positions of the particles
    


print(f"Average Rho: {np.mean(average_rho[1:])} s")
print(f"Average Poisson: {np.mean(average_poisson[1:])} s")
print(f"Average E: {np.mean(average_E[1:])} s")
print(f"Average Electron Force: {np.mean(average_electron_Force[1:])} s")
print(f"Average Ion Force: {np.mean(average_ion_Force[1:])} s")
print(f"Average Electron Update: {np.mean(average_e_update[1:])} s")
print(f"Average Ion Update: {np.mean(average_ion_update[1:])} s")
print(f"Average Plotting: {np.mean(average_plot[1:])} s")

totaltime = np.mean(average_rho[1:]) + np.mean(average_poisson[1:]) + np.mean(average_E[1:]) + np.mean(average_electron_Force[1:]) + np.mean(average_ion_Force[1:]) + np.mean(average_e_update[1:]) + np.mean(average_ion_update[1:]) + np.mean(average_plot[1:])
print(f'Average Time Per Step {totaltime} s')

totaljittime = np.mean(average_rho[0]) + np.mean(average_poisson[0]) + np.mean(average_E[0]) + np.mean(average_electron_Force[0]) + np.mean(average_ion_Force[0]) + np.mean(average_e_update[0]) + np.mean(average_ion_update[0]) + np.mean(average_plot[0])
print(f'JIT Compile Time: {totaljittime} s')
