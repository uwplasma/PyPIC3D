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

q_e = 1.602e-19
# charge of electron
q_i = -1.602e-19
# charge of ion

N_electrons = 500
N_ions      = 500

t_wind = 10e-9
Nt     = 1000
dt     = t_wind / Nt
print(f'time window: {t_wind}')
print(f'Nt:          {Nt}')
print(f'dt:          {dt}')

Nx = 1000
Ny = 1000
Nz = 1000

x_wind = 0.5
y_wind = 0.5
z_wind = 0.5

dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz

print(f'Dx: {dx}')
print(f'Dy: {dy}')
print(f'Dz: {dz}')

key = random.key(0)

electron_x = jax.random.uniform(key, shape = (N_electrons,), minval=0, maxval=x_wind)
electron_y = jax.random.uniform(key, shape = (N_electrons,), minval=0, maxval=y_wind)
electron_z = jax.random.uniform(key, shape = (N_electrons,), minval=0, maxval=z_wind)

ion_x      = jax.random.uniform(key, shape = (N_ions,), minval=0, maxval=x_wind)
ion_y      = jax.random.uniform(key, shape = (N_ions,), minval=0, maxval=y_wind)
ion_z      = jax.random.uniform(key, shape = (N_ions,), minval=0, maxval=z_wind)

ev_x        = jax.numpy.zeros(shape = (N_electrons) )
ev_y        = jax.numpy.zeros(shape = (N_electrons) )
ev_z        = jax.numpy.zeros(shape = (N_electrons) )

iv_x        = jax.numpy.zeros(shape = (N_ions) )
iv_y        = jax.numpy.zeros(shape = (N_ions) )
iv_z        = jax.numpy.zeros(shape = (N_ions) )


Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )

Bx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
By = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Bz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )

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
    
@jit
def laplacian(field):
    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field)/(dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field)/(dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field)/(dz*dz)
    return x_comp + y_comp + z_comp

# @jit
# def CG_poisson(guess, phi, r):
#     rprime = jnp.transpose(r)
#     phiprime = jnp.transpose(phi)
#     alpha = rprime*r / (phiprime*laplacian(phi))
#     guess = guess + alpha * phi
#     rnew = r - rprime*r/phiprime
#     beta = jnp.transpose(rnew) * rnew / (rprime * r)
#     r = rnew
#     phi = r + beta * phi
#     return guess, phi, r


# def compute_poisson(rho, guess, phi, r, eps):
#     for i in range(1000):
#         guess, phi, r = CG_poisson(guess, phi, r)    
#     return phi

start = time.time()
rho    = compute_rho(electron_x, electron_y, electron_z, ion_x, ion_y, ion_z)
end = time.time()
print("Calculated Rho")
print(f'Time Elapsed: {end-start}')


start = time.time()
rho    = compute_rho(electron_x, electron_y, electron_z, ion_x, ion_y, ion_z)
end = time.time()
print("After Jit Calculated Rho")
print(f'Time Elapsed: {end-start}')


phi = jax.scipy.sparse.linalg.cg(laplacian, rho, jnp.zeros(shape = (Nx, Ny, Nz) ), maxiter=1000)
# Use conjugated gradients to calculate the electric potential from the charge density
Ex, Ey, Ez = -1*jnp.gradient(phi)
# Calculate the E field using the gradient of the potential



# guess = jnp.zeros(shape = (Nx, Ny, Nz) )
# r = rho
# phi = r

# start = time.time()
# guess, phi, r    = CG_poisson(guess, phi, r)
# end = time.time()
# print("One CG Pass")
# print(f'Time Elapsed: {end-start}')


# start = time.time()
# guess, phi, r    = CG_poisson(guess, phi, r)
# end = time.time()
# print("After Jit One CG Pass")
# print(f'Time Elapsed: {end-start}')



# start = time.time()
# phi = compute_poisson(rho, guess, phi, r, 1)
# end = time.time()
# print("One Full Poisson Pass")
# print(f"Time Elapsed: {end-start}")
