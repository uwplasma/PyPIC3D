import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import functools

@jit
def laplacian(field, dx, dy, dz):
    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field)/(dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field)/(dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field)/(dz*dz)
    return x_comp + y_comp + z_comp


@jit
def compute_rho(rho, electron_x, electron_y, electron_z, ion_x, ion_y, ion_z, dx, dy, dz, q_i, q_e):
    N_electrons = electron_x.shape[0]
    N_ions      = ion_x.shape[0]

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
def solve_poisson(rho, eps, dx, dy, dz, phi=None, M = None):
    lapl = functools.partial(laplacian, dx=dx, dy=dy, dz=dz)
    phi, exitcode = jax.scipy.sparse.linalg.cg(lapl, rho, rho, maxiter=8000, M=M)
    #print(exitcode)
    return phi