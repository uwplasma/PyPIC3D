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
def index_particles(particle, positions, ds):
    return (positions.at[particle].get() / ds).astype(int)

@jit
def update_rho(Nparticles, particlex, particley, particlez, dx, dy, dz, q, rho):
    def addto_rho(particle, rho):
        x = index_particles(particle, particlex, dx)
        y = index_particles(particle, particley, dy)
        z = index_particles(particle, particlez, dz)
        rho = rho.at[x, y, z].add( q / (dx*dy*dz) )
        return rho
    
    return jax.lax.fori_loop(0, Nparticles-1, addto_rho, rho )

@jit
def solve_poisson(rho, eps, dx, dy, dz, phi, M = None):
    lapl = functools.partial(laplacian, dx=dx, dy=dy, dz=dz)
    phi, exitcode = jax.scipy.sparse.linalg.cg(lapl, rho/eps, phi, maxiter=5000, M=M)
    return phi