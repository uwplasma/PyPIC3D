import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
from jax import lax
from jax._src.scipy.sparse.linalg import _vdot_real_tree, _add, _sub, _mul
from jax.tree_util import tree_leaves
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import functools
from functools import partial
# import external libraries

def current_correction(Jx, Jy, Jz, x, y, z, dx, dy, dz):
    ix, iy, iz = jnp.floor(x/dx).astype(int), jnp.floor(y/dy).astype(int), jnp.floor(z/dz).astype(int)

    nubar, etabar = 0, 0

    deltax = x - ix*dx
    deltay = y - iy*dy
    deltaz = z - iz*dz

    Nx, Ny, Nz = Jx.shape
    staggered_Jx = jnp.zeros((Nx, Ny, Nz))
    # build a staggered array along x ( x = x+1/2)
    staggered_Jx = staggered_Jx.at[ix, iy, iz].set(deltax*nubar*etabar + deltax*deltay*deltaz/12)
    # compute the first x correction for charge conservation along i+1/2, j, k
    staggered_Jx = staggered_Jx.at[ix, iy, iz+1].set(deltax*(1-nubar)*etabar - deltax*deltay*deltaz/12)
    # compute the second x correction for charge conservation along i+1/2, j, k+1