import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
# from jax import lax
# from jax._src.scipy.sparse.linalg import _vdot_real_tree, _add, _sub, _mul
# from jax.tree_util import tree_leaves
import jax.numpy as jnp
import math
# from pyevtk.hl import gridToVTK
import functools
from functools import partial

from PyPIC3D.utils import use_gpu_if_set, interpolate_field

@jit
@use_gpu_if_set
def particle_push(particles, Ex, Ey, Ez, Bx, By, Bz, grid, staggered_grid, dt, GPUs):
    """
    Updates the velocities of particles using the Boris algorithm.

    Args:
        particles (Particles): The particles to be updated.
        Ex (array-like): Electric field component in the x-direction.
        Ey (array-like): Electric field component in the y-direction.
        Ez (array-like): Electric field component in the z-direction.
        Bx (array-like): Magnetic field component in the x-direction.
        By (array-like): Magnetic field component in the y-direction.
        Bz (array-like): Magnetic field component in the z-direction.
        grid (Grid): The grid on which the fields are defined.
        staggered_grid (Grid): The staggered grid for field interpolation.
        dt (float): The time step for the update.

    Returns:
        Particles: The particles with updated velocities.
    """
    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_position()
    vx, vy, vz = particles.get_velocity()
    # get the charge, mass, position, and velocity of the particles
    #newvx, newvy, newvz = boris(x, y, z, vx, vy, vz, q, m, Ex, Ey, Ez, Bx, By, Bz, grid, staggered_grid, dt)

    Ex_interpolate = jax.scipy.interpolate.RegularGridInterpolator(grid, Ex, fill_value=0)
    Ey_interpolate = jax.scipy.interpolate.RegularGridInterpolator(grid, Ey, fill_value=0)
    Ez_interpolate = jax.scipy.interpolate.RegularGridInterpolator(grid, Ez, fill_value=0)
    Bx_interpolate = jax.scipy.interpolate.RegularGridInterpolator(staggered_grid, Bx, fill_value=0)
    By_interpolate = jax.scipy.interpolate.RegularGridInterpolator(staggered_grid, By, fill_value=0)
    Bz_interpolate = jax.scipy.interpolate.RegularGridInterpolator(staggered_grid, Bz, fill_value=0)
    E_interpolate = (Ex_interpolate, Ey_interpolate, Ez_interpolate)
    B_interpolate = (Bx_interpolate, By_interpolate, Bz_interpolate)
    # create interpolators for the electric and magnetic fields

    #boris_push = jax.vmap(partial(boris, q=q, m=m, E_interpolate=E_interpolate, B_interpolate=B_interpolate, grid=grid, staggered_grid=staggered_grid, dt=dt), in_axes=(0, 0, 0, 0, 0, 0))
    # use the boris algorithm to update the velocities
    #newvx, newvy, newvz = boris_push(x, y, z, vx, vy, vz)

    newvx, newvy, newvz = boris(x, y, z, vx, vy, vz, q=q, m=m, E_interpolate=E_interpolate, B_interpolate=B_interpolate, grid=grid, staggered_grid=staggered_grid, dt=dt)
    # use the boris algorithm to update the velocities

    
    # w = jnp.zeros((3,3))
    # g = jnp.zeros((3,3))
    # # w and g are not used in the boris algorithm, but are required as arguments for the modified boris algorithm
    # newvx, newvy, newvz = modified_boris(q=q, m=m, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, E_interpolate=E_interpolate, B_interpolate=B_interpolate, w=w, g=g, grid=grid, staggered_grid=staggered_grid, dt=dt)
    
    particles.set_velocity(newvx, newvy, newvz)
    # set the new velocities of the particles
    return particles


@jit
def boris(x, y, z, vx, vy, vz, q, m, E_interpolate, B_interpolate, grid, staggered_grid, dt):
    """
    Perform the Boris push algorithm to update the velocity of a charged particle in an electromagnetic field.

    Parameters:
    x (float): x-coordinate of the particle position.
    y (float): y-coordinate of the particle position.
    z (float): z-coordinate of the particle position.
    vx (float): x-component of the particle velocity.
    vy (float): y-component of the particle velocity.
    vz (float): z-component of the particle velocity.
    q (float): Charge of the particle.
    m (float): Mass of the particle.
    E_interpolate (tuple of callables): Interpolators for the electric field components (Ex, Ey, Ez).
    B_interpolate (tuple of callables): Interpolators for the magnetic field components (Bx, By, Bz).
    grid (object): Grid object representing the simulation domain.
    staggered_grid (object): Staggered grid object for field interpolation.
    dt (float): Time step for the update.

    Returns:
    tuple: Updated velocity components (newvx, newvy, newvz).
    """

    Ex_interpolate, Ey_interpolate, Ez_interpolate = E_interpolate
    Bx_interpolate, By_interpolate, Bz_interpolate = B_interpolate
    # unpack the interpolators for the electric and magnetic fields
    points = jnp.stack([x, y, z], axis=-1)
    # create a 3x1 vector of the particle positions

    efield_atx = Ex_interpolate(points)
    efield_aty = Ey_interpolate(points)
    efield_atz = Ez_interpolate(points)
    # interpolate the electric field component arrays and calculate the e field at the particle positions

    bfield_atx = Bx_interpolate(points)
    bfield_aty = By_interpolate(points)
    bfield_atz = Bz_interpolate(points)
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


@jit
def modified_boris(q, m, x, y, z, vx, vy, vz, E_interpolate, B_interpolate, w, g, grid, staggered_grid, dt):
    """
    Perform a modified Boris algorithm to update the velocity of a charged particle in an electromagnetic field.
    Parameters:
    q (float): Charge of the particle.
    m (float): Mass of the particle.
    x (float): x-coordinate of the particle position.
    y (float): y-coordinate of the particle position.
    z (float): z-coordinate of the particle position.
    vx (float): x-component of the particle velocity.
    vy (float): y-component of the particle velocity.
    vz (float): z-component of the particle velocity.
    E_interpolate (tuple): Tuple of interpolating functions for the electric field components (Ex, Ey, Ez).
    B_interpolate (tuple): Tuple of interpolating functions for the magnetic field components (Bx, By, Bz).
    w (ndarray): Frequency matrix (3x3).
    g (ndarray): Gyrofrequency matrix (3x3).
    grid (ndarray): Grid for the field interpolation.
    staggered_grid (ndarray): Staggered grid for the field interpolation.
    dt (float): Time step for the integration.
    Returns:
    tuple: Updated velocity components (newvx, newvy, newvz).
    """

    Ex_interpolate, Ey_interpolate, Ez_interpolate = E_interpolate
    Bx_interpolate, By_interpolate, Bz_interpolate = B_interpolate
    # unpack the interpolators for the electric and magnetic fields
    points = jnp.stack([x, y, z], axis=-1)
    # create a 3x1 vector of the particle positions

    efield_atx = Ex_interpolate(points)
    efield_aty = Ey_interpolate(points)
    efield_atz = Ez_interpolate(points)
    # interpolate the electric field component arrays and calculate the e field at the particle positions

    bfield_atx = Bx_interpolate(points)
    bfield_aty = By_interpolate(points)
    bfield_atz = Bz_interpolate(points)
    # interpolate the magnetic field component arrays and calculate the b field at the particle positions

    wdotx = jnp.matmul(w, jnp.array([x, y, z]))
    # calculate the dot product of the frequency matrix and the position vector
    # w is a 3x3 matrix and x is a 3x1 vector, so the result is a 3x1 vector

    vxminus = vx + (dt/2)*( (q/m)*efield_atx - wdotx[0] )
    vyminus = vy + (dt/2)*( (q/m)*efield_aty - wdotx[1] )
    vzminus = vz + (dt/2)*( (q/m)*efield_atz - wdotx[2] )
    # calculate the v minus vector used in the boris push algorithm


    ##################### ROTATION MATRIX #####################
    bn = (dt/2)*(q/m)*jnp.array([bfield_atx, bfield_aty, bfield_atz])
    # calculate the b field vector

    iplusgamma =  jnp.matmul( jnp.identity(3) + (dt/2)*g, bn )[:,0]
    detL = jnp.linalg.det( jnp.identity(3) + (dt/2)*g )   +  jnp.dot( (dt/2)*g, iplusgamma )
    
    ematrix = jnp.zeros((3,3))
    ematrix = ematrix.at[0, 1].set(iplusgamma[2])
    ematrix = ematrix.at[0, 2].set(-iplusgamma[1])
    ematrix = ematrix.at[1, 0].set(-iplusgamma[2])
    ematrix = ematrix.at[1, 2].set(iplusgamma[0])
    ematrix = ematrix.at[2, 0].set(iplusgamma[1])
    ematrix = ematrix.at[2, 1].set(-iplusgamma[0])

    R = ( jnp.linalg.det((jnp.identity(3) + (dt/2)*g))*(jnp.identity(3) + (dt/2)*g)/(jnp.identity(3) - (dt/2)*g) \
        + ematrix + jnp.matmul(jnp.identity(3), (bn + bn) ) ) / detL
    # calculate the rotation matrix
    #############################################################


    vminus = jnp.array([vxminus, vyminus, vzminus])
    vplus = jnp.matmul(R, vminus)
    # calculate the v plus vector

    newvx = vplus[0] + (dt/2)*( (q/m)*efield_atx - wdotx[0] )
    newvy = vplus[1] + (dt/2)*( (q/m)*efield_aty - wdotx[1] )
    newvz = vplus[2] + (dt/2)*( (q/m)*efield_atz - wdotx[2] )
    # calculate the new velocity
    return newvx, newvy, newvz