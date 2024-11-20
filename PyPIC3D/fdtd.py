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

from PyPIC3D.utils import interpolate_and_stagger_field, interpolate_field, use_gpu_if_set
from PyPIC3D.boundaryconditions import apply_zero_boundary_condition


def centered_finite_difference_laplacian(field, dx, dy, dz, bc):
    """
    Calculates the Laplacian of a given field using centered finite difference and applies the specified boundary conditions.

    Parameters:
    - field: numpy.ndarray
        The input field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.
    - boundary_condition: str
        The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - numpy.ndarray
        The Laplacian of the field with the specified boundary conditions applied.
    """

    if bc == 'dirichlet':
        field = apply_zero_boundary_condition(field)
        # apply zero boundary condition at the edges of the field

    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field) / (dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field) / (dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field) / (dz*dz)
    # calculate the Laplacian of the field using centered finite difference

    if bc == 'neumann':
        x_comp = apply_zero_boundary_condition(x_comp)
        y_comp = apply_zero_boundary_condition(y_comp)
        z_comp = apply_zero_boundary_condition(z_comp)

    return x_comp + y_comp + z_comp

def centered_finite_difference_curl(field_x, field_y, field_z, dx, dy, dz, bc):
    """
    Computes the curl of a vector field using centered finite differencing and applies the specified boundary conditions.

    Parameters:
    - field_x: numpy.ndarray
        The x-component of the vector field.
    - field_y: numpy.ndarray
        The y-component of the vector field.
    - field_z: numpy.ndarray
        The z-component of the vector field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.
    - bc: str
        The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - tuple of numpy.ndarray
        The curl components (curl_x, curl_y, curl_z) of the vector field with the specified boundary conditions applied.
    """
    if bc == 'dirichlet':
        field_x = apply_zero_boundary_condition(field_x)
        field_y = apply_zero_boundary_condition(field_y)
        field_z = apply_zero_boundary_condition(field_z)

    curl_x = (jnp.roll(field_z, shift=-1, axis=1) - jnp.roll(field_z, shift=1, axis=1)) / (2 * dy) - \
                (jnp.roll(field_y, shift=-1, axis=2) - jnp.roll(field_y, shift=1, axis=2)) / (2 * dz)
    curl_y = (jnp.roll(field_x, shift=-1, axis=2) - jnp.roll(field_x, shift=1, axis=2)) / (2 * dz) - \
                (jnp.roll(field_z, shift=-1, axis=0) - jnp.roll(field_z, shift=1, axis=0)) / (2 * dx)
    curl_z = (jnp.roll(field_y, shift=-1, axis=0) - jnp.roll(field_y, shift=1, axis=0)) / (2 * dx) - \
                (jnp.roll(field_x, shift=-1, axis=1) - jnp.roll(field_x, shift=1, axis=1)) / (2 * dy)

    if bc == 'neumann':
        curl_x = apply_zero_boundary_condition(curl_x)
        curl_y = apply_zero_boundary_condition(curl_y)
        curl_z = apply_zero_boundary_condition(curl_z)

    return curl_x, curl_y, curl_z

def centered_finite_difference_gradient(field, dx, dy, dz, bc):
    """
    Computes the gradient of a scalar field using centered finite differencing and applies the specified boundary conditions.

    Parameters:
    - field: numpy.ndarray
        The input scalar field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.
    - bc: str
        The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - tuple of numpy.ndarray
        The gradient components (grad_x, grad_y, grad_z) of the scalar field with the specified boundary conditions applied.
    """
    if bc == 'dirichlet':
        field = apply_zero_boundary_condition(field)

    grad_x = (jnp.roll(field, shift=-1, axis=0) - jnp.roll(field, shift=1, axis=0)) / (2 * dx)
    grad_y = (jnp.roll(field, shift=-1, axis=1) - jnp.roll(field, shift=1, axis=1)) / (2 * dy)
    grad_z = (jnp.roll(field, shift=-1, axis=2) - jnp.roll(field, shift=1, axis=2)) / (2 * dz)

    if bc == 'neumann':
        grad_x = apply_zero_boundary_condition(grad_x)
        grad_y = apply_zero_boundary_condition(grad_y)
        grad_z = apply_zero_boundary_condition(grad_z)

    return grad_x, grad_y, grad_z

def centered_finite_difference_divergence(field_x, field_y, field_z, dx, dy, dz, bc):
    """
    Computes the divergence of a vector field using centered finite differencing and applies the specified boundary conditions.

    Parameters:
    - field_x: numpy.ndarray
        The x-component of the vector field.
    - field_y: numpy.ndarray
        The y-component of the vector field.
    - field_z: numpy.ndarray
        The z-component of the vector field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.
    - bc: str
        The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - numpy.ndarray
        The divergence of the vector field with the specified boundary conditions applied.
    """
    if bc == 'dirichlet':
        field_x = apply_zero_boundary_condition(field_x)
        field_y = apply_zero_boundary_condition(field_y)
        field_z = apply_zero_boundary_condition(field_z)

    div_x = (jnp.roll(field_x, shift=-1, axis=0) - jnp.roll(field_x, shift=1, axis=0)) / (2 * dx)
    div_y = (jnp.roll(field_y, shift=-1, axis=1) - jnp.roll(field_y, shift=1, axis=1)) / (2 * dy)
    div_z = (jnp.roll(field_z, shift=-1, axis=2) - jnp.roll(field_z, shift=1, axis=2)) / (2 * dz)

    if bc == 'neumann':
        div_x = apply_zero_boundary_condition(div_x)
        div_y = apply_zero_boundary_condition(div_y)
        div_z = apply_zero_boundary_condition(div_z)

    return div_x + div_y + div_z

def update_B(Bx, By, Bz, Ex, Ey, Ez, world, dt, boundary_condition):
    """
    Update the magnetic field components Bx, By, and Bz based on the electric field components Ex, Ey, and Ez.

    Parameters:
    - Bx (float): The x-component of the magnetic field.
    - By (float): The y-component of the magnetic field.
    - Bz (float): The z-component of the magnetic field.
    - Ex (float): The x-component of the electric field.
    - Ey (float): The y-component of the electric field.
    - Ez (float): The z-component of the electric field.
    - dx (float): The spacing in the x-direction.
    - dy (float): The spacing in the y-direction.
    - dz (float): The spacing in the z-direction.
    - dt (float): The time step.
    - boundary_condition (str): The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - Bx (float): The updated x-component of the magnetic field.
    - By (float): The updated y-component of the magnetic field.
    - Bz (float): The updated z-component of the magnetic field.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    curlx, curly, curlz = centered_finite_difference_curl(Ex, Ey, Ez, dx, dy, dz, boundary_condition)
    Bx = Bx - dt/2*curlx
    By = By - dt/2*curly
    Bz = Bz - dt/2*curlz
    # update the magnetic field
    return Bx, By, Bz

def update_E(Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, world, dt, constants, boundary_condition):
    """
    Update the electric field components Ex, Ey, and Ez based on the magnetic field components Bx, By, and Bz.

    Parameters:
    - Ex (float): The x-component of the electric field.
    - Ey (float): The y-component of the electric field.
    - Ez (float): The z-component of the electric field.
    - Bx (float): The x-component of the magnetic field.
    - By (float): The y-component of the magnetic field.
    - Bz (float): The z-component of the magnetic field.
    - dx (float): The spacing in the x-direction.
    - dy (float): The spacing in the y-direction.
    - dz (float): The spacing in the z-direction.
    - dt (float): The time step.
    - C (float): The Courant number.
    - boundary_condition (str): The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - Ex (float): The updated x-component of the electric field.
    - Ey (float): The updated y-component of the electric field.
    - Ez (float): The updated z-component of the electric field.
    """
    eps = constants['eps']
    C = constants['C']
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    curlx, curly, curlz = centered_finite_difference_curl(Bx, By, Bz, dx, dy, dz, boundary_condition)
    Ex = Ex + ( C**2 * curlx - 1/eps * Jx) * dt / 2
    Ey = Ey + ( C**2 * curly - 1/eps * Jy) * dt / 2
    Ez = Ez + ( C**2 * curlz - 1/eps * Jz) * dt / 2
    # update the electric field
    return Ex, Ey, Ez

@use_gpu_if_set
def particle_push(particles, Ex, Ey, Ez, Bx, By, Bz, E_grid, B_grid, dt, GPUs = False):
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
    newvx, newvy, newvz = boris(q, m, x, y, z, vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, E_grid, B_grid, dt)
    # use the boris algorithm to update the velocities
    particles.set_velocity(newvx, newvy, newvz)
    # set the new velocities of the particles
    return particles


@jit
def boris(q, m, x, y, z, vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, grid, staggered_grid, dt):
    """
    Perform the Boris algorithm to update the velocity of a charged particle in an electromagnetic field.

    Parameters:
    q (float): Charge of the particle.
    m (float): Mass of the particle.
    x (float): x-coordinate of the particle's position.
    y (float): y-coordinate of the particle's position.
    z (float): z-coordinate of the particle's position.
    vx (float): x-component of the particle's velocity.
    vy (float): y-component of the particle's velocity.
    vz (float): z-component of the particle's velocity.
    Ex (ndarray): x-component of the electric field array.
    Ey (ndarray): y-component of the electric field array.
    Ez (ndarray): z-component of the electric field array.
    Bx (ndarray): x-component of the magnetic field array.
    By (ndarray): y-component of the magnetic field array.
    Bz (ndarray): z-component of the magnetic field array.
    grid (ndarray): Grid for the electric field.
    staggered_grid (ndarray): Staggered grid for the magnetic field.
    dt (float): Time step for the update.

    Returns:
    tuple: Updated velocity components (newvx, newvy, newvz).
    """

    efield_atx = interpolate_field(Ex, grid, x, y, z)
    efield_aty = interpolate_field(Ey, grid, x, y, z)
    efield_atz = interpolate_field(Ez, grid, x, y, z)
    # interpolate the electric field component arrays and calculate the e field at the particle positions
    ygrid, xgrid, zgrid = grid
    ystagger, xstagger, zstagger = staggered_grid

    bfield_atx = interpolate_field(Bx, (xstagger, ygrid, zgrid), x, y, z)
    bfield_aty = interpolate_field(By, (xgrid, ystagger, zgrid), x, y, z)
    bfield_atz = interpolate_field(Bz, (xgrid, ygrid, zstagger), x, y, z)
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

def fdtd_current_correction(particles, Nx, Ny, Nz):
    Jx, Jy, Jz = jnp.zeros((Nx, Ny, Nz)), jnp.zeros((Nx, Ny, Nz)), jnp.zeros((Nx, Ny, Nz))
    # initialize the current arrays as 0
    for species in particles:
        q = species.get_charge()
        # get the charge of the species

        zeta1, zeta2, eta1, eta2, xi1, xi2 = species.get_subcell_position()
        # get the particle positions

        dx, dy, dz = species.get_resolution()

        deltax = zeta2 - zeta1
        deltay = eta2 - eta1
        deltaz = xi2 - xi1

        zetabar = 0.5*(zeta1 + zeta2)
        etabar = 0.5*(eta1 + eta2)
        xibar = 0.5*(xi1 + xi2)
        # compute the displacement variables from Buneman, Vilasenor 1991

        ix, iy, iz = species.get_index()

        if jnp.any(iy+1 > Ny) or jnp.any(ix+1 > Nx) or jnp.any(iz+1 > Nz):
            print("AHHHHHHHHHHHHHHHHHHHHHH")

        Jx = Jx.at[ix, iy+1, iz+1].add( q*(deltax*zetabar*xibar + deltax*deltay*deltaz/12))
        # compute the first x correction for charge conservation along i+1/2, j, k
        Jx = Jx.at[ix, iy, iz+1].add( q*(deltax*(1-etabar)*xibar - deltax*deltay*deltaz/12))
        # compute the second x correction for charge conservation along i+1/2, j, k+1
        Jx = Jx.at[ix, iy+1, iz].add( q*(deltax*etabar*(1-xibar) - deltax*deltay*deltaz/12))
        # compute the third x correction for charge conservation along i+1/2, j+1, k
        Jx = Jx.at[ix, iy, iz].add( q*(deltax*(1-etabar)*(1-xibar) + deltax*deltay*deltaz/12))
        # compute the fourth x correction for charge conservation along i+1/2, j, k

        Jy = Jy.at[ix+1, iy, iz+1].add( q*(deltay*zetabar*xibar + deltax*deltay*deltaz/12))
        # compute the first y correction for charge conservation along i, j+1/2, k
        Jy = Jy.at[ix+1, iy, iz].add( q*(deltay*(1-zetabar)*xibar - deltax*deltay*deltaz/12))
        # compute the second y correction for charge conservation along i+1, j+1/2, k
        Jy = Jy.at[ix, iy, iz+1].add( q*(deltay*zetabar*(1-xibar) - deltax*deltay*deltaz/12))
        # compute the third y correction for charge conservation along i, j+1/2, k+1
        Jy = Jy.at[ix, iy, iz].add( q*(deltay*(1-zetabar)*(1-xibar) + deltax*deltay*deltaz/12))
        # compute the fourth y correction for charge conservation along i, j+1/2, k

        Jz = Jz.at[ix+1, iy+1, iz].add( q*(deltaz*(1-zetabar)*etabar + deltax*deltay*deltaz/12))
        # compute the first z correction for charge conservation along i+1, j+1, k+1/2
        Jz = Jz.at[ix, iy+1, iz].add( q*(deltaz*(1-zetabar)*(1-etabar) - deltax*deltay*deltaz/12))
        # compute the second z correction for charge conservation along i, j+1, k+1/2
        Jz = Jz.at[ix+1, iy, iz].add( q*(deltaz*zetabar*(1-etabar) - deltax*deltay*deltaz/12))
        # compute the third z correction for charge conservation along i+1, j, k+1/2
        Jz = Jz.at[ix, iy, iz].add( q*(deltaz*(1-zetabar)*(1-etabar) + deltax*deltay*deltaz/12))
        # compute the fourth z correction for charge conservation along i, j, k+1/2

    return Jx, Jy, Jz
    # return the current corrections