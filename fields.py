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

def debugprint(value):
    jax.debug.print('{x}', x=value)

def initialize_fields(Nx, Ny, Nz):
    """
    Initializes the electric and magnetic field arrays, as well as the electric potential and charge density arrays.

    Parameters:
    - Nx (int): Number of grid points in the x-direction.
    - Ny (int): Number of grid points in the y-direction.
    - Nz (int): Number of grid points in the z-direction.

    Returns:
    - Ex (ndarray): Electric field array in the x-direction.
    - Ey (ndarray): Electric field array in the y-direction.
    - Ez (ndarray): Electric field array in the z-direction.
    - Bx (ndarray): Magnetic field array in the x-direction.
    - By (ndarray): Magnetic field array in the y-direction.
    - Bz (ndarray): Magnetic field array in the z-direction.
    - phi (ndarray): Electric potential array.
    - rho (ndarray): Charge density array.
    """
    Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric field arrays as 0
    Bx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    By = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Bz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the magnetic field arrays as 0

    phi = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric potential and charge density arrays as 0

    return Ex, Ey, Ez, Bx, By, Bz, phi, rho



@jit
def periodic_laplacian(field, dx, dy, dz):
    """
    Calculates the Laplacian of a given field with Periodic boundary conditions.

    Parameters:
    - field: numpy.ndarray
        The input field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.

    Returns:
    - numpy.ndarray
        The Laplacian of the field.
    """
    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field)/(dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field)/(dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field)/(dz*dz)
    return x_comp + y_comp + z_comp

@jit
def neumann_laplacian(field, dx, dy, dz):
    """
    Calculates the Laplacian of a given field with Neumann boundary conditions.

    Parameters:
    - field: numpy.ndarray
        The input field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.
    - bc: str
        The boundary condition.

    Returns:
    - numpy.ndarray
        The Laplacian of the field.
    """


    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field)/(dx*dx) 
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field)/(dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field)/(dz*dz)

    x_comp = x_comp.at[0, :, :].set(0)
    x_comp = x_comp.at[-1, :, :].set(0)
    y_comp = y_comp .at[:, 0, :].set(0)
    y_comp = y_comp.at[:, -1, :].set(0)
    z_comp = z_comp.at[:, :, 0].set(0)
    z_comp = z_comp.at[:, :, -1].set(0)

    return x_comp + y_comp + z_comp
@jit
def dirichlet_laplacian(field, dx, dy, dz):
    """
    Calculates the Laplacian of a given field with Dirichlet boundary conditions.

    Parameters:
    - field: numpy.ndarray
        The input field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.
    - bc: str
        The boundary condition.

    Returns:
    - numpy.ndarray
        The Laplacian of the field.
    """
    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field)/(dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field)/(dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field)/(dz*dz)

    x_comp = x_comp.at[0, :, :].set((jnp.roll(field, shift=1, axis=0) - 2*field).at[0, :, :].get()/(dx*dx))
    x_comp = x_comp.at[-1, :, :].set((jnp.roll(field, shift=-1, axis=0) - 2*field).at[-1, :, :].get()/(dx*dx))
    y_comp = y_comp.at[:, 0, :].set((jnp.roll(field, shift=1, axis=1) - 2*field).at[:, 0, :].get()/(dy*dy))
    y_comp = y_comp.at[:, -1, :].set((jnp.roll(field, shift=-1, axis=1) - 2*field).at[:, -1, :].get()/(dy*dy))
    z_comp = z_comp.at[:, :, 0].set((jnp.roll(field, shift=1, axis=2) - 2*field).at[:, :, 0].get()/(dz*dz))
    z_comp = z_comp.at[:, :, -1].set((jnp.roll(field, shift=-1, axis=2) - 2*field).at[:, :, -1].get()/(dz*dz))

    return x_comp + y_comp + z_comp

@jit
def index_particles(particle, positions, ds):
    """
    Calculate the index of a particle in a given position array.

    Parameters:
    - particle: int
        The index of the particle.
    - positions (array-like): The position array containing the particle positions.
    - ds: float
        The grid spacing.

    Returns:
    - index: int
        The index of the particle in the position array, rounded down to the nearest integer.
    """
    return (positions.at[particle].get() / ds).astype(int)

@jit
def particle_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind):
    """
    Distributes the charge of a particle to the surrounding grid points.

    Parameters:
    q (float): Charge of the particle.
    x (float): x-coordinate of the particle.
    y (float): y-coordinate of the particle.
    z (float): z-coordinate of the particle.
    rho (ndarray): Charge density array.
    dx (float): Grid spacing in the x-direction.
    dy (float): Grid spacing in the y-direction.
    dz (float): Grid spacing in the z-direction.
    x_wind (float): Window in the x-direction.
    y_wind (float): Window in the y-direction.
    z_wind (float): Window in the z-direction.

    Returns:
    ndarray: Updated charge density array.
    """


    x0, y0, z0 = (x + x_wind/2).astype(int), (y + y_wind/2).astype(int), (z + z_wind/2).astype(int)
    deltax, deltay, deltaz = (x + x_wind/2) - x0, (y + y_wind/2) - y0, (z + z_wind/2) - z0
    # calculate the difference between x and its nearest grid point
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    # calculate the index of the next grid point

    wx = jnp.select( [x0 == 0, deltax == 0, deltax != 0], [0, 0, deltax/( x + x_wind/2 )] )
    wy = jnp.select( [y0 == 0, deltay == 0, deltay != 0], [0, 0, deltay/( y + y_wind/2 )] )
    wz = jnp.select( [z0 == 0, deltaz == 0, deltaz != 0], [0, 0, deltaz/( z + z_wind/2 )] )
    # calculate the weights for the surrounding grid points

    dv = dx*dy*dz
    # calculate the volume of each grid point

    rho = rho.at[x0, y0, z0].add( (q/dv)*(1 - wx)*(1 - wy)*(1 - wz), mode='drop' )
    rho = rho.at[x1, y0, z0].add( (q/dv)*wx*(1 - wy)*(1 - wz)      , mode='drop')
    rho = rho.at[x0, y1, z0].add( (q/dv)*(1 - wx)*wy*(1 - wz)      , mode='drop')
    rho = rho.at[x0, y0, z1].add( (q/dv)*(1 - wx)*(1 - wy)*wz      , mode='drop')
    # distribute the charge of the particle to the surrounding grid points

    return rho

@jit
def update_rho(Nparticles, particlex, particley, particlez, dx, dy, dz, q, x_wind, y_wind, z_wind, rho):
    """
    Update the charge density (rho) based on the positions of particles.
    Parameters:
    Nparticles (int): Number of particles.
    particlex (array-like): Array containing the x-coordinates of particles.
    particley (array-like): Array containing the y-coordinates of particles.
    particlez (array-like): Array containing the z-coordinates of particles.
    dx (float): Grid spacing in the x-direction.
    dy (float): Grid spacing in the y-direction.
    dz (float): Grid spacing in the z-direction.
    q (float): Charge of a single particle.
    x_wind (array-like): Window function in the x-direction.
    y_wind (array-like): Window function in the y-direction.
    z_wind (array-like): Window in the z-direction.
    rho (array-like): Initial charge density array to be updated.
    Returns:
    array-like: Updated charge density array.
    """


    def addto_rho(particle, rho):
        x = particlex.at[particle].get()
        y = particley.at[particle].get()
        z = particlez.at[particle].get()
        rho = particle_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind)
        return rho

    # def addto_rho(particle, rho):
    #     x = index_particles(particle, particlex, dx)
    #     y = index_particles(particle, particley, dy)
    #     z = index_particles(particle, particlez, dz)
    #     rho = rho.at[x, y, z].add( q / (dx*dy*dz) )
    #     return rho
    
    return jax.lax.fori_loop(0, Nparticles-1, addto_rho, rho )


@jit
def apply_M(Ax, M):
    """
    Apply the preconditioner

    Parameters:
    M (ndarray): The preconditioner.
    Ax (ndarray): The laplacian of x.

    Returns:
    ndarray: The inverse laplacian of the laplacian of the data.
    """

    M_Ax = jnp.einsum('ij,jlk -> ilk', M, Ax)
    M_Ay = jnp.einsum('ij, lik -> ljk', M, Ax)
    M_Az = jnp.einsum('ij, lki -> lkj', M, Ax)

    return (1/9)*(M_Ax + M_Ay + M_Az)



def conjugate_grad(A, b, x0, tol=1e-6, atol=0.0, maxiter=10000, M=None):

    #_dot = functools.partial(jnp.dot, precision=lax.Precision.HIGHEST)

    if M is None:
        noM = True
        M = lambda x: x
    else:
        noM = False
        #M = partial(_dot, M)
        M = partial(apply_M, M=M)

    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma.real if noM is True else _vdot_real_tree(r, r)
        return (rs > atol2) & (k < maxiter)


    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / _vdot_real_tree(p, Ap).astype(dtype)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_).astype(dtype)
        beta_ = gamma_ / gamma
        p_ = _add(z_, _mul(beta_, p))
        return x_, r_, gamma_, p_, k + 1


    r0 = _sub(b, A(x0))
    p0 = z0 = M(r0)
    dtype = jnp.result_type(*tree_leaves(p0))
    gamma0 = _vdot_real_tree(r0, z0).astype(dtype)
    initial_value = (x0, r0, gamma0, p0, 0)

    x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final





# @jit
# def cg_loop(p, rk_norm, x, residual, dx, dy, dz):
#     Ap = laplacian(p, dx, dy, dz)
#     pAp = jnp.tensordot(p, Ap, axes=3)
#     alpha = rk_norm / pAp
#     x = x + alpha * p
#     residual = residual - alpha * Ap
#     newrk_norm = jnp.linalg.norm(residual)
#     beta = newrk_norm / rk_norm
#     rk_norm = newrk_norm
#     p = beta * p + residual
#     return x, residual, rk_norm, p

# def conjugate_grad(b, x, dx, dy, dz, tol=1e-6, max_iter=10000):
#     residual  = b - laplacian(x, dx, dy, dz)
#     # r = b - Ax
#     p = residual
#     rk_norm = jnp.linalg.norm(residual)
#     for k in range(max_iter):
#         x, residual, rk_norm, p = cg_loop(p, rk_norm, x, residual, dx, dy, dz)
#         if rk_norm < tol:
#             return x
#     print(f"Did not converge, residual norm = {rk_norm}")
#     return x


def solve_poisson(rho, eps, dx, dy, dz, phi, bc='periodic', M = None):
    """
    Solve the Poisson equation for electrostatic potential.

    Parameters:
    - rho (ndarray): Charge density.
    - eps (float): Permittivity.
    - dx (float): Grid spacing in the x-direction.
    - dy (float): Grid spacing in the y-direction.
    - dz (float): Grid spacing in the z-direction.
    - phi (ndarray): Initial guess for the electrostatic potential.
    - bc (str): Boundary condition.
    - M (ndarray, optional): Preconditioner matrix for the conjugate gradient solver.

    Returns:
    - phi (ndarray): Solution to the Poisson equation.
    """

    if bc == 'periodic':
        lapl = functools.partial(periodic_laplacian, dx=dx, dy=dy, dz=dz)
    elif bc == 'dirichlet':
        lapl = functools.partial(dirichlet_laplacian, dx=dx, dy=dy, dz=dz)
    elif bc == 'neumann':
        lapl = functools.partial(neumann_laplacian, dx=dx, dy=dy, dz=dz)
    #phi = conjugate_grad(lapl, rho/eps, phi, tol=1e-6, maxiter=10000, M=M)
    #phi, exitcode = jax.scipy.sparse.linalg.cg(lapl, -rho/eps, phi, tol=1e-6, maxiter=20000, M=M)
    phi = conjugate_grad(lapl, -rho/eps, phi, tol=1e-6, maxiter=20000, M=M)
    return phi



def compute_pe(phi, rho, eps, dx, dy, dz, bc='periodic'):
    """
    Compute the relative percentage difference of the Poisson solver.

    Parameters:
    phi (ndarray): The potential field.
    rho (ndarray): The charge density.
    eps (float): The permittivity.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.
    bc (str): The boundary condition.

    Returns:
    float: The relative percentage difference of the Poisson solver.
    """
    if bc == 'periodic':
        x = periodic_laplacian(phi, dx, dy, dz)
    elif bc == 'dirichlet':
        x = dirichlet_laplacian(phi, dx, dy, dz)
    elif bc == 'neumann':
        x = neumann_laplacian(phi, dx, dy, dz)
    poisson_error = x + rho/eps
    index         = jnp.argmax(poisson_error)
    return 200 * jnp.abs( jnp.ravel(poisson_error)[index]) / ( jnp.abs(jnp.ravel(rho/eps)[index])+ jnp.abs(jnp.ravel(x)[index]) )
    # this method computes the relative percentage difference of poisson solver


def calculateE(N_electrons, electron_x, electron_y, electron_z, \
               N_ions, ion_x, ion_y, ion_z,                     \
               dx, dy, dz, q_e, q_i, rho, eps, phi, t, M, Nx, Ny, Nz, x_wind, y_wind, z_wind, bc, verbose, GPUs):
    """
                Calculates the electric field components (Ex, Ey, Ez), electric potential (phi), and charge density (rho) based on the given parameters.

                Parameters:
                - N_electrons (int): Number of electrons.
                - electron_x (array-like): x-coordinates of electrons.
                - electron_y (array-like): y-coordinates of electrons.
                - electron_z (array-like): z-coordinates of electrons.
                - N_ions (int): Number of ions.
                - ion_x (array-like): x-coordinates of ions.
                - ion_y (array-like): y-coordinates of ions.
                - ion_z (array-like): z-coordinates of ions.
                - dx (float): Grid spacing in the x-direction.
                - dy (float): Grid spacing in the y-direction.
                - dz (float): Grid spacing in the z-direction.
                - q_e (float): Charge of an electron.
                - q_i (float): Charge of an ion.
                - rho (array-like): Initial charge density.
                - eps (float): Permittivity of the medium.
                - phi (array-like): Initial electric potential.
                - t (int): Time step.
                - M (array-like): Matrix for solving Poisson's equation.
                - Nx (int): Number of grid points in the x-direction.
                - Ny (int): Number of grid points in the y-direction.
                - Nz (int): Number of grid points in the z-direction.
                - bc (str): Boundary condition.
                - verbose (bool): Whether to print additional information.
                - GPUs (bool): Whether to use GPUs for Poisson solver.

                Returns:
                - Ex (array-like): x-component of the electric field.
                - Ey (array-like): y-component of the electric field.
                - Ez (array-like): z-component of the electric field.
                - phi (array-like): Updated electric potential.
                - rho (array-like): Updated charge density.
                """
    
    if GPUs:
        with jax.default_device(jax.devices('gpu')[0]):
                rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
                # reset value of charge density
                rho = update_rho(N_electrons, electron_x, electron_y, electron_z, dx, dy, dz, q_e, x_wind, y_wind, z_wind, rho)
                rho = update_rho(N_ions, ion_x, ion_y, ion_z, dx, dy, dz, q_i, x_wind, y_wind, z_wind, rho)
                # update the charge density field
    else:
        rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
        # reset value of charge density
        rho = update_rho(N_electrons, electron_x, electron_y, electron_z, dx, dy, dz, q_e, x_wind, y_wind, z_wind, rho)
        rho = update_rho(N_ions, ion_x, ion_y, ion_z, dx, dy, dz, q_i, x_wind, y_wind, z_wind, rho)
        # update the charge density field

    if verbose: print(f"Calculating Charge Density, Max Value: {jnp.max(rho)}")
    # print the maximum value of the charge density

    if GPUs:
        with jax.default_device(jax.devices('gpu')[0]):
                if t == 0:
                    phi = solve_poisson(rho, eps, dx, dy, dz, phi=rho, bc=bc, M=None)
                else:
                    phi = solve_poisson(rho, eps, dx, dy, dz, phi=phi, bc=bc, M=M)
    else:
        if t == 0:
            phi = solve_poisson(rho, eps, dx, dy, dz, phi=rho, bc=bc, M=None)
        else:
            phi = solve_poisson(rho, eps, dx, dy, dz, phi=phi, bc=bc, M=M)



    if verbose: print(f"Calculating Electric Potential, Max Value: {jnp.max(phi)}")
    # print the maximum value of the electric potential
    if verbose: print( f'Poisson Relative Percent Difference: {compute_pe(phi, rho, eps, dx, dy, dz)}%')
    # Use conjugated gradients to calculate the electric potential from the charge density

    E_fields = jnp.gradient(phi)
    Ex       = -1*E_fields[0]
    Ey       = -1*E_fields[1]
    Ez       = -1*E_fields[2]
    # Calculate the E field using the gradient of the potential

    return Ex, Ey, Ez, phi, rho

@jit
def boris(q, Ex, Ey, Ez, Bx, By, Bz, x, y, z, vx, vy, vz, dt, m):
    """
    Perform Boris push algorithm to update the velocity of a charged particle in an electromagnetic field.

    Parameters:
    q (float): Charge of the particle.
    Ex (ndarray): Electric field component array in the x-direction.
    Ey (ndarray): Electric field component array in the y-direction.
    Ez (ndarray): Electric field component array in the z-direction.
    Bx (ndarray): Magnetic field component array in the x-direction.
    By (ndarray): Magnetic field component array in the y-direction.
    Bz (ndarray): Magnetic field component array in the z-direction.
    x (ndarray): Particle position array in the x-direction.
    y (ndarray): Particle position array in the y-direction.
    z (ndarray): Particle position array in the z-direction.
    vx (ndarray): Particle velocity array in the x-direction.
    vy (ndarray): Particle velocity array in the y-direction.
    vz (ndarray): Particle velocity array in the z-direction.
    dt (float): Time step size.
    m (float): Mass of the particle.

    Returns:
    tuple: Updated velocity of the particle in the x, y, and z directions.
    """
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

@jit
def curlx(yfield, zfield, dy, dz):
    """
    Calculate the curl of a vector field in the x-direction.

    Parameters:
    - yfield (ndarray): The y-component of the vector field.
    - zfield (ndarray): The z-component of the vector field.
    - dy (float): The spacing between y-values.
    - dz (float): The spacing between z-values.

    Returns:
    - ndarray: The x-component of the curl of the vector field.
    """
    delZdely = (jnp.roll(zfield, shift=1, axis=1) + jnp.roll(zfield, shift=-1, axis=1) - 2*zfield)/(dy*dy)
    delYdelz = (jnp.roll(yfield, shift=1, axis=2) + jnp.roll(yfield, shift=-1, axis=2) - 2*yfield)/(dz*dz)
    return delZdely - delYdelz

@jit
def curly(xfield, zfield, dx, dz):
    """
    Calculates the curl of a vector field in 2D.

    Parameters:
    - xfield (ndarray): The x-component of the vector field.
    - zfield (ndarray): The z-component of the vector field.
    - dx (float): The spacing between grid points in the x-direction.
    - dz (float): The spacing between grid points in the z-direction.

    Returns:
    - ndarray: The curl of the vector field.

    """
    delXdelz = (jnp.roll(xfield, shift=1, axis=2) + jnp.roll(xfield, shift=-1, axis=2) - 2*xfield)/(dz*dz)
    delZdelx = (jnp.roll(zfield, shift=1, axis=0) + jnp.roll(zfield, shift=-1, axis=0) - 2*zfield)/(dx*dx)
    return delXdelz - delZdelx

@jit
def curlz(yfield, xfield, dx, dy):
    """
    Calculate the curl of a 2D vector field in the z-direction.

    Parameters:
    - yfield (ndarray): The y-component of the vector field.
    - xfield (ndarray): The x-component of the vector field.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.

    Returns:
    - ndarray: The z-component of the curl of the vector field.
    """
    delYdelx = (jnp.roll(yfield, shift=1, axis=0) + jnp.roll(yfield, shift=-1, axis=0) - 2*yfield)/(dx*dx)
    delXdely = (jnp.roll(xfield, shift=1, axis=1) + jnp.roll(xfield, shift=-1, axis=1) - 2*xfield)/(dy*dy)
    return delYdelx - delXdely

@jit
def update_B(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt):
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

    Returns:
    - Bx (float): The updated x-component of the magnetic field.
    - By (float): The updated y-component of the magnetic field.
    - Bz (float): The updated z-component of the magnetic field.
    """
    Bx = Bx - dt*curlx(Ey, Ez, dy, dz)
    By = By - dt*curly(Ex, Ez, dx, dz)
    Bz = Bz - dt*curlz(Ex, Ey, dx, dy)
    return Bx, By, Bz


def probe(fieldx, fieldy, fieldz, x, y, z):
    """
    Probe the value of a vector field at a given point.

    Parameters:
    - fieldx (ndarray): The x-component of the vector field.
    - fieldy (ndarray): The y-component of the vector field.
    - fieldz (ndarray): The z-component of the vector field.
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.
    - z (float): The z-coordinate of the point.

    Returns:
    - tuple: The value of the vector field at the given point.
    """
    return fieldx.at[x, y, z].get(), fieldy.at[x, y, z].get(), fieldz.at[x, y, z].get()


def magnitude_probe(fieldx, fieldy, fieldz, x, y, z):
    """
    Probe the magnitude of a vector field at a given point.

    Parameters:
    - fieldx (ndarray): The x-component of the vector field.
    - fieldy (ndarray): The y-component of the vector field.
    - fieldz (ndarray): The z-component of the vector field.
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.
    - z (float): The z-coordinate of the point.

    Returns:
    - float: The magnitude of the vector field at the given point.
    """
    return jnp.sqrt(fieldx.at[x, y, z].get()**2 + fieldy.at[x, y, z].get()**2 + fieldz.at[x, y, z].get()**2)
@jit
def number_density(n, Nparticles, particlex, particley, particlez, dx, dy, dz, Nx, Ny, Nz):
    """
    Calculate the number density of particles at each grid point.

    Parameters:
    - n (array-like): The initial number density array.
    - Nparticles (int): The number of particles.
    - particlex (array-like): The x-coordinates of the particles.
    - particley (array-like): The y-coordinates of the particles.
    - particlez (array-like): The z-coordinates of the particles.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.

    Returns:
    - ndarray: The number density of particles at each grid point.
    """
    x_wind = (Nx * dx).astype(int)
    y_wind = (Ny * dy).astype(int)
    z_wind = (Nz * dz).astype(int)
    n = update_rho(Nparticles, particlex, particley, particlez, dx, dy, dz, 1, x_wind, y_wind, z_wind, n)

    return n

def freq(n, Nelectrons, ex, ey, ez, Nx, Ny, Nz, dx, dy, dz):
    ne = jnp.ravel(number_density(n, Nelectrons, ex, ey, ez, dx, dy, dz, Nx, Ny, Nz))
    # compute the number density of the electrons
    eps = 8.854e-12
    # permitivity of freespace
    q_e = -1.602e-19
    # charge of electron
    me = 9.1093837e-31 # Kg
    # mass of the electron
    c1 = q_e**2 / (eps*me)

    mask = jnp.where(  ne  > 0  )[0]
    # Calculate mean using the mask
    electron_density = jnp.mean(ne[mask])
    freq = jnp.sqrt( c1 * electron_density )
    return freq
# computes the average plasma frequency over the middle 75% of the world volume

def freq_probe(n, x, y, z, Nelectrons, ex, ey, ez, Nx, Ny, Nz, dx, dy, dz):
    ne = number_density(n, Nelectrons, ex, ey, ez, dx, dy, dz, Nx, Ny, Nz)
    # compute the number density of the electrons
    eps = 8.854e-12
    # permitivity of freespace
    q_e = -1.602e-19
    # charge of electron
    me = 9.1093837e-31 # Kg
    # mass of the electron
    xi, yi, zi = int(x/dx + Nx/2), int(y/dy + Ny/2), int(z/dz + Nz/2)
    # get the array spacings for x, y, and z
    c1 = q_e**2 / (eps*me)
    freq = jnp.sqrt( c1 * ne.at[xi,yi,zi].get() )    # calculate the plasma frequency at the array point: x, y, z
    return freq