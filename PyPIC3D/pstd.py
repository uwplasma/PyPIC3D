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

from PyPIC3D.utils import interpolate_and_stagger_field, interpolate_field, use_gpu_if_set
from PyPIC3D.particle import particle_species
from PyPIC3D.J import compute_current_density

@partial(jit, static_argnums=(1, 2, 3))
def detect_gibbs_phenomenon(field, dx, dy, dz, threshold=0.1):
    """
    Detect the Gibbs phenomenon in a given field by checking for oscillations near discontinuities.

    Parameters:
    - field (ndarray): The input field to check for Gibbs phenomenon.
    - threshold (float): The threshold value for detecting significant oscillations.

    Returns:
    - bool: True if Gibbs phenomenon is detected, False otherwise.
    """
    # Compute the gradient of the field
    grad = spectral_gradient(field, dx, dy, dz)

    # Compute the second derivative (Laplacian) of the field
    laplacian = spectral_laplacian(field, dx, dy, dz)

    # Detect regions where the gradient is high (indicating a discontinuity)
    discontinuities = jnp.abs(grad) > threshold

    # Check for oscillations near the discontinuities
    oscillations = jnp.abs(laplacian) > threshold

    # If oscillations are detected near discontinuities, Gibbs phenomenon is present
    gibbs_detected = jnp.any(discontinuities & oscillations)

    return gibbs_detected

@jit
def spectral_divergence_correction(Ex, Ey, Ez, rho, world, constants):
    """
    Corrects the divergence of the electric field in Fourier space.

    Parameters:
    Ex (ndarray): Electric field component in the x-direction.
    Ey (ndarray): Electric field component in the y-direction.
    Ez (ndarray): Electric field component in the z-direction.
    rho (ndarray): Charge density.
    dx (float): Grid spacing in the x-direction.
    dy (float): Grid spacing in the y-direction.
    dz (float): Grid spacing in the z-direction.
    dt (float): Time step.
    constants (dict): Dictionary containing physical constants, including 'eps' (permittivity).

    Returns:
    tuple: Corrected electric field components (Ex, Ey, Ez).
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    dt = world['dt']

    Nx, Ny, Nz = Ex.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers

    rho_fft = jnp.fft.fftn(rho)
    # calculate the Fourier transform of the charge density

    correction_mag = kx*Ex + ky*Ey + kz*Ez + 1j*rho_fft/constants['eps']
    # calculate the magnitude of the correction term
    kmag = jnp.sqrt(kx**2 + ky**2 + kz**2)
    kmag = kmag.at[0, 0, 0].set(1.0)
    # avoid division by zero

    x_correction = correction_mag * kx / kmag
    x_correction = x_correction.at[0, 0, 0].set(0)

    y_correction = correction_mag * ky / kmag
    y_correction = y_correction.at[0, 0, 0].set(0)

    z_correction = correction_mag * kz / kmag
    z_correction = z_correction.at[0, 0, 0].set(0)
    # calculate the correction term in Fourier space

    Ex_fft = jnp.fft.fftn(Ex)
    Ey_fft = jnp.fft.fftn(Ey)
    Ez_fft = jnp.fft.fftn(Ez)
    # calculate the Fourier transform of the electric field

    Ex = jnp.fft.ifftn(Ex_fft - x_correction).real
    Ey = jnp.fft.ifftn(Ey_fft - y_correction).real
    Ez = jnp.fft.ifftn(Ez_fft - z_correction).real
    # apply the correction to the electric field

    return Ex, Ey, Ez

@jit
def spectral_poisson_solve(rho, constants, world):
    """
    Solve the Poisson equation for electrostatic potential using spectral method.

    Parameters:
    - rho (ndarray): Charge density.
    - eps (float): Permittivity.
    - dx (float): Grid spacing in the x-direction.
    - dy (float): Grid spacing in the y-direction.
    - dz (float): Grid spacing in the z-direction.

    Returns:
    - phi (ndarray): Solution to the Poisson equation.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    eps = constants['eps']

    Nx, Ny, Nz = rho.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers
    k2 = kx**2 + ky**2 + kz**2
    # calculate the squared wavenumber
    k2 = k2.at[0, 0, 0].set(1.0)
    #k2 = jnp.where(k2 == 0, 1e-12, k2)
    # avoid division by zero
    phi = jnp.fft.fftn(rho) / (eps*k2)
    # calculate the Fourier transform of the charge density and divide by the permittivity and squared wavenumber
    phi = phi.at[0, 0, 0].set(0)
    # set the DC component to zero
    phi = jnp.fft.ifftn(phi).real
    # calculate the inverse Fourier transform to obtain the electric potential
    return phi

@jit
def spectral_divergence(xfield, yfield, zfield, world):
    """
    Calculate the spectral divergence of a 3D vector field.

    This function computes the divergence of a vector field in the spectral domain
    using the Fast Fourier Transform (FFT). The input fields are assumed to be 
    periodic in all three dimensions.

    Parameters:
    xfield (ndarray): The x-component of the vector field.
    yfield (ndarray): The y-component of the vector field.
    zfield (ndarray): The z-component of the vector field.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.

    Returns:
    ndarray: The real part of the inverse FFT of the spectral divergence.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    Nx, Ny, Nz = xfield.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers

    xfft = jnp.fft.fftn(xfield)
    yfft = jnp.fft.fftn(yfield)
    zfft = jnp.fft.fftn(zfield)
    # calculate the Fourier transform of the vector field

    div = (-1j)*kx*xfft + (-1j)*ky*yfft + (-1j)*kz*zfft
    # calculate the divergence of the vector field

    return jnp.fft.ifftn(div).real



@jit
def spectral_curl(xfield, yfield, zfield, world):
    """
    Compute the curl of a 3D vector field using spectral methods.

    Parameters:
    xfield (ndarray): The x-component of the vector field.
    yfield (ndarray): The y-component of the vector field.
    zfield (ndarray): The z-component of the vector field.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.

    Returns:
    tuple: A tuple containing the x, y, and z components of the curl of the vector field.
    """

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    Nx, Ny, Nz = xfield.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers

    xfft = jnp.fft.fftn(xfield)
    yfft = jnp.fft.fftn(yfield)
    zfft = jnp.fft.fftn(zfield)
    # calculate the Fourier transform of the vector field

    dfx_dy = jnp.fft.ifftn(-1j*ky*xfft).real
    dfx_dz = jnp.fft.ifftn(-1j*kz*xfft).real

    dfy_dx = jnp.fft.ifftn(-1j*kx*yfft).real
    dfy_dz = jnp.fft.ifftn(-1j*kz*yfft).real

    dfz_dx = jnp.fft.ifftn(-1j*kx*zfft).real
    dfz_dy = jnp.fft.ifftn(-1j*ky*zfft).real

    curlx = dfz_dy - dfy_dz
    curly = dfx_dz - dfz_dx
    curlz = dfy_dx - dfx_dy
    # calculate the curl of the vector field

    return curlx, curly, curlz

@jit
def spectral_gradient(field, world):
    """
    Compute the gradient of a 3D scalar field using spectral methods.

    Parameters:
    field (ndarray): The input scalar field.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.

    Returns:
    tuple: A tuple containing the x, y, and z components of the gradient of the field.
    """

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    Nx, Ny, Nz = field.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers

    field_fft = jnp.fft.fftn(field)
    # calculate the Fourier transform of the field

    gradx = -1j * kx * field_fft
    grady = -1j * ky * field_fft
    gradz = -1j * kz * field_fft
    # calculate the gradient in Fourier space

    return jnp.fft.ifftn(gradx).real, jnp.fft.ifftn(grady).real, jnp.fft.ifftn(gradz).real


@jit
def spectral_laplacian(field, world):
    """
    Calculates the Laplacian of a given field using spectral method. Assumes periodic boundary conditions.

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

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    Nx, Ny, Nz = field.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers
    lapl = -(kx**2 + ky**2 + kz**2) * jnp.fft.fftn(field)
    # calculate the laplacian in Fourier space
    return jnp.fft.ifftn(lapl).real

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
    newvx, newvy, newvz = boris(q, m, x, y, z, vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, grid, staggered_grid, dt)
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


@partial(jit, static_argnums=(3, 4, 5, 6))
def solve_magnetic_vector_potential(Jx, Jy, Jz, dx, dy, dz, mu0):
    """
    Solve for the magnetic vector potential using the magnetostatic Laplacian equation in the Coulomb gauge.

    Parameters:
    - Jx (ndarray): The x-component of the current density.
    - Jy (ndarray): The y-component of the current density.
    - Jz (ndarray): The z-component of the current density.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.
    - mu0 (float): The permeability of free space.

    Returns:
    - Ax (ndarray): The x-component of the magnetic vector potential.
    - Ay (ndarray): The y-component of the magnetic vector potential.
    - Az (ndarray): The z-component of the magnetic vector potential.
    """
    Nx, Ny, Nz = Jx.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2 = k2.at[0, 0, 0].set(1.0)
    # avoid division by zero

    Jx_fft = jnp.fft.fftn(Jx)
    Jy_fft = jnp.fft.fftn(Jy)
    Jz_fft = jnp.fft.fftn(Jz)
    # calculate the Fourier transform of the current density

    Ax_fft = mu0 * Jx_fft / k2
    Ay_fft = mu0 * Jy_fft / k2
    Az_fft = mu0 * Jz_fft / k2
    # solve for the magnetic vector potential in Fourier space

    Ax_fft = Ax_fft.at[0, 0, 0].set(0)
    Ay_fft = Ay_fft.at[0, 0, 0].set(0)
    Az_fft = Az_fft.at[0, 0, 0].set(0)
    # set the DC component to zero

    Ax = jnp.fft.ifftn(Ax_fft).real
    Ay = jnp.fft.ifftn(Ay_fft).real
    Az = jnp.fft.ifftn(Az_fft).real
    # calculate the inverse Fourier transform to obtain the magnetic vector potential

    return Ax, Ay, Az

@partial(jit, static_argnums=(5))
def initialize_magnetic_field(particles, grid, staggered_grid, world, constants, GPUs):
    """
    Initialize the magnetic field using the current density from the list of particles.

    Parameters:
    - particles (list): List of particle species.
    - grid (Grid): The grid on which the fields are defined.
    - staggered_grid (Grid): The staggered grid for field interpolation.
    - world (dict): Dictionary containing the simulation world parameters.
    - constants (dict): Dictionary containing physical constants, including 'mu0' (permeability).

    Returns:
    - Bx (ndarray): The x-component of the magnetic field.
    - By (ndarray): The y-component of the magnetic field.
    - Bz (ndarray): The z-component of the magnetic field.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    mu0 = constants['mu']

    # Initialize current density arrays
    Nx = grid[0].shape[0]
    Ny = grid[1].shape[0]
    Nz = grid[2].shape[0]
    Jx = jnp.zeros((Nx, Ny, Nz))
    Jy = jnp.zeros((Nx, Ny, Nz))
    Jz = jnp.zeros((Nx, Ny, Nz))

    # Compute the current density from the particles
    Jx, Jy, Jz = compute_current_density(particles, Jx, Jy, Jz, world, GPUs)

    # Solve for the magnetic vector potential
    Ax, Ay, Az = solve_magnetic_vector_potential(Jx, Jy, Jz, dx, dy, dz, mu0)

    # Compute the magnetic field from the vector potential
    Bx, By, Bz = spectral_curl(Ax, Ay, Az, dx, dy, dz)

    return Bx, By, Bz

@jit
def spectral_marder_correction(Ex, Ey, Ez, rho, world, constants, d):
    """
    Apply the Marder correction to the electric field to suppress numerical instabilities.

    Parameters:
    - E (ndarray): The electric field.
    - rho (ndarray): The charge density.
    - world (dict): Dictionary containing the simulation world parameters.
    - constants (dict): Dictionary containing physical constants, including 'eps' (permittivity).
    - d (float): The Marder damping parameter.

    Returns:
    - E_corrected (ndarray): The corrected electric field.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    eps = constants['eps']

    # Compute the spectral divergence of the electric field
    divE = spectral_divergence(Ex, Ey, Ez, dx, dy, dz)

    # Compute the correction term
    gradx, grady, gradz = spectral_gradient(divE - rho/eps, dx, dy, dz)
    Ex_corrected = Ex - d*gradx
    Ey_corrected = Ey - d*grady
    Ez_corrected = Ez - d*gradz
    # Apply the correction to the electric field

    return Ex_corrected, Ey_corrected, Ez_corrected