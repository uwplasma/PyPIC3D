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

from utils import interpolate_and_stagger_field
from charge_conservation import current_correction
from particle import particle_species

@jit
def spectral_poisson_solve(rho, eps, dx, dy, dz):
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
    Nx, Ny, Nz = rho.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers
    k2 = kx**2 + ky**2 + kz**2
    # calculate the squared wavenumber
    k2 = jnp.where(k2 == 0, 1e-6, k2)
    # avoid division by zero
    phi = -jnp.fft.fftn(rho) / (eps*k2)
    # calculate the Fourier transform of the charge density and divide by the permittivity and squared wavenumber
    phi = jnp.fft.ifftn(phi).real
    # calculate the inverse Fourier transform to obtain the electric potential
    return phi

@jit
def spectral_divergence(xfield, yfield, zfield, dx, dy, dz):
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

    div = 1j*kx*xfft + 1j*ky*yfft + 1j*kz*zfft
    # calculate the divergence of the vector field

    return jnp.fft.ifftn(div).real



@jit
def spectral_curl(xfield, yfield, zfield, dx, dy, dz):
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

    curlx = 1j*kz*yfft - 1j*ky*zfft
    curly = 1j*kx*zfft - 1j*kz*xfft
    curlz = 1j*ky*xfft - 1j*kx*yfft
    # calculate the curl of the vector field

    return jnp.fft.ifftn(curlx).real, jnp.fft.ifftn(curly).real, jnp.fft.ifftn(curlz).real


@jit
def spectralBsolve(grid, staggered_grid, Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt):
    """
    Solve the magnetic field equations using the spectral method and half leapfrog.

    Parameters:
    - Bx (ndarray): The x-component of the magnetic field.
    - By (ndarray): The y-component of the magnetic field.
    - Bz (ndarray): The z-component of the magnetic field.
    - Ex (ndarray): The x-component of the electric field.
    - Ey (ndarray): The y-component of the electric field.
    - Ez (ndarray): The z-component of the electric field.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.
    - dt (float): The time step.

    Returns:
    - Bx (ndarray): The updated x-component of the magnetic field.
    - By (ndarray): The updated y-component of the magnetic field.
    - Bz (ndarray): The updated z-component of the magnetic field.
    """
    curlx, curly, curlz = spectral_curl(Ex, Ey, Ez, dx, dy, dz)
    # calculate the curl of the electric field
    # curlx = interpolate_and_stagger_field(curlx, grid, staggered_grid)
    # curly = interpolate_and_stagger_field(curly, grid, staggered_grid)
    # curlz = interpolate_and_stagger_field(curlz, grid, staggered_grid)
    # interpolate the curl of the electric field and get the values at the cell faces
    Bx = Bx - dt/2*curlx
    By = By - dt/2*curly
    Bz = Bz - dt/2*curlz

    return Bx, By, Bz

@jit
def spectralEsolve(grid, staggered_grid, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, dx, dy, dz, dt, C, eps):
    """
    Solve the electric field equations using the spectral method and half leapfrog.

    Parameters:
    - Ex (ndarray): The x-component of the electric field.
    - Ey (ndarray): The y-component of the electric field.
    - Ez (ndarray): The z-component of the electric field.
    - Bx (ndarray): The x-component of the magnetic field.
    - By (ndarray): The y-component of the magnetic field.
    - Bz (ndarray): The z-component of the magnetic field.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.
    - dt (float): The time step.
    - q (float): The charge of the particle.
    - m (float): The mass of the particle.

    Returns:
    - Ex (ndarray): The updated x-component of the electric field.
    - Ey (ndarray): The updated y-component of the electric field.
    - Ez (ndarray): The updated z-component of the electric field.
    """
    curlx, curly, curlz = spectral_curl(Bx, By, Bz, dx, dy, dz)
    # calculate the curl of the magnetic field
    # curlx = interpolate_and_stagger_field(curlx, staggered_grid, grid)
    # curly = interpolate_and_stagger_field(curly, staggered_grid, grid)
    # curlz = interpolate_and_stagger_field(curlz, staggered_grid, grid)
    # interpolate the curl of the magnetic field and get the values at the cell centers
    Ex = Ex +  ( C**2 * curlx - 1/eps * Jx) * dt/2
    Ey = Ey +  ( C**2 * curly - 1/eps * Jy) * dt/2
    Ez = Ez +  ( C**2 * curlz - 1/eps * Jz) * dt/2

    return Ex, Ey, Ez

@jit
def spectral_laplacian(field, dx, dy, dz):
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
    Nx, Ny, Nz = field.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers
    lapl = -(kx**2 + ky**2 + kz**2) * jnp.fft.fftn(field)
    # calculate the laplacian in Fourier space
    return jnp.fft.ifftn(lapl).real

def psatd(Ex, Ey, Ez, Bx, By, Bz, q, m, dx, dy, dz, dt, x, y, z, vx, vy, vz):
    """
    Perform the PSATD algorithm to update the electric and magnetic fields.

    Parameters:
    - Ex (ndarray): The x-component of the electric field.
    - Ey (ndarray): The y-component of the electric field.
    - Ez (ndarray): The z-component of the electric field.
    - Bx (ndarray): The x-component of the magnetic field.
    - By (ndarray): The y-component of the magnetic field.
    - Bz (ndarray): The z-component of the magnetic field.
    - q (float): The charge of the particle.
    - m (float): The mass of the particle.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.
    - dt (float): The time step.
    - x (ndarray): The x-coordinates of the particles.
    - y (ndarray): The y-coordinates of the particles.
    - z (ndarray): The z-coordinates of the particles.
    - vx (ndarray): The x-component of the velocity of the particles.
    - vy (ndarray): The y-component of the velocity of the particles.
    - vz (ndarray): The z-component of the velocity of the particles.

    Returns:
    - Ex (ndarray): The updated x-component of the electric field.
    - Ey (ndarray): The updated y-component of the electric field.
    - Ez (ndarray): The updated z-component of the electric field.
    - Bx (ndarray): The updated x-component of the magnetic field.
    - By (ndarray): The updated y-component of the magnetic field.
    - Bz (ndarray): The updated z-component of the magnetic field.
    """
    # C = 1.0
    # # speed of light
    # Ex, Ey, Ez = spectralEsolve(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, dt, C)
    # # update the electric field
    # Bx, By, Bz = spectralBsolve(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt)
    # # update the magnetic field
    # return Ex, Ey, Ez, Bx, By, Bz

    # Build Fourier Mesh
    Nx, Ny, Nz = Ex.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers
    
    # FFT of the fields
    Ex_hat = jnp.fft.fftn(Ex)
    Ey_hat = jnp.fft.fftn(Ey)
    Ez_hat = jnp.fft.fftn(Ez)
    Bx_hat = jnp.fft.fftn(Bx)
    By_hat = jnp.fft.fftn(By)
    Bz_hat = jnp.fft.fftn(Bz)

    return