import jax
from jax import random
from jax import jit
from jax import lax
import jax.numpy as jnp
import functools
from functools import partial
import jaxdecomp
# import external libraries

from PyPIC3D.particle import particle_species
from PyPIC3D.J import compute_current_density
# import internal libraries


def check_nyquist_criterion(Ex, Ey, Ez, Bx, By, Bz, world):
    """
    Check if the E and B fields meet the Nyquist criterion.

    Args:
        Ex (ndarray): The electric field component in the x-direction.
        Ey (ndarray): The electric field component in the y-direction.
        Ez (ndarray): The electric field component in the z-direction.
        Bx (ndarray): The magnetic field component in the x-direction.
        By (ndarray): The magnetic field component in the y-direction.
        Bz (ndarray): The magnetic field component in the z-direction.
        world (dict): A dictionary containing the spatial resolution parameters.
            - 'dx' (float): Spatial resolution in the x-direction.
            - 'dy' (float): Spatial resolution in the y-direction.
            - 'dz' (float): Spatial resolution in the z-direction.

    Returns:
        bool: True if the fields meet the Nyquist criterion, False otherwise.
    """
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    nx, ny, nz = Ex.shape

    # Calculate the maximum wavenumber that can be resolved
    kx_max = jnp.pi / dx
    ky_max = jnp.pi / dy
    kz_max = jnp.pi / dz

    # Calculate the wavenumber components of the fields
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi

    # Check if the wavenumber components exceed the maximum wavenumber
    for field_name, field in {'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Bx': Bx, 'By': By, 'Bz': Bz}.items():
        kx_field = jnp.fft.fftn(field, axes=(0,))
        ky_field = jnp.fft.fftn(field, axes=(1,))
        kz_field = jnp.fft.fftn(field, axes=(2,))
        if jnp.any(jnp.abs(kx_field) > kx_max) or jnp.any(jnp.abs(ky_field) > ky_max) or jnp.any(jnp.abs(kz_field) > kz_max):
            print(f"Warning: The {field_name} field does not meet the Nyquist criterion. FFT may introduce aliasing.")


@partial(jit, static_argnums=(1, 2, 3))
def detect_gibbs_phenomenon(field, dx, dy, dz, threshold=0.1):
    """
    Detect the Gibbs phenomenon in a given field by checking for oscillations near discontinuities.

    Args:
        field (ndarray): The input field to check for Gibbs phenomenon.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        dz (float): The grid spacing in the z-direction.
        threshold (float): The threshold value for detecting significant oscillations.

    Returns:
        bool: True if Gibbs phenomenon is detected, False otherwise.
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

    Args:
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

    k = jaxdecomp.fft.fftfreq3d(Ex)
    # get the wavevector

    rho_fft = jaxdecomp.fft.pfft3d(rho)
    # calculate the Fourier transform of the charge density

    correction_mag = k[0]*Ex + k[1]*Ey + k[2]*Ez + 1j*rho_fft/constants['eps']
    # calculate the magnitude of the correction term
    kmag = jnp.sqrt(k[0]**2 + k[1]**2 + k[2]**2)
    kmag = kmag.at[0, 0, 0].set(1.0)
    # avoid division by zero

    x_correction = correction_mag * k[0] / kmag
    x_correction = x_correction.at[0, 0, 0].set(0)

    y_correction = correction_mag * k[1] / kmag
    y_correction = y_correction.at[0, 0, 0].set(0)

    z_correction = correction_mag * k[2] / kmag
    z_correction = z_correction.at[0, 0, 0].set(0)
    # calculate the correction term in Fourier space

    Ex_fft = jaxdecomp.fft.pfft3d(Ex)
    Ey_fft = jaxdecomp.fft.pfft3d(Ey)
    Ez_fft = jaxdecomp.fft.pfft3d(Ez)
    # calculate the Fourier transform of the electric field

    Ex = jaxdecomp.fft.pifft3d(Ex_fft - x_correction).real
    Ey = jaxdecomp.fft.pifft3d(Ey_fft - y_correction).real
    Ez = jaxdecomp.fft.pifft3d(Ez_fft - z_correction).real
    # apply the correction to the electric field

    return Ex, Ey, Ez

@jit
def spectral_poisson_solve(rho, constants, world):
    """
    Solve the Poisson equation for electrostatic potential using spectral method.

    Args:
        rho (ndarray): Charge density.
        eps (float): Permittivity.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.

    Returns:
        phi (ndarray): Solution to the Poisson equation.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    eps = constants['eps']

    krho = jaxdecomp.fft.pfft3d(rho)
    # fourier transform that charge density
    k = jaxdecomp.fft.fftfreq3d(krho)
    # get the wavenumbers
    k2 = k[0]**2 + k[1]**2 + k[2]**2
    # calculate the squared wavenumber
    k2 = k2.at[0, 0, 0].set(1.0)
    phi = -1 * krho / (eps*k2)
    phi = phi.at[0, 0, 0].set(0)
    # set the DC component to zero
    phi = jaxdecomp.fft.pifft3d(phi).real
    # calculate the inverse Fourier transform to obtain the electric potential
    return phi

@jit
def spectral_divergence(xfield, yfield, zfield, world):
    """
    Calculate the spectral divergence of a 3D vector field.

    This function computes the divergence of a vector field in the spectral domain
    using the Fast Fourier Transform (FFT). The input fields are assumed to be 
    periodic in all three dimensions.

    Args:
        xfield (ndarray): The x-component of the vector field.
        yfield (ndarray): The y-component of the vector field.
        zfield (ndarray): The z-component of the vector field.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        dz (float): The grid spacing in the z-direction.

    Returns:
        ndarray: The real part of the inverse FFT of the spectral divergence.
    """

    xfft = jaxdecomp.fft.pfft3d(xfield)
    yfft = jaxdecomp.fft.pfft3d(yfield)
    zfft = jaxdecomp.fft.pfft3d(zfield)
    # calculate the Fourier transform of the vector field

    k = jaxdecomp.fft.fftfreq3d(xfft)
    # get the wavevector

    div = -1j * k[0] * xfft + -1j * k[1] * yfft + -1j * k[2] * zfft
    # calculate the divergence in Fourier space

    return jaxdecomp.fft.pifft3d(div).real

@jit
def spectral_curl(xfield, yfield, zfield, world):
    """
    Compute the curl of a 3D vector field using spectral methods.

    Args:
        xfield (ndarray): The x-component of the vector field.
        yfield (ndarray): The y-component of the vector field.
        zfield (ndarray): The z-component of the vector field.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        dz (float): The grid spacing in the z-direction.

    Returns:
        tuple: A tuple containing the x, y, and z components of the curl of the vector field.
    """

    xfft = jaxdecomp.fft.pfft3d(xfield)
    yfft = jaxdecomp.fft.pfft3d(yfield)
    zfft = jaxdecomp.fft.pfft3d(zfield)
    # calculate the Fourier transform of the vector field

    k = jaxdecomp.fft.fftfreq3d(xfft)
    # get the wavevector

    curlx = -1j * k[1] * zfft - -1j * k[2] * yfft
    curly = -1j * k[2] * xfft - -1j * k[0] * zfft
    curlz = -1j * k[0] * yfft - -1j * k[1] * xfft
    # calculate the curl in Fourier space

    return jaxdecomp.fft.pifft3d(curlx).real, jaxdecomp.fft.pifft3d(curly).real, jaxdecomp.fft.pifft3d(curlz).real

@jit
def spectral_gradient(field, world):
    """
    Compute the gradient of a 3D scalar field using spectral methods.

    Args:
        field (ndarray): The input scalar field.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        dz (float): The grid spacing in the z-direction.

    Returns:
        tuple: A tuple containing the x, y, and z components of the gradient of the field.
    """

    field_fft = jaxdecomp.fft.pfft3d(field)
    # calculate the Fourier transform of the field

    k = jaxdecomp.fft.fftfreq3d(field_fft)
    # get the wavevector

    gradx = -1j * k[0] * field_fft
    grady = -1j * k[1] * field_fft
    gradz = -1j * k[2] * field_fft
    # calculate the gradient in Fourier space

    return jaxdecomp.fft.pifft3d(gradx).real, jaxdecomp.fft.pifft3d(grady).real, jaxdecomp.fft.pifft3d(gradz).real


@jit
def spectral_laplacian(field, world):
    """
    Calculates the Laplacian of a given field using spectral method. Assumes periodic boundary conditions.

    Args:
        field: numpy.ndarray
            The input field.
        dx: float
            The spacing between grid points in the x-direction.
        dy: float
            The spacing between grid points in the y-direction.
        dz: float
            The spacing between grid points in the z-direction.

    Returns:
        numpy.ndarray
            The Laplacian of the field.
    """
    field_fft = jaxdecomp.fft.pfft3d(field)
    # calculate the Fourier transform of the field

    k = jaxdecomp.fft.fftfreq3d(field_fft)
    # get the wavevector

    lapl = -(k[0]**2 + k[1]**2 + k[2]**2) * field_fft
    # calculate the laplacian in Fourier space

    return jaxdecomp.fft.pifft3d(lapl).real


#@partial(jit, static_argnums=(3, 4, 5, 6))
@jit
def solve_magnetic_vector_potential(Jx, Jy, Jz, dx, dy, dz, mu0):
    """
    Solve for the magnetic vector potential using the magnetostatic Laplacian equation in the Coulomb gauge.

    Args:
        Jx (ndarray): The x-component of the current density.
        Jy (ndarray): The y-component of the current density.
        Jz (ndarray): The z-component of the current density.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        dz (float): The grid spacing in the z-direction.
        mu0 (float): The permeability of free space.

    Returns:
        Ax (ndarray): The x-component of the magnetic vector potential.
        Ay (ndarray): The y-component of the magnetic vector potential.
        Az (ndarray): The z-component of the magnetic vector potential.
    """
    Nx, Ny, Nz = Jx.shape
    k = jaxdecomp.fft.fftfreq3d(Jx)
    # get the wavevector

    k2 = k[0]**2 + k[1]**2 + k[2]**2
    k2 = k2.at[0, 0, 0].set(1.0)
    # avoid division by zero

    Jx_fft = jaxdecomp.fft.pfft3d(Jx)
    Jy_fft = jaxdecomp.fft.pfft3d(Jy)
    Jz_fft = jaxdecomp.fft.pfft3d(Jz)
    # calculate the Fourier transform of the current density

    Ax_fft = mu0 * Jx_fft / k2
    Ay_fft = mu0 * Jy_fft / k2
    Az_fft = mu0 * Jz_fft / k2
    # solve for the magnetic vector potential in Fourier space

    Ax_fft = Ax_fft.at[0, 0, 0].set(0)
    Ay_fft = Ay_fft.at[0, 0, 0].set(0)
    Az_fft = Az_fft.at[0, 0, 0].set(0)
    # set the DC component to zero

    Ax = jaxdecomp.fft.pifft3d(Ax_fft).real
    Ay = jaxdecomp.fft.pifft3d(Ay_fft).real
    Az = jaxdecomp.fft.pifft3d(Az_fft).real
    # calculate the inverse Fourier transform to obtain the magnetic vector potential

    return Ax, Ay, Az

@partial(jit, static_argnums=(5))
def initialize_magnetic_field(particles, grid, staggered_grid, world, constants, GPUs):
    """
    Initialize the magnetic field using the current density from the list of particles.

    Args:
        particles (list): List of particle species.
        grid (Grid): The grid on which the fields are defined.
        staggered_grid (Grid): The staggered grid for field interpolation.
        world (dict): Dictionary containing the simulation world parameters.
        constants (dict): Dictionary containing physical constants, including 'mu0' (permeability).

    Returns:
        Bx (ndarray): The x-component of the magnetic field.
        By (ndarray): The y-component of the magnetic field.
        Bz (ndarray): The z-component of the magnetic field.
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
    Bx, By, Bz = spectral_curl(Ax, Ay, Az, world)

    return Bx, By, Bz

@jit
def spectral_marder_correction(Ex, Ey, Ez, rho, world, constants):
    """
    Apply the Marder correction to the electric field to suppress numerical instabilities.

    Args:
        E (ndarray): The electric field.
        rho (ndarray): The charge density.
        world (dict): Dictionary containing the simulation world parameters.
        constants (dict): Dictionary containing physical constants, including 'eps' (permittivity).
        d (float): The Marder damping parameter.

    Returns:
        E_corrected (ndarray): The corrected electric field.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    dt = world['dt']
    eps = constants['eps']

    # Compute the spectral divergence of the electric field
    divE = spectral_divergence(Ex, Ey, Ez, world)

    d = 1/(2*dt) * (dx**2 * dy**2 * dz**2) / (dx**2 + dy**2 + dz**2)
    # compute the diffusion parameter

    correction = d * (divE - rho/eps)
    # Compute the correction term

    gradx, grady, gradz = spectral_gradient( correction, world)
    # Compute the gradient of the correction term
    Ex_corrected = Ex + dt*gradx
    Ey_corrected = Ey + dt*grady
    Ez_corrected = Ez + dt*gradz
    # Apply the correction to the electric field

    return Ex_corrected, Ey_corrected, Ez_corrected