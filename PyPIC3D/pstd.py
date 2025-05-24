from jax import jit
import jax
import jax.numpy as jnp
from functools import partial
import jaxdecomp
# import external libraries

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

    krho = jnp.fft.fftn(rho)
    # fourier transform that charge density

    nx, ny, nz = rho.shape
    # get the number of grid points in each direction
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi
    # get the wavenumbers

    k = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create a meshgrid for the wavenumbers

    k2 = k[0]**2 + k[1]**2 + k[2]**2
    # calculate the squared wavenumber

    k2 = k2.at[0, 0, 0].set(1.0)
    # set the DC component to 1.0 to avoid division by zero
    phi = -krho / (eps*k2)
    # calculate the potential in Fourier space
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

    xfft = jnp.fft.fftn(xfield)
    yfft = jnp.fft.fftn(yfield)
    zfft = jnp.fft.fftn(zfield)
    # calculate the Fourier transform of the vector field

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    nx, ny, nz = xfield.shape
    # get the number of grid points in each direction
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi
    # get the wavenumbers

    k = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create a meshgrid for the wavenumbers

    div = -1j * k[0] * xfft + -1j * k[1] * yfft + -1j * k[2] * zfft
    # calculate the divergence in Fourier space

    return jnp.fft.ifftn(div).real

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

    xfft = jnp.fft.fftn(xfield)
    yfft = jnp.fft.fftn(yfield)
    zfft = jnp.fft.fftn(zfield)
    # calculate the Fourier transform of the vector field

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    nx, ny, nz = xfield.shape
    # get the number of grid points in each direction
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi
    # get the wavenumbers

    k = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create a meshgrid for the wavenumbers

    curlx = -1j * k[1] * zfft - -1j * k[2] * yfft
    curly = -1j * k[2] * xfft - -1j * k[0] * zfft
    curlz = -1j * k[0] * yfft - -1j * k[1] * xfft
    # calculate the curl in Fourier space

    return jnp.fft.ifftn(curlx).real, jnp.fft.ifftn(curly).real, jnp.fft.ifftn(curlz).real

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

    field_fft = jnp.fft.fftn(field)
    # calculate the Fourier transform of the field


    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    nx, ny, nz = field.shape
    # get the number of grid points in each direction

    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi
    # get the wavenumbers
    k = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create a meshgrid for the wavenumbers

    gradx = -1j * k[0] * field_fft
    grady = -1j * k[1] * field_fft
    gradz = -1j * k[2] * field_fft
    # calculate the gradient in Fourier space

    return jnp.fft.ifftn(gradx).real, jnp.fft.ifftn(grady).real, jnp.fft.ifftn(gradz).real

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

    field_fft = jnp.fft.fftn(field)
    # calculate the Fourier transform of the field

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    nx, ny, nz = field.shape
    # get the number of grid points in each direction

    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi
    # get the wavenumbers
    k = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create a meshgrid for the wavenumbers

    lapl = -(k[0]**2 + k[1]**2 + k[2]**2) * field_fft
    # calculate the laplacian in Fourier space

    return jnp.fft.ifftn(lapl).real