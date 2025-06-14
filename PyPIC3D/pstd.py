from jax import jit
import jax
import jax.numpy as jnp
from functools import partial
# import external libraries

def create_k_mesh(nx, ny, nz, dx, dy, dz):
    """
    Create a mesh of wave numbers for FFT operations.

    Args:
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        nz (int): Number of grid points in the z-direction.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.

    Returns:
        tuple: Meshgrid of wave numbers (kx, ky, kz).
    """
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi
    return jnp.meshgrid(kx, ky, kz, indexing='ij')

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
    nx, ny, nz = krho.shape
    kx, ky, kz = create_k_mesh(nx, ny, nz, dx, dy, dz)
    k2 = kx**2 + ky**2 + kz**2
    k2 = k2.at[0, 0, 0].set(1.0)
    phi = krho / (eps*k2)
    phi = phi.at[0, 0, 0].set(0)
    phi = jnp.fft.ifftn(phi).real
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
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    nx, ny, nz = xfft.shape
    kx, ky, kz = create_k_mesh(nx, ny, nz, dx, dy, dz)
    div = 1j * kx * xfft + 1j * ky * yfft + 1j * kz * zfft
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
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    nx, ny, nz = xfft.shape
    kx, ky, kz = create_k_mesh(nx, ny, nz, dx, dy, dz)
    curlx = 1j * ky * zfft - 1j * kz * yfft
    curly = 1j * kz * xfft - 1j * kx * zfft
    curlz = 1j * kx * yfft - 1j * ky * xfft
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
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    nx, ny, nz = field_fft.shape
    kx, ky, kz = create_k_mesh(nx, ny, nz, dx, dy, dz)
    gradx = 1j * kx * field_fft
    grady = 1j * ky * field_fft
    gradz = 1j * kz * field_fft
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
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    nx, ny, nz = field_fft.shape
    kx, ky, kz = create_k_mesh(nx, ny, nz, dx, dy, dz)
    lapl = -(kx**2 + ky**2 + kz**2) * field_fft
    return jnp.fft.ifftn(lapl).real