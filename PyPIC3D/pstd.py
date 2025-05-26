from jax import jit
import jax
import jax.numpy as jnp
from functools import partial
import jaxdecomp
# import external libraries

def get_kmesh(k_array, dx, dy, dz):
    """
    Generate the wavevector components (kx, ky, kz) for a given 3D array using FFT frequencies.

    This function computes the wavevector components for a 3D array `k_array` based on its shape 
    and the spatial resolutions `dx`, `dy`, and `dz`. The wavevector components are reshaped to 
    match the structure of the input array.

    Parameters:
    -----------
    k_array : jax.numpy.ndarray
        A 3D array for which the wavevector components are to be computed.
    dx : float
        Spatial resolution along the x-axis.
    dy : float
        Spatial resolution along the y-axis.
    dz : float
        Spatial resolution along the z-axis.

    Returns:
    --------
    tuple of jax.numpy.ndarray
        A tuple containing the wavevector components (kx, ky, kz), each reshaped to match the 
        structure of the input array:
        - kx: Wavevector component along the x-axis, reshaped to [-1, 1, 1].
        - ky: Wavevector component along the y-axis, reshaped to [1, -1, 1].
        - kz: Wavevector component along the z-axis, reshaped to [1, 1, -1].

    Notes:
    ------
    - The FFT frequencies are computed using `jnp.fft.fftfreq` and scaled by `2 * pi`.
    - The input array structure is preserved using `jax.tree.structure` and `jax.tree.unflatten`.
    """

    # Compute the FFT frequencies for each axis
    kx = jnp.fft.fftfreq(k_array.shape[0], d=dy) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(k_array.shape[1], d=dz) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(k_array.shape[2], d=dx) * 2 * jnp.pi
    # FFT arrays are TRANSPOSED, so defining my k wavenumbers with the transposed resolution!

    k_array_structure = jax.tree.structure(k_array)
    kx = jax.tree.unflatten(k_array_structure, (kx,))
    ky = jax.tree.unflatten(k_array_structure, (ky,))
    kz = jax.tree.unflatten(k_array_structure, (kz,))
    # Unflatten the kx, ky, kz arrays to match the structure of the input array

    kx = kx.reshape([-1, 1, 1])
    ky = ky.reshape([1, -1, 1])
    kz = kz.reshape([1, 1, -1])
    # Reshape the wavevector components to match the shape of the input array

    return (kx, ky, kz)

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

    k = get_kmesh(krho, dx, dy, dz)
    # use the get_kmesh function to get the wavenumbers

    k2 = k[0]**2 + k[1]**2 + k[2]**2
    # calculate the squared wavenumber

    k2 = k2.at[0, 0, 0].set(1.0)
    phi = -krho / (eps*k2)
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

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    # get the grid spacing in each direction

    k = get_kmesh(xfft, dx, dy, dz)
    # use the get_kmesh function to get the wavenumbers

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

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    # get the grid resolution in each direction

    k = get_kmesh(xfft, dx, dy, dz)
    # use the get_kmesh function to get the wavenumbers

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

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    # get the grid resolution in each direction

    k = get_kmesh(field_fft, dx, dy, dz)
    # use the get_kmesh function to get the wavenumbers

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

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    # get the grid resolution in each direction

    k = get_kmesh(field_fft, dx, dy, dz)
    # use the get_kmesh function to get the wavenumbers

    lapl = -(k[0]**2 + k[1]**2 + k[2]**2) * field_fft
    # calculate the laplacian in Fourier space

    return jaxdecomp.fft.pifft3d(lapl).real