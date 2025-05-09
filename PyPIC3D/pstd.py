from jax import jit
import jax.numpy as jnp
from functools import partial
import jaxdecomp
# import external libraries


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
    #k = jaxdecomp.fft.fftfreq3d(krho, d=dx)
    # krho = jnp.fft.fftn(rho)

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
    phi = -krho / (eps*k2)
    phi = phi.at[0, 0, 0].set(0)
    # set the DC component to zero
    phi = jaxdecomp.fft.pifft3d(phi).real
    # phi = jnp.fft.ifftn(phi).real
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

    # k = jaxdecomp.fft.fftfreq3d(xfft)
    # # get the wavevector

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

    #k = jaxdecomp.fft.fftfreq3d(field_fft)
    # get the wavevector

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

    return jaxdecomp.fft.pifft3d(lapl).real