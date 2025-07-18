import jax.numpy as jnp
from scipy import stats

from PyPIC3D.solvers.fdtd import (
    centered_finite_difference_laplacian, centered_finite_difference_gradient,
    centered_finite_difference_divergence, centered_finite_difference_curl
)

from PyPIC3D.solvers.pstd import (
    spectral_laplacian, spectral_gradient,
    spectral_divergence, spectral_curl
)

from PyPIC3D.utils import (
    convergence_test, mae
)

def fdtd_laplacian_comparison(nx):
    """
    Computes the mean squared error between the numerical and analytical Laplacian of a test function
    on a 3D periodic grid.

    The test function is phi(x, y, z) = x^2 + y^2 + z^2, whose Laplacian is analytically 6 everywhere.
    The function constructs a cubic grid of size `nx` in each dimension, computes the numerical Laplacian
    using a centered finite difference scheme with periodic boundary conditions, and compares it to the
    analytical solution (6) in the interior of the domain.

    Parameters
    ----------
    nx : int
        Number of grid points in each spatial dimension.

    Returns
    -------
    error : float
        Mean squared error between the numerical and analytical Laplacian in the interior of the grid.
    dx : float
        Grid spacing in the x-direction (equal to y and z spacing).
    """
    x_wind = 0.25
    y_wind = 0.25
    z_wind = 0.25
    # symmetric world volume

    x = jnp.linspace(-x_wind/2, x_wind/2, nx)
    y = jnp.linspace(-y_wind/2, y_wind/2, nx)
    z = jnp.linspace(-z_wind/2, z_wind/2, nx)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    # build x, y, z meshgrid

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx
    # spatial resolution

    phi = X**2 + Y**2 + Z**2
    expected_solution = 6 * jnp.ones_like(phi)
    # define a analytical phi and its expected solution

    laplacian =  centered_finite_difference_laplacian(phi, dx, dy, dz, 'periodic')
    # compute the numerical result

    slicer = (slice(1, -1), slice(1, -1), slice(1, -1))
    # slice for the solution comparison

    error = mae( laplacian[slicer], expected_solution[slicer] )
    # compute the mean squared error of the laplacian against the analytical solution

    return error,  dx

def fdtd_gradient_comparison(nx):
    """
    Computes the mean squared error between the analytical and numerical gradients
    of a 3D scalar field using centered finite differences with periodic boundary conditions.

    The scalar field is defined as:
        phi = sin(2πX) + sin(2πY) + sin(2πZ)
    over a symmetric cubic domain of size 0.25 in each direction, discretized into `nx` points per axis.

    Parameters
    ----------
    nx : int
        Number of grid points along each spatial dimension.

    Returns
    -------
    error : float
        The sum of mean squared errors between the numerical and analytical gradients
        in the x, y, and z directions.
    dx : float
        The spatial grid spacing along each axis.
    """
    x_wind = 0.25
    y_wind = 0.25
    z_wind = 0.25
    # symmetric world volume

    x = jnp.linspace(-x_wind/2, x_wind/2, nx)
    y = jnp.linspace(-y_wind/2, y_wind/2, nx)
    z = jnp.linspace(-z_wind/2, z_wind/2, nx)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    # build x, y, z meshgrid

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx
    # spatial resolution

    phi = jnp.sin(2 * jnp.pi * X) + jnp.sin(2 * jnp.pi * Y) + jnp.sin(2 * jnp.pi * Z)
    expected_gradx = 2 * jnp.pi * jnp.cos(2 * jnp.pi * X)
    expected_grady = 2 * jnp.pi * jnp.cos(2 * jnp.pi * Y)
    expected_gradz = 2 * jnp.pi * jnp.cos(2 * jnp.pi * Z)
    # build analytical phi with known solution

    gradx, grady, gradz = centered_finite_difference_gradient(phi, dx, dy, dz, 'periodic')
    # compute the numerical result

    x_error = mae(gradx, expected_gradx)
    y_error = mae(grady, expected_grady)
    z_error = mae(gradz, expected_gradz)
    error = (x_error + y_error + z_error) / 3
    # compute the mean squared error of the gradient function

    return error, dx


def fdtd_divergence_comparison(nx):
    """
    Computes the mean squared error between the numerical and analytical divergence
    of the vector field F = (X, Y, Z) on a 3D periodic grid.

    The function constructs a symmetric 3D grid of size `nx` in each dimension,
    defines the vector field F = (X, Y, Z), and computes its divergence using a
    centered finite difference scheme with periodic boundary conditions. The analytical
    divergence of F is 3 everywhere. The function returns the mean squared error
    between the computed and expected divergence, along with the grid spacing `dx`.

    Args:
        nx (int): Number of grid points in each spatial dimension.

    Returns:
        tuple:
            error (float): Mean squared error between numerical and analytical divergence.
            dx (float): Grid spacing in the x-direction.
    """
    x_wind = 0.25
    y_wind = 0.25
    z_wind = 0.25
    # symmetric world volume

    x = jnp.linspace(-x_wind/2, x_wind/2, nx)
    y = jnp.linspace(-y_wind/2, y_wind/2, nx)
    z = jnp.linspace(-z_wind/2, z_wind/2, nx)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    # build x, y, z meshgrid

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx
    # spatial resolution

    Fx = X
    Fy = Y
    Fz = Z
    # F = (X, Y, Z), div F = 3
    expected_divergence = 3 * jnp.ones_like(X)

    divergence = centered_finite_difference_divergence(Fx, Fy, Fz, dx, dy, dz, 'periodic')
    # compute the numerical divergence

    error = mae(divergence, expected_divergence)
    # compute the mean squared error of the divergence

    return error, dx


def fdtd_curl_comparison(nx):
    """
    Computes the mean squared error between the analytical and numerical curl of a test vector field
    on a 3D periodic grid, as a function of grid resolution.

    The test vector field is F = (-Y, X, 0), whose analytical curl is (0, 0, 2).
    The numerical curl is computed using a centered finite difference scheme with periodic boundaries.

    Args:
        nx (int): Number of grid points in each spatial dimension.

    Returns:
        tuple:
            error (float): The total mean squared error between the analytical and numerical curl components.
            dx (float): The grid spacing in each dimension.
    """

    x_wind = 0.25
    y_wind = 0.25
    z_wind = 0.25
    # symmetric world volume

    x = jnp.linspace(-x_wind/2, x_wind/2, nx)
    y = jnp.linspace(-y_wind/2, y_wind/2, nx)
    z = jnp.linspace(-z_wind/2, z_wind/2, nx)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    # build x, y, z meshgrid

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx
    # spatial resolution

    Fx = -1 * Y
    Fy = X
    Fz = jnp.zeros_like(Z)
    # Vector field: F = (-Y, X, 0), curl F = (0, 0, 2)

    expected_curlx = jnp.zeros_like(X)
    expected_curly = jnp.zeros_like(Y)
    expected_curlz = 2 * jnp.ones_like(Z)
    # compute the analytical curl

    curlx, curly, curlz = centered_finite_difference_curl(Fx, Fy, Fz, dx, dy, dz, 'periodic')
    # compute the numerical curl

    error_x = mae(curlx, expected_curlx)
    error_y = mae(curly, expected_curly)
    error_z = mae(curlz, expected_curlz)
    error = (error_x + error_y + error_z) / 3
    # compute the error in the numerical curl

    return error, dx


def pstd_laplacian_comparison(nx):
    """
    Computes the mean squared error between the numerical and analytical Laplacian of a test function
    on a 3D periodic grid.

    The test function is phi(x, y, z) = x^2 + y^2 + z^2, whose Laplacian is analytically 6 everywhere.
    The function constructs a cubic grid of size `nx` in each dimension, computes the numerical Laplacian
    using a centered finite difference scheme with periodic boundary conditions, and compares it to the
    analytical solution (6) in the interior of the domain.

    Parameters
    ----------
    nx : int
        Number of grid points in each spatial dimension.

    Returns
    -------
    error : float
        Mean squared error between the numerical and analytical Laplacian in the interior of the grid.
    dx : float
        Grid spacing in the x-direction (equal to y and z spacing).
    """
    x_wind = 0.25
    y_wind = 0.25
    z_wind = 0.25
    # symmetric world volume

    x = jnp.linspace(-x_wind/2, x_wind/2, nx)
    y = jnp.linspace(-y_wind/2, y_wind/2, nx)
    z = jnp.linspace(-z_wind/2, z_wind/2, nx)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    # build x, y, z meshgrid

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx
    # spatial resolution

    phi = jnp.sin(X + Y + Z)
    expected_solution = -3 * phi
    # define a analytical phi and its expected solution

    laplacian =  spectral_laplacian(phi, world={'dx': dx, 'dy': dy, 'dz': dz})
    # compute the numerical result

    slicer = (slice(10, -10), slice(10, -10), slice(10, -10))
    # slice for the solution comparison

    error = mae( laplacian[slicer], expected_solution[slicer] )
    # compute the mean squared error of the laplacian against the analytical solution

    return error,  dx


def pstd_gradient_comparison(nx):
    """
    Computes the mean squared error (MSE) between the numerical and analytical gradients
    of a scalar field using spectral methods.

    The function constructs a 3D meshgrid over a symmetric cubic domain, defines a scalar
    field `phi = sin(X + Y + Z)`, and computes its gradient both analytically and numerically.
    The numerical gradient is computed using the `spectral_gradient` function. The MSE between
    the numerical and analytical gradients is calculated for each component and summed.

    Args:
        nx (int): Number of grid points along each spatial dimension.

    Returns:
        tuple:
            error (float): The total mean squared error between the numerical and analytical gradients.
            dx (float): The spatial resolution along each axis.
    """
    x_wind = 0.25
    y_wind = 0.25
    z_wind = 0.25
    # symmetric world volume

    x = jnp.linspace(-x_wind/2, x_wind/2, nx)
    y = jnp.linspace(-y_wind/2, y_wind/2, nx)
    z = jnp.linspace(-z_wind/2, z_wind/2, nx)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    # build x, y, z meshgrid

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx
    # spatial resolution

    phi = jnp.sin(X + Y + Z)
    expected_gradx = jnp.cos(X + Y + Z)
    expected_grady = jnp.cos(X + Y + Z)
    expected_gradz = jnp.cos(X + Y + Z)
    # define a analytical phi and its expected solution

    gradx, grady, gradz = spectral_gradient(phi, world={'dx': dx, 'dy': dy, 'dz': dz})
    # compute the numerical result

    error_x = mae( gradx, expected_gradx )
    error_y = mae( grady, expected_grady )
    error_z = mae( gradz, expected_gradz )
    error = (error_x + error_y + error_z)/3
    # compute the mean squared error of the gradient against the analytical solution

    return error, dx


def pstd_divergence_comparison(nx):
    """
    Computes the mean squared error between the numerical and analytical divergence of a vector field.

    This function constructs a 3D meshgrid over a symmetric cubic domain, defines a vector field
    with components Fx, Fy, Fz = cos(X + Y + Z), and computes its divergence both analytically and
    numerically (using a spectral method). It then returns the mean squared error between the two
    divergence calculations, along with the spatial resolution.

    Args:
        nx (int): Number of grid points along each axis.

    Returns:
        tuple:
            error (float): Mean squared error between the numerical and analytical divergence.
            dx (float): Spatial resolution along each axis.
    """

    x_wind = 0.25
    y_wind = 0.25
    z_wind = 0.25
    # symmetric world volume

    x = jnp.linspace(-x_wind/2, x_wind/2, nx)
    y = jnp.linspace(-y_wind/2, y_wind/2, nx)
    z = jnp.linspace(-z_wind/2, z_wind/2, nx)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    # build x, y, z meshgrid

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx
    # spatial resolution

    Fx = jnp.cos(X + Y + Z)
    Fy = jnp.cos(X + Y + Z)
    Fz = jnp.cos(X + Y + Z)
    expected_div = -3 * jnp.sin(X + Y + Z)
    # define a analytical phi and its expected solution

    divF = spectral_divergence(Fx, Fy, Fz, world={'dx': dx, 'dy': dy, 'dz': dz})
    # compute the numerical result

    error = mae( divF, expected_div )
    # compute the mean squared error of the divergence against the analytical solution

    return error, dx

def pstd_curl_comparison(nx):
    """
    Computes the mean squared error (MSE) between the numerical and analytical curl of a vector field
    defined on a 3D grid using spectral methods.

    The vector field is defined as F = (sin(Y), sin(Z), sin(X)), and its analytical curl is
    (cos(Z) - cos(Y), cos(X) - cos(Z), cos(Y) - cos(X)). The function generates a symmetric 3D grid,
    computes the numerical curl using the `spectral_curl` function, and returns the total MSE along
    with the grid spacing.

    Args:
        nx (int): Number of grid points along each spatial dimension.

    Returns:
        tuple:
            error (float): The sum of mean squared errors for each component of the curl.
            dx (float): The spatial resolution (grid spacing) in the x-direction.
    """

    x_wind = 0.25
    y_wind = 0.25
    z_wind = 0.25
    # symmetric world volume

    x = jnp.linspace(-x_wind/2, x_wind/2, nx)
    y = jnp.linspace(-y_wind/2, y_wind/2, nx)
    z = jnp.linspace(-z_wind/2, z_wind/2, nx)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    # build x, y, z meshgrid

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx
    # spatial resolution

    Fx = jnp.sin(Y)
    Fy = jnp.sin(Z)
    Fz = jnp.sin(X)
    expected_curlx = jnp.cos(Z) - jnp.cos(Y)
    expected_curly = jnp.cos(X) - jnp.cos(Z)
    expected_curlz = jnp.cos(Y) - jnp.cos(X)
    # For curl test: F = (sin(Y), sin(Z), sin(X)), curl(F) = (cos(Z)-cos(Y), cos(X)-cos(Z), cos(Y)-cos(X))
    # define a analytical phi and its expected solution

    curlx, curly, curlz = spectral_curl(Fx, Fy, Fz, world={'dx': dx, 'dy': dy, 'dz': dz})
    # compute the numerical result

    error_x = mae( curlx, expected_curlx )
    error_y = mae( curly, expected_curly )
    error_z = mae( curlz, expected_curlz )
    error = (error_x + error_y + error_z) / 3
    # compute the mean squared error of the curl against the analytical solution

    return error, dx


if __name__ == "__main__":

    #################### FINITE DIFFERENCE CONVERGENCE TEST ##########################

    print("Convergence Testing of the Finite Difference Methods in PyPIC3D")

    ################### LAPLACIAN CONVERGENCE TEST #############################
    slope = convergence_test(fdtd_laplacian_comparison)
    print(f"\nExpected Order of Laplacian Method: 2")
    print(f"Calculated Order of Laplacian Method: {slope}")
    print(f"Error in Order: {jnp.abs( 100 * (slope - 2) / 2 )} %")
    ############################################################################


    ################## GRADIENT CONVERGENCE TEST ###############################
    slope = convergence_test(fdtd_gradient_comparison)
    print(f"\nExpected Order of Gradient Method: 2")
    print(f"Calculated Order of Gradient Method: {slope}")
    print(f"Error in Order: {jnp.abs(100 * (slope - 2) / 2)} %")
    ############################################################################


    ################## DIVERGENCE CONVERGENCE TEST #############################
    slope = convergence_test(fdtd_divergence_comparison)
    print(f"\nExpected Order of Divergence Method: 2")
    print(f"Calculated Order of Divergence Method: {slope}")
    print(f"Error in Order: {jnp.abs(100 * (slope - 2) / 2)} %")
    ############################################################################


    ################# CURL CONVERGENCE TEST ####################################
    slope = convergence_test(fdtd_curl_comparison)
    print(f"\nExpected Order of Curl Method: 2")
    print(f"Calculated Order of Curl Method: {slope}")
    print(f"Error in Order: {jnp.abs(100 * (slope - 2) / 2)} %")
    ############################################################################

    print("###################################################################")
    ############################################################################


    #################### SPECTRAL METHODS CONVERGENCE TEST ##########################
    print("\nConvergence Testing of the Spectral Methods in PyPIC3D")

    ################### LAPLACIAN CONVERGENCE TEST #############################
    slope = convergence_test(pstd_laplacian_comparison)
    print(f"\nExpected Order of Laplacian Method: 2")
    print(f"Calculated Order of Laplacian Method: {slope}")
    print(f"Error in Order: {jnp.abs( 100 * (slope - 2) / 2 )} %")
    ############################################################################

    ################## GRADIENT CONVERGENCE TEST ###############################
    slope = convergence_test(pstd_gradient_comparison)
    print(f"\nExpected Order of Gradient Method: 2")
    print(f"Calculated Order of Gradient Method: {slope}")
    print(f"Error in Order: {jnp.abs(100 * (slope - 2) / 2)} %")
    ############################################################################

    ################## DIVERGENCE CONVERGENCE TEST #############################
    slope = convergence_test(pstd_divergence_comparison)
    print(f"\nExpected Order of Divergence Method: 2")
    print(f"Calculated Order of Divergence Method: {slope}")
    print(f"Error in Order: {jnp.abs(100 * (slope - 2) / 2)} %")
    ############################################################################

    ################# CURL CONVERGENCE TEST ####################################
    slope = convergence_test(pstd_curl_comparison)
    print(f"\nExpected Order of Curl Method: 2")
    print(f"Calculated Order of Curl Method: {slope}")
    print(f"Error in Order: {jnp.abs(100 * (slope - 2) / 2)} %")
    ############################################################################

    ############################################################################