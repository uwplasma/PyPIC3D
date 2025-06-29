import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import stats

from PyPIC3D.fdtd import (
    centered_finite_difference_laplacian, centered_finite_difference_gradient, centered_finite_difference_divergence, centered_finite_difference_curl
)

def mse(x, y):
    """
    Compute the mean squared error (MSE) between two arrays.

    Parameters
    ----------
    x : array-like
        The first input array.
    y : array-like
        The second input array, must be broadcastable to the shape of `x`.

    Returns
    -------
    float
        The mean squared error between `x` and `y`.
    """
    return jnp.mean( (x-y)**2 )

def laplacian_comparison(nx):
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

    error = mse( laplacian[slicer], expected_solution[slicer] )
    # compute the mean squared error of the laplacian against the analytical solution

    return error,  dx

def gradient_comparison(nx):
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

    x_error = mse(gradx, expected_gradx)
    y_error = mse(grady, expected_grady)
    z_error = mse(gradz, expected_gradz)
    error = x_error + y_error + z_error
    # compute the mean squared error of the gradient function

    return error, dx


def divergence_comparison(nx):
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

    error = mse(divergence, expected_divergence)
    # compute the mean squared error of the divergence

    return error, dx


def curl_comparison(nx):
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

    error_x = mse(curlx, expected_curlx)
    error_y = mse(curly, expected_curly)
    error_z = mse(curlz, expected_curlz)
    error = error_x + error_y + error_z + error_z
    # compute the error in the numerical curl

    return error, dx

def convergence_test(func):
    """
    Computes the order of convergence for a numerical method by measuring the error at increasing grid resolutions.

    Args:
        func (callable): A function that takes an integer `nx` (number of grid points) as input and returns a tuple `(error, dx)`,
                         where `error` is the error at that resolution and `dx` is the grid spacing.

    Returns:
        float: The absolute value of the slope from a linear regression of log(error) vs. log(dx), representing the order of convergence.
    """

    nxs = [10*i + 30 for i in range(20)]
    # build list of different number of grid points

    errors = []
    dxs    = []
    # initialize the error and resolution lists

    for nx in nxs:
        error, dx = func(nx)
        errors.append( error )
        dxs.append( dx )
    # measure the error for increasing resolutions

    dxs = jnp.asarray(dxs)
    errors = jnp.asarray(errors)
    # convert the result lists to ndarrays

    res = stats.linregress( jnp.log(dxs), jnp.log(errors) )
    slope = jnp.abs( res.slope )
    # compute the order of the convergence using a line fit of the log(y)/log(x)

    return slope


if __name__ == "__main__":

    print("Convergence Testing of the Finite Difference Methods in PyPIC3D")

    ################### LAPLACIAN CONVERGENCE TEST #############################
    slope = convergence_test(laplacian_comparison)
    print(f"\nExpected Order of Laplacian Method: 2")
    print(f"Calculated Order of Laplacian Method: {slope}")
    print(f"Error in Order: {jnp.abs( 100 * (slope - 2) / 2 )} %")
    ############################################################################


    ################## GRADIENT CONVERGENCE TEST ###############################
    slope = convergence_test(gradient_comparison)
    print(f"\nExpected Order of Gradient Method: 2")
    print(f"Calculated Order of Gradient Method: {slope}")
    print(f"Error in Order: {jnp.abs(100 * (slope - 1) / 1)} %")
    ############################################################################


    ################## DIVERGENCE CONVERGENCE TEST #############################
    slope = convergence_test(divergence_comparison)
    print(f"\nExpected Order of Divergence Method: 2")
    print(f"Calculated Order of Divergence Method: {slope}")
    print(f"Error in Order: {jnp.abs(100 * (slope - 1) / 1)} %")
    ############################################################################


    ################# CURL CONVERGENCE TEST ####################################
    slope = convergence_test(curl_comparison)
    print(f"\nExpected Order of Curl Method: 2")
    print(f"Calculated Order of Curl Method: {slope}")
    print(f"Error in Order: {jnp.abs(100 * (slope - 1) / 1)} %")
    ############################################################################