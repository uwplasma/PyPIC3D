import jax.numpy as jnp
from scipy import stats

from PyPIC3D.boris import interpolate_field_to_particles
from functools import partial


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
    

def interpolation_wave_test(nx, shape_factor):

    # Define a symmetric domain
    x_wind = 2.0 * jnp.pi
    y_wind = 2.0 * jnp.pi
    z_wind = 2.0 * jnp.pi

    # Create uniform grid
    x_grid = jnp.linspace(0, x_wind, nx, endpoint=False)
    y_grid = jnp.linspace(0, y_wind, nx, endpoint=False)
    z_grid = jnp.linspace(0, z_wind, nx, endpoint=False)

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx
    # Create meshgrid for field evaluation
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Define analytical test function: smooth trigonometric function
    # f(x,y,z) = sin(x) * cos(y) * sin(z)
    analytical_field = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)

    grid = (x_grid, y_grid, z_grid)

    # Create test points that are offset from grid points
    # This tests the interpolation accuracy between grid points
    n_test = nx
    x_test = jnp.linspace(dx/3, x_wind - dx/3, n_test, endpoint=False)
    y_test = jnp.linspace(dy/3, y_wind - dy/3, n_test, endpoint=False)
    z_test = jnp.linspace(dz/3, z_wind - dz/3, n_test, endpoint=False)

    X_test, Y_test, Z_test = jnp.meshgrid(x_test, y_test, z_test, indexing='ij')

    # Flatten for vectorized interpolation
    x_test_flat = X_test.flatten()
    y_test_flat = Y_test.flatten()
    z_test_flat = Z_test.flatten()

    # Compute analytical values at test points
    analytical_values = jnp.sin(x_test_flat) * jnp.cos(y_test_flat) * jnp.sin(z_test_flat)

    # Interpolate values at test points
    interpolated_values = interpolate_field_to_particles(
        analytical_field, x_test_flat, y_test_flat, z_test_flat, grid, shape_factor
    )

    error = jnp.sqrt( dx**3 * jnp.sum( (interpolated_values - analytical_values)**2 )    )
    # compute L2 error norm, accounting for grid spacing in 3D

    return error, dx


def interpolation_polynomial_test(nx, shape_factor):

    # Define domain
    x_wind = 1.0
    y_wind = 1.0
    z_wind = 1.0

    # Create uniform grid
    x_grid = jnp.linspace(0, x_wind, nx, endpoint=False)
    y_grid = jnp.linspace(0, y_wind, nx, endpoint=False)
    z_grid = jnp.linspace(0, z_wind, nx, endpoint=False)

    dx = x_wind / nx
    # create grid spacing

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Use a polynomial test function: f(x,y,z) = x³y + y³z + z³x
    # This has continuous derivatives and tests interpolation of curved surfaces
    analytical_field = X**3 * Y + Y**3 * Z + Z**3 * X

    grid = (x_grid, y_grid, z_grid)

    # Test at mid-points between grid points (maximum interpolation error)
    n_test = max(4, nx)  # Ensure we have enough test points
    x_test = jnp.linspace(dx/2, x_wind - dx/2, n_test, endpoint=False)
    y_test = jnp.linspace(dx/2, y_wind - dx/2, n_test, endpoint=False)
    z_test = jnp.linspace(dx/2, z_wind - dx/2, n_test, endpoint=False)

    X_test, Y_test, Z_test = jnp.meshgrid(x_test, y_test, z_test, indexing='ij')

    # Flatten
    x_test_flat = X_test.flatten()
    y_test_flat = Y_test.flatten()
    z_test_flat = Z_test.flatten()

    # Analytical values
    analytical_values = (x_test_flat**3 * y_test_flat +
                        y_test_flat**3 * z_test_flat +
                        z_test_flat**3 * x_test_flat)

    # Interpolated values
    interpolated_values = interpolate_field_to_particles(
        analytical_field, x_test_flat, y_test_flat, z_test_flat, grid, shape_factor
    )

    error = jnp.sqrt( dx**3 * jnp.sum( (interpolated_values - analytical_values)**2 )    )
    # compute L2 error norm, accounting for grid spacing in 3D

    return error, dx


def interpolation_oscillatory_test(nx, shape_factor):

    # Define domain
    x_wind = jnp.pi
    y_wind = jnp.pi
    z_wind = jnp.pi

    # Create grid
    x_grid = jnp.linspace(0, x_wind, nx, endpoint=False)
    y_grid = jnp.linspace(0, y_wind, nx, endpoint=False)
    z_grid = jnp.linspace(0, z_wind, nx, endpoint=False)  # Avoid endpoint to prevent duplicate point at z=0 and z=pi

    dx = x_wind / nx
    # create grid spacing

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # High-frequency test function: f(x,y,z) = cos(2x) * sin(3y) * cos(4z)
    analytical_field = jnp.cos(2*X) * jnp.sin(3*Y) * jnp.cos(4*Z)

    grid = (x_grid, y_grid, z_grid)

    # Test points slightly offset from grid
    n_test = max(6, nx)
    x_test = jnp.linspace(dx/4, x_wind - dx/4, n_test, endpoint=False)
    y_test = jnp.linspace(dx/4, y_wind - dx/4, n_test, endpoint=False)
    z_test = jnp.linspace(dx/4, z_wind - dx/4, n_test, endpoint=False)

    X_test, Y_test, Z_test = jnp.meshgrid(x_test, y_test, z_test, indexing='ij')

    # Flatten
    x_test_flat = X_test.flatten()
    y_test_flat = Y_test.flatten()
    z_test_flat = Z_test.flatten()

    # Analytical values
    analytical_values = (jnp.cos(2*x_test_flat) *
                        jnp.sin(3*y_test_flat) *
                        jnp.cos(4*z_test_flat))

    # Interpolated values
    interpolated_values = interpolate_field_to_particles(
        analytical_field, x_test_flat, y_test_flat, z_test_flat, grid, shape_factor
    )

    error = jnp.sqrt( dx**3 * jnp.sum( (interpolated_values - analytical_values)**2 )    )
    # compute L2 error norm, accounting for grid spacing in 3D

    return error, dx


if __name__ == "__main__":
    print("Convergence Testing of First Order Shape Factor Interpolation Method in PyPIC3D")

    ################### BASIC TRIGONOMETRIC TEST #############################
    print("\n1. Basic Trigonometric Function Test")
    print("   Function: f(x,y,z) = sin(x) * cos(y) * sin(z)")
    slope = convergence_test(partial(interpolation_wave_test, shape_factor=1))
    print(f"   Expected Order: 2 (trilinear interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 2) / 2):.1f}%")
    ############################################################################

    ################### POLYNOMIAL TEST #######################################
    print("\n2. Polynomial Function Test")
    print("   Function: f(x,y,z) = x³y + y³z + z³x")
    slope = convergence_test(partial(interpolation_polynomial_test, shape_factor=1))
    print(f"   Expected Order: 2 (trilinear interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 2) / 2):.1f}%")
    ############################################################################

    ################### HIGH-FREQUENCY TEST ###################################
    print("\n3. High-Frequency Oscillatory Test")
    print("   Function: f(x,y,z) = cos(2x) * sin(3y) * cos(4z)")
    slope = convergence_test(partial(interpolation_oscillatory_test, shape_factor=1))
    print(f"   Expected Order: 2 (trilinear interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 2) / 2):.1f}%")
    ############################################################################

    print("#" * 70)

    print("\nConvergence Testing of 2nd Order Shape Factor Interpolation Method in PyPIC3D")

    ################### BASIC TRIGONOMETRIC TEST #############################
    print("\n1. Basic Trigonometric Function Test")
    print("   Function: f(x,y,z) = sin(x) * cos(y) * sin(z)")
    slope = convergence_test(partial(interpolation_wave_test, shape_factor=2))
    print(f"   Expected Order: 2 (second-order particle-shape interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 2) / 2):.1f}%")
    ############################################################################

    ################### POLYNOMIAL TEST #######################################
    print("\n2. Polynomial Function Test")
    print("   Function: f(x,y,z) = x³y + y³z + z³x")
    slope = convergence_test(partial(interpolation_polynomial_test, shape_factor=2))
    print(f"   Expected Order: 2 (second-order particle-shape interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 2) / 2):.1f}%")
    ############################################################################

    ################### HIGH-FREQUENCY TEST ###################################
    print("\n3. High-Frequency Oscillatory Test")
    print("   Function: f(x,y,z) = cos(2x) * sin(3y) * cos(4z)")
    slope = convergence_test(partial(interpolation_oscillatory_test, shape_factor=2))
    print(f"   Expected Order: 2 (second-order particle-shape interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 2) / 2):.1f}%")
    ############################################################################
