import jax.numpy as jnp
from scipy import stats

from PyPIC3D.boris import create_trilinear_interpolator, create_quadratic_interpolator
from PyPIC3D.utils import convergence_test, mse

from functools import partial


def interpolation_wave_test(nx, interp_func):

    # Define a symmetric domain
    x_wind = 2.0 * jnp.pi
    y_wind = 2.0 * jnp.pi
    z_wind = 2.0 * jnp.pi

    # Create uniform grid
    x_grid = jnp.linspace(0, x_wind, nx)
    y_grid = jnp.linspace(0, y_wind, nx)
    z_grid = jnp.linspace(0, z_wind, nx)

    dx = x_wind / nx
    dy = y_wind / nx
    dz = z_wind / nx

    # Create meshgrid for field evaluation
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Define analytical test function: smooth trigonometric function
    # f(x,y,z) = sin(x) * cos(y) * sin(z)
    analytical_field = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)

    # Create the trilinear interpolator
    grid = (x_grid, y_grid, z_grid)
    interpolator = interp_func(analytical_field, grid)

    # Create test points that are offset from grid points
    # This tests the interpolation accuracy between grid points
    n_test = nx
    x_test = jnp.linspace(dx/3, x_wind - dx/3, n_test)
    y_test = jnp.linspace(dy/3, y_wind - dy/3, n_test)
    z_test = jnp.linspace(dz/3, z_wind - dz/3, n_test)

    X_test, Y_test, Z_test = jnp.meshgrid(x_test, y_test, z_test, indexing='ij')

    # Flatten for vectorized interpolation
    x_test_flat = X_test.flatten()
    y_test_flat = Y_test.flatten()
    z_test_flat = Z_test.flatten()

    # Compute analytical values at test points
    analytical_values = jnp.sin(x_test_flat) * jnp.cos(y_test_flat) * jnp.sin(z_test_flat)

    # Interpolate values at test points
    interpolated_values = interpolator(x_test_flat, y_test_flat, z_test_flat)

    # Compute mean squared error
    error = mse(interpolated_values, analytical_values)

    return error, dx


def interpolation_polynomial_test(nx, interp_func):

    # Define domain
    x_wind = 1.0
    y_wind = 1.0
    z_wind = 1.0

    # Create uniform grid
    x_grid = jnp.linspace(0, x_wind, nx)
    y_grid = jnp.linspace(0, y_wind, nx)
    z_grid = jnp.linspace(0, z_wind, nx)


    dx = x_wind / nx
    # create grid spacing

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Use a polynomial test function: f(x,y,z) = x²y + y²z + z²x
    # This has continuous derivatives and tests interpolation of curved surfaces
    analytical_field = X**2 * Y + Y**2 * Z + Z**2 * X

    # Create interpolator
    grid = (x_grid, y_grid, z_grid)
    interpolator = interp_func(analytical_field, grid)

    # Test at mid-points between grid points (maximum interpolation error)
    n_test = max(4, nx)  # Ensure we have enough test points
    x_test = jnp.linspace(dx/2, x_wind - dx/2, n_test)
    y_test = jnp.linspace(dx/2, y_wind - dx/2, n_test)
    z_test = jnp.linspace(dx/2, z_wind - dx/2, n_test)

    X_test, Y_test, Z_test = jnp.meshgrid(x_test, y_test, z_test, indexing='ij')

    # Flatten
    x_test_flat = X_test.flatten()
    y_test_flat = Y_test.flatten()
    z_test_flat = Z_test.flatten()

    # Analytical values
    analytical_values = (x_test_flat**2 * y_test_flat +
                        y_test_flat**2 * z_test_flat +
                        z_test_flat**2 * x_test_flat)

    # Interpolated values
    interpolated_values = interpolator(x_test_flat, y_test_flat, z_test_flat)

    # Compute error
    error = mse(interpolated_values, analytical_values)

    return error, dx


def interpolation_oscillatory_test(nx, interp_func):

    # Define domain
    x_wind = jnp.pi
    y_wind = jnp.pi
    z_wind = jnp.pi

    # Create grid
    x_grid = jnp.linspace(0, x_wind, nx)
    y_grid = jnp.linspace(0, y_wind, nx)
    z_grid = jnp.linspace(0, z_wind, nx)

    dx = x_wind / nx
    # create grid spacing

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # High-frequency test function: f(x,y,z) = cos(2x) * sin(3y) * cos(4z)
    analytical_field = jnp.cos(2*X) * jnp.sin(3*Y) * jnp.cos(4*Z)

    # Create interpolator
    grid = (x_grid, y_grid, z_grid)
    interpolator = interp_func(analytical_field, grid)

    # Test points slightly offset from grid
    n_test = max(6, nx)
    x_test = jnp.linspace(dx/4, x_wind - dx/4, n_test)
    y_test = jnp.linspace(dx/4, y_wind - dx/4, n_test)
    z_test = jnp.linspace(dx/4, z_wind - dx/4, n_test)

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
    interpolated_values = interpolator(x_test_flat, y_test_flat, z_test_flat)

    # Compute error
    error = mse(interpolated_values, analytical_values)

    return error, dx


if __name__ == "__main__":
    print("Convergence Testing of Trilinear Interpolation Method in PyPIC3D")

    ################### BASIC TRIGONOMETRIC TEST #############################
    print("\n1. Basic Trigonometric Function Test")
    print("   Function: f(x,y,z) = sin(x) * cos(y) * sin(z)")
    slope = convergence_test(partial(interpolation_wave_test, interp_func=create_trilinear_interpolator))
    print(f"   Expected Order: 2 (trilinear interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 2) / 2):.1f}%")
    ############################################################################

    ################### POLYNOMIAL TEST #######################################
    print("\n2. Polynomial Function Test")
    print("   Function: f(x,y,z) = x²y + y²z + z²x")
    slope = convergence_test(partial(interpolation_polynomial_test, interp_func=create_trilinear_interpolator))
    print(f"   Expected Order: 2 (trilinear interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 2) / 2):.1f}%")
    ############################################################################

    ################### HIGH-FREQUENCY TEST ###################################
    print("\n3. High-Frequency Oscillatory Test")
    print("   Function: f(x,y,z) = cos(2x) * sin(3y) * cos(4z)")
    slope = convergence_test(partial(interpolation_oscillatory_test, interp_func=create_trilinear_interpolator))
    print(f"   Expected Order: 2 (trilinear interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 2) / 2):.1f}%")
    ############################################################################

    print("#" * 70)

    print("\nConvergence Testing of Quadratic Interpolation Method in PyPIC3D")

    ################### BASIC TRIGONOMETRIC TEST #############################
    print("\n1. Basic Trigonometric Function Test")
    print("   Function: f(x,y,z) = sin(x) * cos(y) * sin(z)")
    slope = convergence_test(partial(interpolation_wave_test, interp_func=create_quadratic_interpolator))
    print(f"   Expected Order: 3 (quadratic interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 3) / 3):.1f}%")
    ############################################################################

    ################### POLYNOMIAL TEST #######################################
    print("\n2. Polynomial Function Test")
    print("   Function: f(x,y,z) = x²y + y²z + z²x")
    slope = convergence_test(partial(interpolation_polynomial_test, interp_func=create_quadratic_interpolator))
    print(f"   Expected Order: 3 (quadratic interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 3) / 3):.1f}%")
    ############################################################################

    ################### HIGH-FREQUENCY TEST ###################################
    print("\n3. High-Frequency Oscillatory Test")
    print("   Function: f(x,y,z) = cos(2x) * sin(3y) * cos(4z)")
    slope = convergence_test(partial(interpolation_oscillatory_test, interp_func=create_quadratic_interpolator))
    print(f"   Expected Order: 3 (quadratic interpolation)")
    print(f"   Calculated Order: {slope:.3f}")
    print(f"   Error in Order: {abs(100 * (slope - 3) / 3):.1f}%")
    ############################################################################