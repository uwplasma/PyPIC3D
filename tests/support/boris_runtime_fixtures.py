import unittest
import jax
import jax.numpy as jnp

from PyPIC3D.pusher.boris import boris_single_particle
from PyPIC3D.pusher.particle_interpolation import (
    galerkin_ordering,
    gather_shape_orders,
    interpolate_field_to_particles,
)
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.pusher.higuera_cary import higuera_cary_single_particle
from tests.kernel_fixtures import kernel_parameters


def mae(x, y):
    """
    Calculates the root mean squared error between two arrays.
    """
    return jnp.sqrt( jnp.mean( (x-y)**2 ) )


def interpolation_wave_test(nx, shape_factor):
    # Define a symmetric domain
    x_wind = 2.0 * jnp.pi
    y_wind = 2.0 * jnp.pi
    z_wind = 2.0 * jnp.pi

    # Create uniform grid
    x_grid = jnp.linspace(0, x_wind, nx)
    y_grid = jnp.linspace(0, y_wind, nx)
    z_grid = jnp.linspace(0, z_wind, nx)

    dx = x_wind / (nx - 1)
    dy = y_wind / (nx - 1)
    dz = z_wind / (nx - 1)

    # Create meshgrid for field evaluation
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Define analytical test function: smooth trigonometric function
    # f(x,y,z) = sin(x) * cos(y) * sin(z)
    analytical_field = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)

    grid = (x_grid, y_grid, z_grid)

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
    interpolated_values = interpolate_field_to_particles(
        analytical_field, x_test_flat, y_test_flat, z_test_flat, grid, shape_factor
    )

    # Compute mean squared error
    error = mae(interpolated_values, analytical_values)

    return error, dx


def interpolation_polynomial_test(nx, shape_factor):

    # Define domain
    x_wind = 1.0
    y_wind = 1.0
    z_wind = 1.0

    # Create uniform grid
    x_grid = jnp.linspace(0, x_wind, nx)
    y_grid = jnp.linspace(0, y_wind, nx)
    z_grid = jnp.linspace(0, z_wind, nx)

    dx = x_wind / (nx - 1)
    # create grid spacing

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Use a polynomial test function: f(x,y,z) = x³y + y³z + z³x
    # This has continuous derivatives and tests interpolation of curved surfaces
    analytical_field = X**3 * Y + Y**3 * Z + Z**3 * X

    grid = (x_grid, y_grid, z_grid)

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
    analytical_values = (x_test_flat**3 * y_test_flat +
                        y_test_flat**3 * z_test_flat +
                        z_test_flat**3 * x_test_flat)

    # Interpolated values
    interpolated_values = interpolate_field_to_particles(
        analytical_field, x_test_flat, y_test_flat, z_test_flat, grid, shape_factor
    )

    # Compute error
    error = mae(interpolated_values, analytical_values)

    return error, dx


def interpolation_oscillatory_test(nx, shape_factor):

    # Define domain
    x_wind = jnp.pi
    y_wind = jnp.pi
    z_wind = jnp.pi

    # Create grid
    x_grid = jnp.linspace(0, x_wind, nx)
    y_grid = jnp.linspace(0, y_wind, nx)
    z_grid = jnp.linspace(0, z_wind, nx)

    dx = x_wind / (nx - 1)
    # create grid spacing

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # High-frequency test function: f(x,y,z) = cos(2x) * sin(3y) * cos(4z)
    analytical_field = jnp.cos(2*X) * jnp.sin(3*Y) * jnp.cos(4*Z)

    grid = (x_grid, y_grid, z_grid)

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
    interpolated_values = interpolate_field_to_particles(
        analytical_field, x_test_flat, y_test_flat, z_test_flat, grid, shape_factor
    )

    # Compute error
    error = mae(interpolated_values, analytical_values)

    return error, dx
