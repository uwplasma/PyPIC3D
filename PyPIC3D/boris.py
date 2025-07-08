import jax
from jax import jit
import jax.numpy as jnp


@jit
def particle_push(particles, E, B, grid, staggered_grid, dt, constants, periodic=True):
    """
    Updates the velocities of particles using the Boris algorithm.

    Args:
        particles (Particles): The particles to be updated.
        E (tuple): Electric field components (Ex, Ey, Ez).
        B (tuple): Magnetic field components (Bx, By, Bz).
        grid (Grid): The grid on which the fields are defined.
        staggered_grid (Grid): The staggered grid for field interpolation.
        dt (float): The time step for the update.
        constants (dict): Dictionary containing physical constants. Must include:
            - 'C': Speed of light in vacuum (m/s).
        periodic (tuple): A tuple of three booleans indicating whether each dimension is periodic.
                         Default is (True, True, True) for fully periodic domains.

    Returns:
        Particles: The particles with updated velocities.
    """
    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_backward_position()
    vx, vy, vz = particles.get_velocity()
    # get the charge, mass, position, and velocity of the particles

    shape_factor = particles.get_shape()
    # get the shape factor of the particles

    dx = grid[0][1] - grid[0][0]
    dy = grid[1][1] - grid[1][0]
    dz = grid[2][1] - grid[2][0]
    # calculate the grid spacing in each direction

    # ##################### ENERGY CONSERVING GRIDS ##########################
    # Ex_grid = grid[0] - dx, grid[1], grid[2]
    # Ey_grid = grid[0], grid[1] - dy, grid[2]
    # Ez_grid = grid[0], grid[1], grid[2] - dz
    # # create the grids for the electric field components

    # Bx_grid = staggered_grid[0], staggered_grid[1] - dy, staggered_grid[2] - dz
    # By_grid = staggered_grid[0] - dx, staggered_grid[1], staggered_grid[2] - dz
    # Bz_grid = staggered_grid[0] - dx, staggered_grid[1] - dy, staggered_grid[2]
    # # create the staggered grids for the magnetic field components
    # ########################################################################

    ################## MOMENTUM CONSERVING GRIDS ##########################
    Ex_grid = grid
    Ey_grid = grid
    Ez_grid = grid
    # create the grids for the electric field components
    Bx_grid = staggered_grid[0], staggered_grid[1], staggered_grid[2]
    By_grid = staggered_grid[0], staggered_grid[1], staggered_grid[2]
    Bz_grid = staggered_grid[0], staggered_grid[1], staggered_grid[2]
    # create the staggered grids for the magnetic field components

    Ex, Ey, Ez = E
    # unpack the electric field components
    # calculate the electric field at the particle positions using their specific grids
    efield_atx = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Ex, Ex_grid, periodic)(x, y, z),
        lambda _: create_quadratic_interpolator(Ex, Ex_grid, periodic)(x, y, z),
        operand=None
    )
    efield_aty = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Ey, Ey_grid, periodic)(x, y, z),
        lambda _: create_quadratic_interpolator(Ey, Ey_grid, periodic)(x, y, z),
        operand=None
    )
    efield_atz = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Ez, Ez_grid, periodic)(x, y, z),
        lambda _: create_quadratic_interpolator(Ez, Ez_grid, periodic)(x, y, z),
        operand=None
    )
    # unpack the magnetic field components
    Bx, By, Bz = B
    # calculate the magnetic field at the particle positions using their specific staggered grids
    bfield_atx = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Bx, Bx_grid, periodic)(x, y, z),
        lambda _: create_quadratic_interpolator(Bx, Bx_grid, periodic)(x, y, z),
        operand=None
    )
    bfield_aty = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(By, By_grid, periodic)(x, y, z),
        lambda _: create_quadratic_interpolator(By, By_grid, periodic)(x, y, z),
        operand=None
    )
    bfield_atz = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Bz, Bz_grid, periodic)(x, y, z),
        lambda _: create_quadratic_interpolator(Bz, Bz_grid, periodic)(x, y, z),
        operand=None
    )
    # calculate the magnetic field at the particle positions

    boris_vmap = jax.vmap(boris_single_particle, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None))
    newvx, newvy, newvz = boris_vmap(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants)
    # apply the Boris algorithm to update the velocities of the particles

    particles.set_velocity(newvx, newvy, newvz)
    # set the new velocities of the particles
    return particles

@jit
def boris_single_particle(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants):
    """
    Updates the velocity of a single particle using the Boris algorithm.
    Args:
        x (float): Initial x position of the particle.
        y (float): Initial y position of the particle.
        z (float): Initial z position of the particle.
        vx (float): Initial x component of the particle's velocity.
        vy (float): Initial y component of the particle's velocity.
        vz (float): Initial z component of the particle's velocity.
        efield_atx (float): x component of the electric field at the particle's position.
        efield_aty (float): y component of the electric field at the particle's position.
        efield_atz (float): z component of the electric field at the particle's position.
        bfield_atx (float): x component of the magnetic field at the particle's position.
        bfield_aty (float): y component of the magnetic field at the particle's position.
        bfield_atz (float): z component of the magnetic field at the particle's position.
        q (float): Charge of the particle.
        m (float): Mass of the particle.
        dt (float): Time step for the update.
    Returns:
        tuple: Updated velocity components (vx, vy, vz) of the particle.
    """

    v = jnp.array([vx, vy, vz])
    # convert v into an array

    vminus = v + q*dt/(2*m)*jnp.array([efield_atx, efield_aty, efield_atz])
    # get v minus vector

    t = q*dt/(2*m)*jnp.array([bfield_atx, bfield_aty, bfield_atz])
    # calculate the t vector
    vprime = vminus + jnp.cross(vminus, t)
    # calculate the v prime vector

    s = 2*t / (1 + t[0]**2 + t[1]**2 + t[2]**2)
    # calculate the s vector
    vplus = vminus + jnp.cross(vprime, s)
    # calculate the v plus vector

    newv = vplus + q*dt/(2*m)*jnp.array([efield_atx, efield_aty, efield_atz])
    # calculate the new velocity
    return newv[0], newv[1], newv[2]



@jit
def relativistic_boris_single_particle(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants):
    """
    Perform a single step of the relativistic Boris algorithm for a charged particle.

    This function calculates the updated velocity of a charged particle under the influence
    of electric and magnetic fields using the relativistic Boris algorithm. The algorithm
    ensures energy conservation and is widely used in particle-in-cell (PIC) simulations.

    Args:
        vx (float): Initial velocity of the particle in the x-direction (m/s).
        vy (float): Initial velocity of the particle in the y-direction (m/s).
        vz (float): Initial velocity of the particle in the z-direction (m/s).
        efield_atx (float): Electric field at the particle's position in the x-direction (V/m).
        efield_aty (float): Electric field at the particle's position in the y-direction (V/m).
        efield_atz (float): Electric field at the particle's position in the z-direction (V/m).
        bfield_atx (float): Magnetic field at the particle's position in the x-direction (T).
        bfield_aty (float): Magnetic field at the particle's position in the y-direction (T).
        bfield_atz (float): Magnetic field at the particle's position in the z-direction (T).
        q (float): Charge of the particle (Coulombs).
        m (float): Mass of the particle (kg).
        dt (float): Time step for the simulation (seconds).
        constants (dict): Dictionary containing physical constants. Must include:
            - 'C': Speed of light in vacuum (m/s).

    Returns:
        tuple: Updated velocity components of the particle in the x, y, and z directions (m/s).
    """


    C = constants['C']
    # speed of light

    v = jnp.array([vx, vy, vz])
    # convert v into an array

    gamma = jnp.sqrt( 1  + (  (v[0]**2 + v[1]**2 + v[2]**2) / C**2 ) )
    # define the gamma factor

    vminus = v * gamma + q*dt/(2*m)*jnp.array([efield_atx, efield_aty, efield_atz])
    # get v minus vector

    gamma_minus = jnp.sqrt( 1  + (  (vminus[0]**2 + vminus[1]**2 + vminus[2]**2) / C**2 ) )

    t = q*dt/(2*m)*jnp.array([bfield_atx, bfield_aty, bfield_atz]) / gamma_minus
    # calculate the t vector
    vprime = vminus + jnp.cross(vminus, t)
    # calculate the v prime vector

    s = 2*t / (1 + t[0]**2 + t[1]**2 + t[2]**2)
    # calculate the s vector
    vplus = vminus + jnp.cross(vprime, s)
    # calculate the v plus vector

    newv = vplus + q*dt/(2*m)*jnp.array([efield_atx, efield_aty, efield_atz])
    # calculate the new velocity

    new_gamma = jnp.sqrt( 1  +  (  (newv[0]**2 + newv[1]**2 + newv[2]**2) / C**2 ) )
    # define the new gamma factor

    return newv[0] / new_gamma, newv[1] / new_gamma, newv[2] / new_gamma



def create_trilinear_interpolator(field, grid, periodic=True):
    """
    Create a trilinear interpolation function for a given 3D field and grid.

    Handles cases where any dimension (nx, ny, nz) has only 1 grid point.
    In such cases, no interpolation is performed along that dimension.

    Args:
        field (ndarray): The 3D field to interpolate.
        grid (tuple): A tuple of three arrays representing the grid points in the x, y, and z directions.
        periodic (tuple): A tuple of three booleans indicating whether each dimension is periodic.
                         Default is (True, True, True) for fully periodic domains.

    Returns:
        function: A function that takes (x, y, z) coordinates and returns the interpolated values.
    """
    x_grid, y_grid, z_grid = grid

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]
    # calculate the grid spacing in each direction

    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    z_min, z_max = z_grid[0], z_grid[-1]
    # get grid bounds

    x_wind = x_max - x_min
    y_wind = y_max - y_min
    z_wind = z_max - z_min
    # get spatial widths of the grid

    Nx = len(x_grid)
    Ny = len(y_grid)
    Nz = len(z_grid)
    # get the number of grid points in each direction

    @jit
    def interpolator(x, y, z):

        # Handle periodic boundaries by wrapping coordinates
        if periodic:
            x = x_min + jnp.mod(x - x_min, x_wind)
            y = y_min + jnp.mod(y - y_min, y_wind)
            z = z_min + jnp.mod(z - z_min, z_wind)

        # Convert coordinates to grid indices (fractional)
        # Handle single-point dimensions specially
        xi = jnp.where(Nx == 1, 0.0, (x - x_min) / dx)
        yi = jnp.where(Ny == 1, 0.0, (y - y_min) / dy)
        zi = jnp.where(Nz == 1, 0.0, (z - z_min) / dz)

        # Find the lower-left-bottom corner indices
        x0 = jnp.where(Nx == 1, 0, jnp.floor(xi).astype(int))
        y0 = jnp.where(Ny == 1, 0, jnp.floor(yi).astype(int))
        z0 = jnp.where(Nz == 1, 0, jnp.floor(zi).astype(int))

        # Handle boundary conditions
        if periodic:
            x0 = jnp.where(Nx == 1, 0, jnp.mod(x0, Nx))
            y0 = jnp.where(Ny == 1, 0, jnp.mod(y0, Ny))
            z0 = jnp.where(Nz == 1, 0, jnp.mod(z0, Nz))
            x1 = jnp.where(Nx == 1, 0, jnp.mod(x0 + 1, Nx))
            y1 = jnp.where(Ny == 1, 0, jnp.mod(y0 + 1, Ny))
            z1 = jnp.where(Nz == 1, 0, jnp.mod(z0 + 1, Nz))
        else:
            x0 = jnp.where(Nx == 1, 0, jnp.clip(x0, 0, Nx - 2))
            y0 = jnp.where(Ny == 1, 0, jnp.clip(y0, 0, Ny - 2))
            z0 = jnp.where(Nz == 1, 0, jnp.clip(z0, 0, Nz - 2))
            x1 = jnp.where(Nx == 1, 0, jnp.clip(x0 + 1, 0, Nx - 1))
            y1 = jnp.where(Ny == 1, 0, jnp.clip(y0 + 1, 0, Ny - 1))
            z1 = jnp.where(Nz == 1, 0, jnp.clip(z0 + 1, 0, Nz - 1))

        # Calculate the fractional parts (weights)
        # For single-point dimensions, weight is meaningless so set to 0
        wx = jnp.where(Nx == 1, 0.0, xi - jnp.floor(xi))
        wy = jnp.where(Ny == 1, 0.0, yi - jnp.floor(yi))
        wz = jnp.where(Nz == 1, 0.0, zi - jnp.floor(zi))

        # Trilinear interpolation
        c000 = field[x0, y0, z0]
        c001 = field[x0, y0, z1]
        c010 = field[x0, y1, z0]
        c011 = field[x0, y1, z1]
        c100 = field[x1, y0, z0]
        c101 = field[x1, y0, z1]
        c110 = field[x1, y1, z0]
        c111 = field[x1, y1, z1]

        # Interpolate along x-axis first
        c00 = c000 * (1 - wx) + c100 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c11 = c011 * (1 - wx) + c111 * wx

        # Then interpolate along y-axis
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy

        # Finally interpolate along z-axis
        value = c0 * (1 - wz) + c1 * wz

        return value

    vmap_interpolator = jax.vmap(interpolator, in_axes=(0, 0, 0), out_axes=0)
    # vectorize the interpolator for batch processing

    return vmap_interpolator

def create_quadratic_interpolator(field, grid, periodic=True):
    """
    Create a quadratic interpolation function for a given 3D field and grid.

    Args:
        field (ndarray): The 3D field to interpolate.
        grid (tuple): A tuple of three arrays representing the grid points in the x, y, and z directions.
        periodic (tuple): A tuple of three booleans indicating whether each dimension is periodic.
                         Default is (True, True, True) for fully periodic domains.

    Returns:
        function: A function that takes (x, y, z) coordinates and returns the interpolated values.
    """
    x_grid, y_grid, z_grid = grid

    # Calculate grid spacing and bounds
    dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
    dy = y_grid[1] - y_grid[0] if len(y_grid) > 1 else 1.0
    dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1.0

    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    z_min, z_max = z_grid[0], z_grid[-1]

    # Calculate domain widths
    x_width = x_max - x_min
    y_width = y_max - y_min
    z_width = z_max - z_min

    Nx = len(x_grid)
    Ny = len(y_grid)
    Nz = len(z_grid)

    @jit
    def interpolator(x, y, z):
        # Handle periodic boundaries by wrapping coordinates
        if periodic:
            x = x_min + jnp.mod(x - x_min, x_width)
            y = y_min + jnp.mod(y - y_min, y_width)
            z = z_min + jnp.mod(z - z_min, z_width)

        # Handle each dimension separately to account for different grid sizes
        def get_indices_and_points(coord, grid_1d, coord_min, d_spacing):
            grid_len = len(grid_1d)

            # Convert coordinate to fractional grid index
            if grid_len == 1:
                frac_idx = 0.0
            else:
                frac_idx = (coord - coord_min) / d_spacing

            # Use JAX-compatible conditionals
            idx_base = jnp.floor(frac_idx).astype(int) if grid_len > 1 else 0

            # For grids with 3+ points, use quadratic interpolation
            idx_quad = jnp.clip(idx_base, 1, grid_len - 2)

            # For grids with 2 points, use linear interpolation
            idx_lin = jnp.clip(idx_base, 0, 0)

            # For single-point grids, use index 0
            idx_single = 0

            # Select appropriate index based on grid size
            idx = jnp.where(grid_len >= 3, idx_quad,
                   jnp.where(grid_len == 2, idx_lin, idx_single))

            # Handle periodic boundary conditions for indices
            if periodic and grid_len > 1:
                # For periodic boundaries, wrap the indices
                idx_m1 = jnp.mod(idx - 1, grid_len)
                idx_0 = jnp.mod(idx, grid_len)
                idx_p1 = jnp.mod(idx + 1, grid_len)

                p0 = jnp.where(grid_len >= 3, grid_1d[idx_m1],
                      jnp.where(grid_len == 2, grid_1d[0], grid_1d[0]))
                p1 = jnp.where(grid_len >= 2, grid_1d[idx_0], grid_1d[0])
                p2 = jnp.where(grid_len >= 3, grid_1d[idx_p1],
                      jnp.where(grid_len == 2, grid_1d[1], grid_1d[0]))
            else:
                # For non-periodic boundaries, use clipping
                # Get the three points for interpolation
                # For 3+ points: normal stencil
                # For 2 points: duplicate last point
                # For 1 point: duplicate the single point
                p0 = jnp.where(grid_len >= 3, grid_1d[jnp.clip(idx - 1, 0, grid_len - 1)],
                      jnp.where(grid_len == 2, grid_1d[0], grid_1d[0]))

                p1 = jnp.where(grid_len >= 2, grid_1d[jnp.clip(idx, 0, grid_len - 1)], grid_1d[0])

                p2 = jnp.where(grid_len >= 3, grid_1d[jnp.clip(idx + 1, 0, grid_len - 1)],
                      jnp.where(grid_len == 2, grid_1d[1], grid_1d[0]))

            return idx, p0, p1, p2

        x_idx, x0, x1, x2 = get_indices_and_points(x, x_grid, x_min, dx)
        y_idx, y0, y1, y2 = get_indices_and_points(y, y_grid, y_min, dy)
        z_idx, z0, z1, z2 = get_indices_and_points(z, z_grid, z_min, dz)

        def quadratic_weights(t, t0, t1, t2):
            # Handle degenerate cases (when points are equal)
            eps = 1e-12

            # Check for degenerate cases using JAX-compatible logic
            all_same = (jnp.abs(t0 - t1) < eps) & (jnp.abs(t1 - t2) < eps)
            t0_eq_t1 = (jnp.abs(t0 - t1) < eps) & (jnp.abs(t1 - t2) >= eps)
            t1_eq_t2 = (jnp.abs(t0 - t1) >= eps) & (jnp.abs(t1 - t2) < eps)

            # Standard quadratic weights
            denom0 = (t0 - t1) * (t0 - t2)
            denom1 = (t1 - t0) * (t1 - t2)
            denom2 = (t2 - t0) * (t2 - t1)

            # Avoid division by zero by adding small epsilon where needed
            denom0 = jnp.where(jnp.abs(denom0) < eps, eps, denom0)
            denom1 = jnp.where(jnp.abs(denom1) < eps, eps, denom1)
            denom2 = jnp.where(jnp.abs(denom2) < eps, eps, denom2)

            w0_standard = (t - t1) * (t - t2) / denom0
            w1_standard = (t - t0) * (t - t2) / denom1
            w2_standard = (t - t0) * (t - t1) / denom2

            # Linear interpolation weights for degenerate cases
            w_lin_01 = jnp.where(jnp.abs(t1 - t0) < eps, 0.0, (t - t0) / (t1 - t0))
            w_lin_12 = jnp.where(jnp.abs(t2 - t1) < eps, 0.0, (t - t1) / (t2 - t1))

            # Select weights based on degenerate cases
            w0 = jnp.where(all_same, 0.0,
                  jnp.where(t0_eq_t1, 0.0,
                   jnp.where(t1_eq_t2, 1.0 - w_lin_01, w0_standard)))

            w1 = jnp.where(all_same, 1.0,
                  jnp.where(t0_eq_t1, 1.0 - w_lin_12,
                   jnp.where(t1_eq_t2, w_lin_01, w1_standard)))

            w2 = jnp.where(all_same, 0.0,
                  jnp.where(t0_eq_t1, w_lin_12,
                   jnp.where(t1_eq_t2, 0.0, w2_standard)))

            return w0, w1, w2

        wx0, wx1, wx2 = quadratic_weights(x, x0, x1, x2)
        wy0, wy1, wy2 = quadratic_weights(y, y0, y1, y2)
        wz0, wz1, wz2 = quadratic_weights(z, z0, z1, z2)

        interpolated_value = 0.0
        for i, wx in enumerate([wx0, wx1, wx2]):
            for j, wy in enumerate([wy0, wy1, wy2]):
                for k, wz in enumerate([wz0, wz1, wz2]):
                    # Calculate array indices with proper bounds checking
                    # Use JAX-compatible conditionals

                    # X index calculation
                    if periodic and Nx > 1:
                        xi_quad = jnp.mod(x_idx - 1 + i, Nx)  # For 3+ points with periodic
                        xi_lin = jnp.mod(x_idx + i - 1, Nx)   # For 2 points with periodic
                    else:
                        xi_quad = x_idx - 1 + i  # For 3+ points
                        xi_lin = jnp.clip(x_idx + i - 1, 0, max(1, Nx - 1))  # For 2 points
                    xi_single = 0  # For 1 point

                    xi = jnp.where(Nx >= 3, xi_quad,
                          jnp.where(Nx == 2, xi_lin, xi_single))

                    # Y index calculation
                    if periodic and Ny > 1:
                        yi_quad = jnp.mod(y_idx - 1 + j, Ny)
                        yi_lin = jnp.mod(y_idx + j - 1, Ny)
                    else:
                        yi_quad = y_idx - 1 + j
                        yi_lin = jnp.clip(y_idx + j - 1, 0, max(1, Ny - 1))
                    yi_single = 0

                    yi = jnp.where(Ny >= 3, yi_quad,
                          jnp.where(Ny == 2, yi_lin, yi_single))

                    # Z index calculation
                    if periodic and Nz > 1:
                        zi_quad = jnp.mod(z_idx - 1 + k, Nz)
                        zi_lin = jnp.mod(z_idx + k - 1, Nz)
                    else:
                        zi_quad = z_idx - 1 + k
                        zi_lin = jnp.clip(z_idx + k - 1, 0, max(1, Nz - 1))
                    zi_single = 0

                    zi = jnp.where(Nz >= 3, zi_quad,
                          jnp.where(Nz == 2, zi_lin, zi_single))

                    # Final bounds check (only needed for non-periodic)
                    if not periodic:
                        xi = jnp.clip(xi, 0, field.shape[0] - 1)
                        yi = jnp.clip(yi, 0, field.shape[1] - 1)
                        zi = jnp.clip(zi, 0, field.shape[2] - 1)

                    interpolated_value += wx * wy * wz * field[xi, yi, zi]

        return interpolated_value

    vmap_interpolator = jax.vmap(interpolator, in_axes=(0, 0, 0), out_axes=0)
    return vmap_interpolator

@jit
def wrap_around(ix, size):
    """Wrap around index to ensure it is within bounds."""
    return jax.lax.cond(
        ix > size - 1,
        lambda _: ix - size,
        lambda _: ix,
        operand=None
    )