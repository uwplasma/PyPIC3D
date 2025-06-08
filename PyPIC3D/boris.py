import jax
from jax import jit
import jax.numpy as jnp


@jit
def particle_push(particles, E, B, grid, staggered_grid, dt, constants):
    """
    Updates the velocities of particles using the Boris algorithm.

    Args:
        particles (Particles): The particles to be updated.
        Ex (array-like): Electric field component in the x-direction.
        Ey (array-like): Electric field component in the y-direction.
        Ez (array-like): Electric field component in the z-direction.
        Bx (array-like): Magnetic field component in the x-direction.
        By (array-like): Magnetic field component in the y-direction.
        Bz (array-like): Magnetic field component in the z-direction.
        grid (Grid): The grid on which the fields are defined.
        staggered_grid (Grid): The staggered grid for field interpolation.
        dt (float): The time step for the update.
        constants (dict): Dictionary containing physical constants. Must include:
            - 'C': Speed of light in vacuum (m/s).

    Returns:
        Particles: The particles with updated velocities.
    """
    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_position()
    vx, vy, vz = particles.get_velocity()
    # get the charge, mass, position, and velocity of the particles

    shape_factor = particles.get_shape()
    # get the shape factor of the particles

    dx = grid[0][1] - grid[0][0]
    dy = grid[1][1] - grid[1][0]
    dz = grid[2][1] - grid[2][0]
    # calculate the grid spacing in each direction

    ##################### ENERGY CONSERVING GRIDS ##########################
    Ex_grid = grid[0] - dx, grid[1], grid[2]
    Ey_grid = grid[0], grid[1] - dy, grid[2]
    Ez_grid = grid[0], grid[1], grid[2] - dz
    # create the grids for the electric field components

    Bx_grid = staggered_grid[0], grid[1] - dy, grid[2] - dz
    By_grid = grid[0] - dx, staggered_grid[1], grid[2] - dz
    Bz_grid = grid[0] - dx, grid[1] - dy, staggered_grid[2]
    # create the staggered grids for the magnetic field components
    ########################################################################

    # ################## MOMENTUM CONSERVING GRIDS ##########################
    # Ex_grid = grid
    # Ey_grid = grid
    # Ez_grid = grid
    # # create the grids for the electric field components
    # Bx_grid = staggered_grid[0], grid[1], grid[2]
    # By_grid = grid[0], staggered_grid[1], grid[2]
    # Bz_grid = grid[0], grid[1], staggered_grid[2]
    # # create the staggered grids for the magnetic field components

    Ex, Ey, Ez = E
    # unpack the electric field components
    # calculate the electric field at the particle positions using their specific grids
    efield_atx = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Ex, Ex_grid)(x, y, z),
        lambda _: create_quadratic_interpolator(Ex, Ex_grid)(x, y, z),
        operand=None
    )
    efield_aty = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Ey, Ey_grid)(x, y, z),
        lambda _: create_quadratic_interpolator(Ey, Ey_grid)(x, y, z),
        operand=None
    )
    efield_atz = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Ez, Ez_grid)(x, y, z),
        lambda _: create_quadratic_interpolator(Ez, Ez_grid)(x, y, z),
        operand=None
    )
    # unpack the magnetic field components
    Bx, By, Bz = B
    # calculate the magnetic field at the particle positions using their specific staggered grids
    bfield_atx = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Bx, Bx_grid)(x, y, z),
        lambda _: create_quadratic_interpolator(Bx, Bx_grid)(x, y, z),
        operand=None
    )
    bfield_aty = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(By, By_grid)(x, y, z),
        lambda _: create_quadratic_interpolator(By, By_grid)(x, y, z),
        operand=None
    )
    bfield_atz = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Bz, Bz_grid)(x, y, z),
        lambda _: create_quadratic_interpolator(Bz, Bz_grid)(x, y, z),
        operand=None
    )
    # calculate the magnetic field at the particle positions

    boris_vmap = jax.vmap(relativistic_boris_single_particle, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None))
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

    t = q*dt/(2*m)*jnp.array([bfield_atx, bfield_aty, bfield_atz]) / gamma
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


def create_trilinear_interpolator(field, grid):
    """
    Create a trilinear interpolation function for a given 3D field and grid.

    Args:
        field (ndarray): The 3D field to interpolate.
        grid (tuple): A tuple of three arrays representing the grid points in the x, y, and z directions.

    Returns:
        function: A function that takes (x, y, z) coordinates and returns the interpolated values.
    """
    x_grid, y_grid, z_grid = grid

    @jit
    def interpolator(x, y, z):
        x_idx = jnp.clip(jnp.searchsorted(x_grid, x) - 1, 0, len(x_grid) - 2)
        y_idx = jnp.clip(jnp.searchsorted(y_grid, y) - 1, 0, len(y_grid) - 2)
        z_idx = jnp.clip(jnp.searchsorted(z_grid, z) - 1, 0, len(z_grid) - 2)

        x0, x1 = x_grid[x_idx], x_grid[x_idx + 1]
        y0, y1 = y_grid[y_idx], y_grid[y_idx + 1]
        z0, z1 = z_grid[z_idx], z_grid[z_idx + 1]

        xd = (x - x0) / (x1 - x0)
        yd = (y - y0) / (y1 - y0)
        zd = (z - z0) / (z1 - z0)

        c00 = field[x_idx, y_idx, z_idx] * (1 - xd) + field[x_idx + 1, y_idx, z_idx] * xd
        c01 = field[x_idx, y_idx, z_idx + 1] * (1 - xd) + field[x_idx + 1, y_idx, z_idx + 1] * xd
        c10 = field[x_idx, y_idx + 1, z_idx] * (1 - xd) + field[x_idx + 1, y_idx + 1, z_idx] * xd
        c11 = field[x_idx, y_idx + 1, z_idx + 1] * (1 - xd) + field[x_idx + 1, y_idx + 1, z_idx + 1] * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        return c0 * (1 - zd) + c1 * zd


    vmap_interpolator = jax.vmap(interpolator, in_axes=(0, 0, 0), out_axes=0)
    # vectorize the interpolator for batch processing

    return vmap_interpolator


def create_quadratic_interpolator(field, grid):
    """
    Create a quadratic interpolation function for a given 3D field and grid.

    Args:
        field (ndarray): The 3D field to interpolate.
        grid (tuple): A tuple of three arrays representing the grid points in the x, y, and z directions.

    Returns:
        function: A function that takes (x, y, z) coordinates and returns the interpolated values.
    """
    x_grid, y_grid, z_grid = grid

    @jit
    def interpolator(x, y, z):
        x_idx = jnp.clip(jnp.searchsorted(x_grid, x) - 1, 1, len(x_grid) - 3)
        y_idx = jnp.clip(jnp.searchsorted(y_grid, y) - 1, 1, len(y_grid) - 3)
        z_idx = jnp.clip(jnp.searchsorted(z_grid, z) - 1, 1, len(z_grid) - 3)

        x0, x1, x2 = x_grid[x_idx - 1], x_grid[x_idx], x_grid[x_idx + 1]
        y0, y1, y2 = y_grid[y_idx - 1], y_grid[y_idx], y_grid[y_idx + 1]
        z0, z1, z2 = z_grid[z_idx - 1], z_grid[z_idx], z_grid[z_idx + 1]

        def quadratic_weights(t, t0, t1, t2):
            w0 = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2))
            w1 = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2))
            w2 = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1))
            return w0, w1, w2

        wx0, wx1, wx2 = quadratic_weights(x, x0, x1, x2)
        wy0, wy1, wy2 = quadratic_weights(y, y0, y1, y2)
        wz0, wz1, wz2 = quadratic_weights(z, z0, z1, z2)

        interpolated_value = 0.0
        for i, wx in enumerate([wx0, wx1, wx2]):
            for j, wy in enumerate([wy0, wy1, wy2]):
                for k, wz in enumerate([wz0, wz1, wz2]):
                    interpolated_value += (
                        wx * wy * wz * field[x_idx - 1 + i, y_idx - 1 + j, z_idx - 1 + k]
                    )

        return interpolated_value

    vmap_interpolator = jax.vmap(interpolator, in_axes=(0, 0, 0), out_axes=0)
    return vmap_interpolator