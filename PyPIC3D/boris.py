import jax
from jax import jit
import jax.numpy as jnp
from functools import partial

from PyPIC3D.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.utils import wrap_around

@jit
def particle_push(particles, E, B, grid, staggered_grid, dt, constants, periodic=True, relativistic=True):
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
    x, y, z = particles.get_forward_position()
    vx, vy, vz = particles.get_velocity()
    # get the charge, mass, position, and velocity of the particles

    shape_factor = particles.get_shape()
    # get the shape factor of the particles

    ################## INTERPOLATION GRIDS ###################################
    Ex_grid = staggered_grid[0], grid[1], grid[2]
    Ey_grid = grid[0], staggered_grid[1], grid[2]
    Ez_grid = grid[0], grid[1], staggered_grid[2]
    # create the grids for the electric field components
    Bx_grid = grid[0], staggered_grid[1], staggered_grid[2]
    By_grid = staggered_grid[0], grid[1], staggered_grid[2]
    Bz_grid = staggered_grid[0], staggered_grid[1], grid[2]
    # create the staggered grids for the magnetic field components
    ##########################################################################


    ################## INTERPOLATE FIELDS TO PARTICLE POSITIONS ##############
    Ex, Ey, Ez = E
    # unpack the electric field components
    Bx, By, Bz = B
    # unpack the magnetic field components

    Ny = len(grid[1])
    Nz = len(grid[2])

    if Ny == 1 and Nz == 1:
        node_stack = jnp.stack((Ey, Ez, Bx), axis=-1)
        face_stack = jnp.stack((Ex, By, Bz), axis=-1)

        node_vals = interpolate_field_to_particles(node_stack, x, y, z, (grid[0], grid[1], grid[2]), shape_factor)
        face_vals = interpolate_field_to_particles(face_stack, x, y, z, (staggered_grid[0], grid[1], grid[2]), shape_factor)

        efield_aty, efield_atz, bfield_atx = node_vals[:, 0], node_vals[:, 1], node_vals[:, 2]
        efield_atx, bfield_aty, bfield_atz = face_vals[:, 0], face_vals[:, 1], face_vals[:, 2]

    elif Nz == 1:
        ex_by = jnp.stack((Ex, By), axis=-1)
        ey_bx = jnp.stack((Ey, Bx), axis=-1)

        ex_by_vals = interpolate_field_to_particles(ex_by, x, y, z, Ex_grid, shape_factor)
        ey_bx_vals = interpolate_field_to_particles(ey_bx, x, y, z, Ey_grid, shape_factor)
        ez_vals = interpolate_field_to_particles(Ez, x, y, z, Ez_grid, shape_factor)
        bz_vals = interpolate_field_to_particles(Bz, x, y, z, Bz_grid, shape_factor)

        efield_atx, bfield_aty = ex_by_vals[:, 0], ex_by_vals[:, 1]
        efield_aty, bfield_atx = ey_bx_vals[:, 0], ey_bx_vals[:, 1]
        efield_atz = ez_vals
        bfield_atz = bz_vals

    else:
        efield_atx = interpolate_field_to_particles(Ex, x, y, z, Ex_grid, shape_factor)
        efield_aty = interpolate_field_to_particles(Ey, x, y, z, Ey_grid, shape_factor)
        efield_atz = interpolate_field_to_particles(Ez, x, y, z, Ez_grid, shape_factor)
        # calculate the electric field at the particle positions on the Yee-staggered component grids
        bfield_atx = interpolate_field_to_particles(Bx, x, y, z, Bx_grid, shape_factor)
        bfield_aty = interpolate_field_to_particles(By, x, y, z, By_grid, shape_factor)
        bfield_atz = interpolate_field_to_particles(Bz, x, y, z, Bz_grid, shape_factor)
        # calculate the magnetic field at the particle positions on the Yee-staggered component grids
    #########################################################################


    #################### BORIS ALGORITHM ####################################
    newvx, newvy, newvz = jax.lax.cond(
        relativistic == True,
        lambda _: relativistic_boris_push(
            vx,
            vy,
            vz,
            efield_atx,
            efield_aty,
            efield_atz,
            bfield_atx,
            bfield_aty,
            bfield_atz,
            q,
            m,
            dt,
            constants,
        ),
        lambda _: boris_push(
            vx,
            vy,
            vz,
            efield_atx,
            efield_aty,
            efield_atz,
            bfield_atx,
            bfield_aty,
            bfield_atz,
            q,
            m,
            dt,
        ),
        operand=None,
    )
    # apply the Boris algorithm (vectorized over particles)
    #########################################################################


    particles.set_velocity(newvx, newvy, newvz)
    # set the new velocities of the particles
    return particles


def boris_push(vx, vy, vz, ex, ey, ez, bx, by, bz, q, m, dt):
    qmdt2 = q * dt / (2 * m)

    vminus_x = vx + qmdt2 * ex
    vminus_y = vy + qmdt2 * ey
    vminus_z = vz + qmdt2 * ez

    t_x = qmdt2 * bx
    t_y = qmdt2 * by
    t_z = qmdt2 * bz

    t2 = t_x * t_x + t_y * t_y + t_z * t_z
    inv = 1.0 / (1.0 + t2)
    s_x = 2.0 * t_x * inv
    s_y = 2.0 * t_y * inv
    s_z = 2.0 * t_z * inv

    vprime_x = vminus_x + (vminus_y * t_z - vminus_z * t_y)
    vprime_y = vminus_y + (vminus_z * t_x - vminus_x * t_z)
    vprime_z = vminus_z + (vminus_x * t_y - vminus_y * t_x)

    vplus_x = vminus_x + (vprime_y * s_z - vprime_z * s_y)
    vplus_y = vminus_y + (vprime_z * s_x - vprime_x * s_z)
    vplus_z = vminus_z + (vprime_x * s_y - vprime_y * s_x)

    newvx = vplus_x + qmdt2 * ex
    newvy = vplus_y + qmdt2 * ey
    newvz = vplus_z + qmdt2 * ez

    return newvx, newvy, newvz


def relativistic_boris_push(vx, vy, vz, ex, ey, ez, bx, by, bz, q, m, dt, constants):
    C = constants["C"]
    qmdt2 = q * dt / (2 * m)

    v2_over_c2 = (vx * vx + vy * vy + vz * vz) / (C * C)
    gamma = 1.0 / jnp.sqrt(1.0 - v2_over_c2)

    uminus_x = vx * gamma + qmdt2 * ex
    uminus_y = vy * gamma + qmdt2 * ey
    uminus_z = vz * gamma + qmdt2 * ez

    uminus2_over_c2 = (uminus_x * uminus_x + uminus_y * uminus_y + uminus_z * uminus_z) / (C * C)
    gamma_minus = jnp.sqrt(1.0 + uminus2_over_c2)

    t_x = (qmdt2 * bx) / gamma_minus
    t_y = (qmdt2 * by) / gamma_minus
    t_z = (qmdt2 * bz) / gamma_minus

    t2 = t_x * t_x + t_y * t_y + t_z * t_z
    inv = 1.0 / (1.0 + t2)
    s_x = 2.0 * t_x * inv
    s_y = 2.0 * t_y * inv
    s_z = 2.0 * t_z * inv

    uprime_x = uminus_x + (uminus_y * t_z - uminus_z * t_y)
    uprime_y = uminus_y + (uminus_z * t_x - uminus_x * t_z)
    uprime_z = uminus_z + (uminus_x * t_y - uminus_y * t_x)

    uplus_x = uminus_x + (uprime_y * s_z - uprime_z * s_y)
    uplus_y = uminus_y + (uprime_z * s_x - uprime_x * s_z)
    uplus_z = uminus_z + (uprime_x * s_y - uprime_y * s_x)

    newu_x = uplus_x + qmdt2 * ex
    newu_y = uplus_y + qmdt2 * ey
    newu_z = uplus_z + qmdt2 * ez

    newu2_over_c2 = (newu_x * newu_x + newu_y * newu_y + newu_z * newu_z) / (C * C)
    new_gamma = jnp.sqrt(1.0 + newu2_over_c2)

    newvx = newu_x / new_gamma
    newvy = newu_y / new_gamma
    newvz = newu_z / new_gamma

    return newvx, newvy, newvz

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

    gamma = 1 / jnp.sqrt( 1  - (  (v[0]**2 + v[1]**2 + v[2]**2) / C**2 ) )
    # define the gamma factor

    uminus = v * gamma + q*dt/(2*m)*jnp.array([efield_atx, efield_aty, efield_atz])
    # get the u minus vector

    gamma_minus = jnp.sqrt( 1  + (  (uminus[0]**2 + uminus[1]**2 + uminus[2]**2) / C**2 ) )
    # define the gamma minus factor

    t = q*dt/(2*m)*jnp.array([bfield_atx, bfield_aty, bfield_atz]) / gamma_minus
    # calculate the t vector
    uprime = uminus + jnp.cross(uminus, t)
    # calculate the u prime vector

    s = 2*t / (1 + t[0]**2 + t[1]**2 + t[2]**2)
    # calculate the s vector
    uplus = uminus + jnp.cross(uprime, s)
    # calculate the u plus vector

    newu = uplus + q*dt/(2*m)*jnp.array([efield_atx, efield_aty, efield_atz])
    # calculate the new velocity

    new_gamma = jnp.sqrt( 1  +  (  (newu[0]**2 + newu[1]**2 + newu[2]**2) / C**2 ) )
    # define the new gamma factor

    newv = newu / new_gamma
    # convert the relativistic velocity back to the lab frame

    return newv[0], newv[1], newv[2]


@partial(jit, static_argnames=("shape_factor",))
def interpolate_field_to_particles(field, x, y, z, grid, shape_factor):
    """
    Interpolate a Yee-grid field component to particle positions using PIC shape functions.

    Args:
        field (ndarray): Field component values on the supplied component grid.
        x, y, z (ndarray): Particle coordinates.
        grid (tuple): Component grid coordinates (x_grid, y_grid, z_grid).
        shape_factor (int): Particle shape order (1 -> first order, 2 -> second order).

    Returns:
        ndarray: Interpolated field value at each particle position.
    """
    x_grid, y_grid, z_grid = grid

    xmin, ymin, zmin = x_grid[0], y_grid[0], z_grid[0]
    # grid minimum coordinates

    Nx = len(x_grid)
    Ny = len(y_grid)
    Nz = len(z_grid)
    # grid point counts for each direction
    x_active = Nx != 1
    y_active = Ny != 1
    z_active = Nz != 1
    # infer effective dimensionality from grid extents

    dx = x_grid[1] - x_grid[0] if Nx > 1 else 1.0
    dy = y_grid[1] - y_grid[0] if Ny > 1 else 1.0
    dz = z_grid[1] - z_grid[0] if Nz > 1 else 1.0
    # grid spacing in each direction

    if shape_factor == 1:
        x0 = jnp.floor((x - xmin) / dx).astype(jnp.int32)
    else:
        x0 = jnp.round((x - xmin) / dx).astype(jnp.int32)
    # compute the stencil anchor points (cell-left for first order, nearest node for second order)

    deltax = (x - xmin) - x0 * dx
    # determine the distance from the closest grid nodes

    if x_active and (not y_active) and (not z_active):
        x0 = wrap_around(x0, Nx)
        if shape_factor == 1:
            r = deltax / dx
            xpts = jnp.stack((x0, wrap_around(x0 + 1, Nx)), axis=0)
            xw = jnp.stack((1.0 - r, r), axis=0)
        else:
            r = deltax / dx
            xpts = jnp.stack((wrap_around(x0 - 1, Nx), x0, wrap_around(x0 + 1, Nx)), axis=0)
            xw = jnp.stack(
                (
                    0.5 * (0.5 - r) ** 2,
                    0.75 - r**2,
                    0.5 * (0.5 + r) ** 2,
                ),
                axis=0,
            )
        if field.ndim == 4:
            return jnp.sum(field[xpts, 0, 0, :] * xw[:, :, None], axis=0)
        return jnp.sum(field[xpts, 0, 0] * xw, axis=0)

    if shape_factor == 1:
        y0 = jnp.floor((y - ymin) / dy).astype(jnp.int32)
        z0 = jnp.floor((z - zmin) / dz).astype(jnp.int32)
    else:
        y0 = jnp.round((y - ymin) / dy).astype(jnp.int32)
        z0 = jnp.round((z - zmin) / dz).astype(jnp.int32)

    deltay = (y - ymin) - y0 * dy
    deltaz = (z - zmin) - z0 * dz

    if shape_factor == 1:
        x_weights, y_weights, z_weights = get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz)
    else:
        x_weights, y_weights, z_weights = get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz)
    x_weights = jnp.asarray(x_weights)
    y_weights = jnp.asarray(y_weights)
    z_weights = jnp.asarray(z_weights)
    # compute the shape function weights for the particles and convert them to arrays

    x0 = wrap_around(x0, Nx)
    y0 = wrap_around(y0, Ny)
    z0 = wrap_around(z0, Nz)
    # wrap around the grid points for periodic boundary conditions
    x1 = wrap_around(x0+1, Nx)
    y1 = wrap_around(y0+1, Ny)
    z1 = wrap_around(z0+1, Nz)
    # calculate the right grid point
    x_minus1 = x0 - 1
    y_minus1 = y0 - 1
    z_minus1 = z0 - 1
    # calculate the left grid point
    xpts = jnp.asarray([x_minus1, x0, x1])
    ypts = jnp.asarray([y_minus1, y0, y1])
    zpts = jnp.asarray([z_minus1, z0, z1])
    # place all the points in a list

    if shape_factor == 1:
        # drop the redundant (-1) stencil point for first-order (its weights are identically 0)
        xpts = xpts[1:, ...]
        ypts = ypts[1:, ...]
        zpts = zpts[1:, ...]
        x_weights = x_weights[1:, ...]
        y_weights = y_weights[1:, ...]
        z_weights = z_weights[1:, ...]

    # Keep full shape-factor computation but collapse inactive axes to an
    # effective stencil size of 1 to avoid redundant interpolation work.
    if x_active:
        xpts_eff = xpts
        x_weights_eff = x_weights
    else:
        xpts_eff = jnp.zeros((1, xpts.shape[1]), dtype=xpts.dtype)
        x_weights_eff = jnp.sum(x_weights, axis=0, keepdims=True)

    if y_active:
        ypts_eff = ypts
        y_weights_eff = y_weights
    else:
        ypts_eff = jnp.zeros((1, ypts.shape[1]), dtype=ypts.dtype)
        y_weights_eff = jnp.sum(y_weights, axis=0, keepdims=True)

    if z_active:
        zpts_eff = zpts
        z_weights_eff = z_weights
    else:
        zpts_eff = jnp.zeros((1, zpts.shape[1]), dtype=zpts.dtype)
        z_weights_eff = jnp.sum(z_weights, axis=0, keepdims=True)

    field_vals = field[
        xpts_eff[:, None, None, :],
        ypts_eff[None, :, None, :],
        zpts_eff[None, None, :, :],
    ]
    weights = (
        x_weights_eff[:, None, None, :]
        * y_weights_eff[None, :, None, :]
        * z_weights_eff[None, None, :, :]
    )
    if field.ndim == 4:
        weights = weights[..., None]
    return jnp.sum(field_vals * weights, axis=(0, 1, 2))
