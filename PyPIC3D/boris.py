import jax
from jax import jit
import jax.numpy as jnp

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
    efield_atx = interpolate_field_to_particles(Ex, x, y, z, Ex_grid, shape_factor)
    efield_aty = interpolate_field_to_particles(Ey, x, y, z, Ey_grid, shape_factor)
    efield_atz = interpolate_field_to_particles(Ez, x, y, z, Ez_grid, shape_factor)
    # calculate the electric field at the particle positions on the Yee-staggered component grids
    Bx, By, Bz = B
    # unpack the magnetic field components
    bfield_atx = interpolate_field_to_particles(Bx, x, y, z, Bx_grid, shape_factor)
    bfield_aty = interpolate_field_to_particles(By, x, y, z, By_grid, shape_factor)
    bfield_atz = interpolate_field_to_particles(Bz, x, y, z, Bz_grid, shape_factor)
    # calculate the magnetic field at the particle positions on the Yee-staggered component grids
    #########################################################################


    #################### BORIS ALGORITHM ####################################
    boris_vmap              = jax.vmap(boris_single_particle, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None))
    relativistic_boris_vmap = jax.vmap(relativistic_boris_single_particle, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None))
    # vectorize the Boris algorithm for batch processing

    newvx, newvy, newvz = jax.lax.cond(
        relativistic == True,
        lambda _: relativistic_boris_vmap(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants),
        lambda _: boris_vmap(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants),
        operand=None
    )
    # apply the Boris algorithm to update the velocities of the particles
    #########################################################################


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


@jit
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

    dx = x_grid[1] - x_grid[0] if Nx > 1 else 1.0
    dy = y_grid[1] - y_grid[0] if Ny > 1 else 1.0
    dz = z_grid[1] - z_grid[0] if Nz > 1 else 1.0
    # grid spacing in each direction

    x0 = jnp.floor((x - xmin) / dx).astype(int)
    y0 = jnp.floor((y - ymin) / dy).astype(int)
    z0 = jnp.floor((z - zmin) / dz).astype(int)
    # compute the closest grid nodes

    deltax = (x - xmin) - x0 * dx
    deltay = (y - ymin) - y0 * dy
    deltaz = (z - zmin) - z0 * dz
    # determine the distance from the closest grid nodes

    x_weights, y_weights, z_weights = jax.lax.cond(
        shape_factor == 1,
        lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
        lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
        operand=None,
    )
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

    def stencil_contribution(stencil_idx):
        i, j, k = stencil_idx
        return (
            field[xpts[i, ...], ypts[j, ...], zpts[k, ...]]
            * x_weights[i, ...]
            * y_weights[j, ...]
            * z_weights[k, ...]
        )
    # define a function to compute the contribution from each point in the 3x3x3 stencil

    stencil_indicies = jnp.asarray( [[i, j, k] for i in range(3) for j in range(3) for k in range(3)] )
    # compute the contribution from each point in the 3x3x3 stencil and add them to an array

    interpolated_field = jnp.sum(jax.vmap(stencil_contribution)(stencil_indicies), axis=0)
    # sum the contributions from all stencil points to get the final interpolated field value at each particle position

    return interpolated_field