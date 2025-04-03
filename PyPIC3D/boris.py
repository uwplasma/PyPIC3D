import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.utils import create_trilinear_interpolator

@jit
def particle_push(particles, E, B, grid, staggered_grid, dt):
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

    Returns:
        Particles: The particles with updated velocities.
    """
    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_position()
    vx, vy, vz = particles.get_velocity()
    # get the charge, mass, position, and velocity of the particles

    Ex, Ey, Ez = E
    Ex_interpolate = create_trilinear_interpolator(Ex, grid)
    Ey_interpolate = create_trilinear_interpolator(Ey, grid)
    Ez_interpolate = create_trilinear_interpolator(Ez, grid)
    # interpolate the electric field

    Bx, By, Bz = B
    Bx_interpolate = create_trilinear_interpolator(Bx, staggered_grid)
    By_interpolate = create_trilinear_interpolator(By, staggered_grid)
    Bz_interpolate = create_trilinear_interpolator(Bz, staggered_grid)
    # interpolate the magnetic field

    efield_atx = Ex_interpolate(x, y, z)
    efield_aty = Ey_interpolate(x, y, z)
    efield_atz = Ez_interpolate(x, y, z)
    # calculate the electric field at the particle positions
    bfield_atx = Bx_interpolate(x, y, z)
    bfield_aty = By_interpolate(x, y, z)
    bfield_atz = Bz_interpolate(x, y, z)
    # calculate the magnetic field at the particle positions

    boris_vmap = jax.vmap(boris_single_particle, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None))
    newvx, newvy, newvz = boris_vmap(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt)

    particles.set_velocity(newvx, newvy, newvz)
    # set the new velocities of the particles
    return particles

@jit
def boris_single_particle(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt):
    """
    Updates the velocity of a single particle using the Boris algorithm.
    Parameters:
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