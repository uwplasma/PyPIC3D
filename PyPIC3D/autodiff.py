import jax
import jax.numpy as jnp
from jax import jit
# Christopher Woolford, October 24, 2024
# This script contains a seperate formulation of the particle in cell method using automatic differentiation

def initialize_fields(particles, k):
    """
    Initialize the electric and magnetic fields for a given set of particles.
    Parameters:
    particles (list): A list of particle objects or data structures containing particle information.
    k (float): A constant or parameter used in the calculation of the electric field.
    Returns:
    tuple: A tuple containing the electric field components (Ex, Ey, Ez) and magnetic field components (Bx, By, Bz).
           Each electric field component is a function of the particle positions and the parameter k.
           Each magnetic field component is a lambda function that returns 0.0 for any input.
    """

    Ex, Ey, Ez = autodiff_electric_field(particles, k)
    Bx = lambda x, y, z: 0.0
    By = lambda x, y, z: 0.0
    Bz = lambda x, y, z: 0.0
    return Ex, Ey, Ez, Bx, By, Bz

def compute_curl_functions(fx, fy, fz):
    """
    Compute the curl of a vector field (fx, fy, fz) and return the curls as functions of x, y, z.

    Parameters:
    - fx (function): Function representing the x-component of the vector field.
    - fy (function): Function representing the y-component of the vector field.
    - fz (function): Function representing the z-component of the vector field.

    Returns:
    - (function, function, function): Functions representing the x, y, and z components of the curl.
    """
    curl_fx = lambda x, y, z: jax.grad(fz, argnums=1)(x, y, z) - jax.grad(fy, argnums=2)(x, y, z)
    curl_fy = lambda x, y, z: jax.grad(fx, argnums=2)(x, y, z) - jax.grad(fz, argnums=0)(x, y, z)
    curl_fz = lambda x, y, z: jax.grad(fy, argnums=0)(x, y, z) - jax.grad(fx, argnums=1)(x, y, z)
    
    return curl_fx, curl_fy, curl_fz

def compute_divergence_function(fx, fy, fz):
    """
    Compute the divergence of a vector field (fx, fy, fz) and return the divergence as a function of x, y, z.

    Parameters:
    - fx (function): Function representing the x-component of the vector field.
    - fy (function): Function representing the y-component of the vector field.
    - fz (function): Function representing the z-component of the vector field.

    Returns:
    - function: Function representing the divergence of the vector field.
    """
    divergence = lambda x, y, z: jax.grad(fx, argnums=0)(x, y, z) + jax.grad(fy, argnums=1)(x, y, z) + jax.grad(fz, argnums=2)(x, y, z)
    
    return divergence

def autodiff_update_B(Bx, By, Bz, Ex, Ey, Ez, dt):
    """
    Update the magnetic field components Bx, By, and Bz using the electric field components Ex, Ey, and Ez
    and the time step dt.

    This function computes the curl of the electric field components and uses it to update the magnetic field
    components according to the given time step.

    Args:
        Bx (function): A function representing the x-component of the magnetic field.
        By (function): A function representing the y-component of the magnetic field.
        Bz (function): A function representing the z-component of the magnetic field.
        Ex (function): A function representing the x-component of the electric field.
        Ey (function): A function representing the y-component of the electric field.
        Ez (function): A function representing the z-component of the electric field.
        dt (float): The time step for the update.

    Returns:
        tuple: Updated magnetic field components (Bx, By, Bz) as functions.
    """
    curl_Ex, curl_Ey, curl_Ez = compute_curl_functions(Ex, Ey, Ez)
    Bx = lambda x, y, z: Bx(x, y, z) - dt/2*curl_Ex(x, y, z)
    By = lambda x, y, z: By(x, y, z) - dt/2*curl_Ey(x, y, z)
    Bz = lambda x, y, z: Bz(x, y, z) - dt/2*curl_Ez(x, y, z)
    # update the magnetic field
    return Bx, By, Bz


def autodiff_update_E(Ex, Ey, Ez, Bx, By, Bz, dt, C):
    """
    Update the electric field components Ex, Ey, and Ez based on the magnetic field components Bx, By, and Bz.

    Parameters:
    - Ex (ndarray): The x-component of the electric field.
    - Ey (ndarray): The y-component of the electric field.
    - Ez (ndarray): The z-component of the electric field.
    - Bx (ndarray): The x-component of the magnetic field.
    - By (ndarray): The y-component of the magnetic field.
    - Bz (ndarray): The z-component of the magnetic field.
    - Jx (ndarray): The x-component of the current density.
    - Jy (ndarray): The y-component of the current density.
    - Jz (ndarray): The z-component of the current density.
    - dt (float): The time step.
    - eps (float): The permittivity of the medium.

    Returns:
    - Ex (ndarray): The updated x-component of the electric field.
    - Ey (ndarray): The updated y-component of the electric field.
    - Ez (ndarray): The updated z-component of the electric field.
    """
    curl_Bx, curl_By, curl_Bz = compute_curl_functions(Bx, By, Bz)
    Ex = lambda x, y, z: Ex(x, y, z) + C**2 * curl_Bx(x, y, z) * dt / 2
    Ey = lambda x, y, z: Ey(x, y, z) + C**2 * curl_By(x, y, z) * dt / 2
    Ez = lambda x, y, z: Ez(x, y, z) + C**2 * curl_Bz(x, y, z) * dt / 2
    return Ex, Ey, Ez


def electric_potential(x, y, z, particles, k):
    """
    Calculate the electric potential at a given point in space due to a collection of particles.

    Parameters:
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.
    - z (float): The z-coordinate of the point.
    - particles (ndarray): An array of shape (N, 3) representing the positions of N particles.
    - k (float): A constant.

    Returns:
    - potential (float): The electric potential at the given point.
    """
    potential = 0.0

    for species in particles:
        N_particles = species.get_number_of_particles()
        charge = species.get_charge()
        particle_x, particle_y, particle_z = species.get_position()

        r_ = jnp.linalg.norm(jnp.array([x, y, z]) - jnp.array([particle_x, particle_y, particle_z]), axis=0)
        potential += jnp.sum( charge * k / r_)
    # compute the analytical solution for the electric potential

    return potential


def autodiff_electric_field(particles, k):
    """
    Calculate the electric field components (Ex, Ey, Ez) at a given point in space due to a collection of particles
    using automatic differentiation.

    Parameters:
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.
    - z (float): The z-coordinate of the point.
    - particles (ndarray): An array of shape (N, 3) representing the positions of N particles.
    - k (float): A constant.

    Returns:
    - Ex (float): The x-component of the electric field.
    - Ey (float): The y-component of the electric field.
    - Ez (float): The z-component of the electric field.
    """
    potential_fn = lambda x, y, z: electric_potential(x, y, z, particles, k)
    Ex = lambda x, y, z: -jax.grad(potential_fn, argnums=0)(x, y, z)
    Ey = lambda x, y, z: -jax.grad(potential_fn, argnums=1)(x, y, z)
    Ez = lambda x, y, z: -jax.grad(potential_fn, argnums=2)(x, y, z)
    
    return Ex, Ey, Ez

def particle_push(species, Ex, Ey, Ez, Bx, By, Bz, grid, staggered_grid, dt, GPUs):
    """
    Loop over a list of particle species and update their velocities and positions using the Boris algorithm.

    Parameters:
    - particles (list): A list of particle species, where each species contains particle properties.
    - Ex (ndarray): The x-component of the electric field.
    - Ey (ndarray): The y-component of the electric field.
    - Ez (ndarray): The z-component of the electric field.
    - Bx (ndarray): The x-component of the magnetic field.
    - By (ndarray): The y-component of the magnetic field.
    - Bz (ndarray): The z-component of the magnetic field.
    - dt (float): The time step for the update.
    """

    q = species.get_charge()
    m = species.get_mass()
    x, y, z = species.get_position()
    vx, vy, vz = species.get_velocity()
    newvx, newvy, newvz = boris(q, m, x, y, z, vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, dt)
    species.set_velocity(newvx, newvy, newvz)

    return species


def boris(q, m, x, y, z, vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, dt):
    """
    Implements the Boris algorithm to advance the velocity of a charged particle 
    in an electromagnetic field over a time step `dt`.
    Parameters:
    q (float): Charge of the particle.
    m (float): Mass of the particle.
    x (float): x-coordinate of the particle's position.
    y (float): y-coordinate of the particle's position.
    z (float): z-coordinate of the particle's position.
    vx (float): x-component of the particle's velocity.
    vy (float): y-component of the particle's velocity.
    vz (float): z-component of the particle's velocity.
    Ex (function): Function to compute the x-component of the electric field at a given position.
    Ey (function): Function to compute the y-component of the electric field at a given position.
    Ez (function): Function to compute the z-component of the electric field at a given position.
    Bx (function): Function to compute the x-component of the magnetic field at a given position.
    By (function): Function to compute the y-component of the magnetic field at a given position.
    Bz (function): Function to compute the z-component of the magnetic field at a given position.
    dt (float): Time step over which to advance the particle's velocity.
    Returns:
    tuple: A tuple containing the new velocity components (newvx, newvy, newvz) of the particle.
    """
    
    efield_atx = Ex(x, y, z)
    efield_aty = Ey(x, y, z)
    efield_atz = Ez(x, y, z)
    # calculate the electric field at the particle positions using the provided functions

    bfield_atx = Bx(x, y, z)
    bfield_aty = By(x, y, z)
    bfield_atz = Bz(x, y, z)
    # calculate the magnetic field at the particle positions using the provided functions

    vxminus = vx + q*dt/(2*m)*efield_atx
    vyminus = vy + q*dt/(2*m)*efield_aty
    vzminus = vz + q*dt/(2*m)*efield_atz
    # calculate the v minus vector used in the boris push algorithm
    tx = q*dt/(2*m)*bfield_atx
    ty = q*dt/(2*m)*bfield_aty
    tz = q*dt/(2*m)*bfield_atz

    vprimex = vxminus + (vyminus*tz - vzminus*ty)
    vprimey = vyminus + (vzminus*tx - vxminus*tz)
    vprimez = vzminus + (vxminus*ty - vyminus*tx)
    # vprime = vminus + vminus cross t

    smag = 2 / (1 + tx*tx + ty*ty + tz*tz)
    sx = smag * tx
    sy = smag * ty
    sz = smag * tz
    # calculate the scaled rotation vector

    vxplus = vxminus + (vprimey*sz - vprimez*sy)
    vyplus = vyminus + (vprimez*sx - vprimex*sz)
    vzplus = vzminus + (vprimex*sy - vprimey*sx)

    newvx = vxplus + q*dt/(2*m)*efield_atx
    newvy = vyplus + q*dt/(2*m)*efield_aty
    newvz = vzplus + q*dt/(2*m)*efield_atz
    # calculate the new velocity

    return newvx, newvy, newvz