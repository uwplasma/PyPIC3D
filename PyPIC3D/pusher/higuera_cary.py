import jax.numpy as jnp
from jax import jit


@jit
def gamma_from_u(u, constants):
    """
    Calculate gamma from the relativistic velocity-like momentum u = gamma * v.
    """
    C = constants["C"]
    # speed of light

    return jnp.sqrt(1.0 + jnp.dot(u, u) / C**2)


@jit
def velocity_from_u(u, constants):
    """
    Convert the relativistic velocity-like momentum u = gamma * v to velocity.
    """
    gamma = gamma_from_u(u, constants)
    # compute gamma from u

    return u / gamma


@jit
def u_from_v(v, constants):
    """
    Convert velocity to the relativistic velocity-like momentum u = gamma * v.
    """
    C = constants["C"]
    # speed of light

    gamma = 1.0 / jnp.sqrt(1.0 - jnp.dot(v, v) / C**2)
    # compute gamma from velocity

    return gamma * v


@jit
def gamma_higuera_cary(u_e, beta, constants):
    """
    Calculate the Higuera-Cary gamma used in the magnetic rotation.
    """
    C = constants["C"]
    # speed of light

    beta_squared = jnp.dot(beta, beta)
    u_star = jnp.dot(u_e, beta) / C
    sigma = gamma_from_u(u_e, constants) ** 2 - beta_squared
    # compute the scalar quantities for the Higuera-Cary gamma expression

    return jnp.sqrt((sigma + jnp.sqrt(sigma**2 + 4.0 * (beta_squared + u_star**2))) / 2.0)


@jit
def higuera_cary_single_particle(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants):
    """
    Updates the velocity of a single particle using the Higuera-Cary algorithm.
    Args:
        vx (float): Initial x component of the particle velocity.
        vy (float): Initial y component of the particle velocity.
        vz (float): Initial z component of the particle velocity.
        efield_atx (float): x component of the electric field at the particle.
        efield_aty (float): y component of the electric field at the particle.
        efield_atz (float): z component of the electric field at the particle.
        bfield_atx (float): x component of the magnetic field at the particle.
        bfield_aty (float): y component of the magnetic field at the particle.
        bfield_atz (float): z component of the magnetic field at the particle.
        q (float): Charge of the particle.
        m (float): Mass of the particle.
        dt (float): Time step for the update.
        constants (dict): Dictionary containing the speed of light.
    Returns:
        tuple: Updated velocity components (vx, vy, vz) of the particle.
    """

    v = jnp.array([vx, vy, vz])
    # convert v into an array

    u = u_from_v(v, constants)
    # convert velocity into relativistic velocity-like momentum

    E = jnp.array([efield_atx, efield_aty, efield_atz])
    B = jnp.array([bfield_atx, bfield_aty, bfield_atz])
    # convert the fields into vectors

    half_charge_timestep = q * dt / (2.0 * m)
    # compute the half time-step charge-to-mass factor

    epsilon = half_charge_timestep * E
    beta = half_charge_timestep * B
    # compute the electric and magnetic update vectors

    u_e = u + epsilon
    # apply the first half electric-field kick

    gamma_next = gamma_higuera_cary(u_e, beta, constants)
    t = beta / gamma_next
    # calculate the Higuera-Cary rotation vector

    s = 1.0 / (1.0 + jnp.dot(t, t))
    u_m = s * (u_e + jnp.dot(u_e, t) * t + jnp.cross(u_e, t))
    # rotate the momentum using the Higuera-Cary magnetic update

    newu = u_m + epsilon + jnp.cross(u_m, t)
    # apply the second electric-field kick and final rotation correction

    newv = velocity_from_u(newu, constants)
    # convert the relativistic velocity-like momentum back to velocity

    return newv[0], newv[1], newv[2]
