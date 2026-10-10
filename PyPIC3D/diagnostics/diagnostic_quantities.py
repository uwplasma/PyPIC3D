import jax.numpy as jnp


def compute_dark_energy(dark_fields, static_parameters, dynamic_parameters, pml_state=None):
    """Physical Proca energy at integer time, using reconstructed A and B.

    This diagnostic is second-order accurate in time, rather than an exactly
    conserved discrete leapfrog Hamiltonian. Halos are excluded from the sum.
    """
    from PyPIC3D.solvers.dark_matter_yee.dark_photon_fields import synchronized_dark_fields

    E, A, phi, B = synchronized_dark_fields(dark_fields, static_parameters, dynamic_parameters, pml_state)
    g = static_parameters.guard_cells
    interior = (slice(None),) * 3 + (slice(g, -g),) * 3
    c2 = dynamic_parameters.C**2
    mass2 = static_parameters.dark_mu**2
    density = sum(e[interior]**2 + c2 * b[interior]**2 + c2 * mass2 * a[interior]**2
                  for e, b, a in zip(E, B, A)) + mass2 * phi[interior]**2
    volume = dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
    return 0.5 * dynamic_parameters.eps * volume * jnp.sum(density)


def compute_energy(particles, E, B, static_parameters, dynamic_parameters, species_config=None):
    """
    Compute the electric, magnetic, and particle kinetic energies.
    """

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    # Use physical interior slices to exclude ghost cells from the energy
    # integral. Tiled fields have leading tile axes followed by local
    # ghost-celled Yee arrays.
    if Ex.ndim == 6:
        g = static_parameters.guard_cells
        interior = (
            slice(None),
            slice(None),
            slice(None),
            slice(g, -g),
            slice(g, -g),
            slice(g, -g),
        )
    else:
        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    dV = dx * dy * dz
    E2_integral = jnp.sum(Ex[interior]**2 + Ey[interior]**2 + Ez[interior]**2) * dV
    B2_integral = jnp.sum(Bx[interior]**2 + By[interior]**2 + Bz[interior]**2) * dV
    e_energy = 0.5 * dynamic_parameters.eps * E2_integral
    b_energy = 0.5 / dynamic_parameters.mu * B2_integral

    C = dynamic_parameters.C
    vx = particles.u[..., 0]
    vy = particles.u[..., 1]
    vz = particles.u[..., 2]
    v2 = vx**2 + vy**2 + vz**2

    active = particles.active.astype(v2.dtype)
    species_mass = species_config.mass * species_config.weight
    mass = jnp.broadcast_to(
        species_mass.reshape((1, 1, 1, species_mass.shape[0], 1)),
        particles.active.shape,
    )
    gamma = 1.0 / jnp.sqrt(1 - v2 / C**2)
    momentum2 = jnp.square(mass * gamma) * v2
    kinetic_energy = jnp.sum(active * (jnp.sqrt(momentum2 * C**2 + mass**2 * C**4) - mass * C**2))

    return e_energy, b_energy, kinetic_energy


def compute_total_momentum(particles, species_config=None):
    """
    Compute the scalar momentum diagnostic for tiled particles.
    """

    vmag = jnp.sqrt(particles.u[..., 0]**2 + particles.u[..., 1]**2 + particles.u[..., 2]**2)
    active = particles.active.astype(vmag.dtype)
    species_mass = species_config.mass * species_config.weight
    mass = jnp.broadcast_to(
        species_mass.reshape((1, 1, 1, species_mass.shape[0], 1)),
        particles.active.shape,
    )
    return jnp.sum(active * vmag * mass)
