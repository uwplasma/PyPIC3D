"""Fixed-shape active-particle batch selection for JAX kernels."""

import jax.numpy as jnp


def prepare_particle_batches(active, configured_batch_size):
    """Build the fixed-shape active-index array for one particle tile."""

    particle_capacity = active.size
    batch_size = min(int(configured_batch_size), particle_capacity)
    active_flat = active.reshape(-1)
    active_indices = jnp.nonzero(
        active_flat,
        size=particle_capacity,
        fill_value=0,
    )[0]
    n_active = jnp.count_nonzero(active_flat)

    return particle_capacity, batch_size, active_indices, n_active


def particle_batch_indices(active_indices, n_active, batch_index, batch_size):
    """Select one fixed-size batch and identify its physical entries."""

    active_offsets = batch_index * batch_size + jnp.arange(batch_size)
    valid = active_offsets < n_active
    safe_offsets = jnp.minimum(
        active_offsets,
        jnp.maximum(n_active - 1, 0),
    )
    particle_indices = active_indices[safe_offsets]

    return particle_indices, valid


def number_of_particle_batches(n_active, batch_size):
    """Return the number of fixed-size batches required for active entries."""

    if batch_size == 0:
        return jnp.asarray(0, dtype=n_active.dtype)
    return (n_active + batch_size - 1) // batch_size
