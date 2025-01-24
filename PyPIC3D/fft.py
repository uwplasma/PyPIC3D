import jax
import jax.numpy as jnp
from jax.experimental import shard_map

@jax.jit
def fft_slab_decomposition(field):
    """
    Perform a slab decomposition of a 3D field and apply FFT to each slab in parallel.

    Args:
        field (jnp.ndarray): A 3D array with shape (Nx, Ny, Nz) representing the field to be transformed.

    Returns:
        jnp.ndarray: A 3D array with the same shape as the input field, containing the FFT-transformed data.
    """
    # Assume field is a 3D array with shape (Nx, Ny, Nz)
    slabs = jnp.array_split(field, jax.device_count(), axis=2)  # Split along z-axis

    def fft_slab(slab):
        return jnp.fft.fftn(slab)

    # Use shard_map to apply FFT to each slab in parallel
    fft_slabs = shard_map(fft_slab, slabs)

    # Concatenate the results back into a single array
    return jnp.concatenate(fft_slabs, axis=2)

@jax.jit
def fft_pencil_decomposition(field):
    """
    Perform FFT on a 3D array using pencil decomposition.

    This function splits the input 3D array into smaller sub-arrays (pencils)
    and applies FFT to each pencil in parallel using JAX's shard_map.

    Args:
        field (jnp.ndarray): A 3D array with shape (Nx, Ny, Nz) representing
                             the input field to be transformed.

    Returns:
        jnp.ndarray: A 3D array with the same shape as the input, containing
                     the FFT-transformed data.
    """
    # Assume field is a 3D array with shape (Nx, Ny, Nz)
    pencils = [jnp.array_split(slab, jax.device_count(), axis=1) for slab in jnp.array_split(field, jax.device_count(), axis=2)]
    pencils = [pencil for sublist in pencils for pencil in sublist]  # Flatten the list of lists

    def fft_pencil(pencil):
        return jnp.fft.fftn(pencil)

    # Use shard_map to apply FFT to each pencil in parallel
    fft_pencils = shard_map(fft_pencil, pencils)

    # Concatenate the results back into a single array
    return jnp.concatenate(fft_pencils, axis=1)