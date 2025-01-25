import jax
import jax.numpy as jnp
from   jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial

@partial(jax.jit, static_argnums=(1,2))
def fft_slab_decomposition(field, axis=2, num_cores=4):
    """
    Perform a slab decomposition of a 3D field and apply FFT to each slab in parallel.

    Args:
        field (jnp.ndarray): A 3D array with shape (Nx, Ny, Nz) representing the field to be transformed.
        axis (int): The axis along which to perform the FFT. Default is 2.

    Returns:
        jnp.ndarray: A 3D array with the same shape as the input field, containing the FFT-transformed data.
    """

    axes = [0, 1, 2]
    axes.remove(axis)
    # Assume field is a 3D array with shape (Nx, Ny, Nz)

    N = field.shape[axes[0]]
    # get the number of slabs

    slabs = jnp.split(field, N, axis=axes[0]) # split along different axis


    #print("Field shape:", field.shape)
    #print("Slabs shape:", [slab.shape for slab in slabs])


    def fft_slab(slab):
        return jnp.fft.fftn(jnp.array(slab), axes=[axis])

    # Use shard_map to apply FFT to each slab in parallel
    #mesh = Mesh(jax.devices()[:num_cores], ('i',))
    #f_shmapped = shard_map(fft_slab, mesh, in_specs=P('i'), out_specs=P('i'))

    # Concatenate the results back into a single array

    #fft_slabs = f_shmapped(slabs)

    fft_slabs = [fft_slab(slab) for slab in slabs]

    #print("FFT Slabs shape:", [slab.shape for slab in fft_slabs])
    fft_field = jnp.array(fft_slabs).reshape(field.shape)
    fft_field = fft_field.transpose(axes[0], axes[1], axis)
    #print("FFT Field shape:", fft_field.shape)
    return fft_field

@partial(jax.jit, static_argnums=(1,2))
def ifft_slab_decomposition(field, axis=2, num_cores=4):
    """
    Perform a slab decomposition of a 3D field and apply FFT to each slab in parallel.

    Args:
        field (jnp.ndarray): A 3D array with shape (Nx, Ny, Nz) representing the field to be transformed.
        axis (int): The axis along which to perform the FFT. Default is 2.

    Returns:
        jnp.ndarray: A 3D array with the same shape as the input field, containing the FFT-transformed data.
    """

    axes = [0, 1, 2]
    axes.remove(axis)
    # Assume field is a 3D array with shape (Nx, Ny, Nz)

    N = field.shape[axes[0]]
    # get the number of slabs

    slabs = jnp.split(field, N, axis=axes[0]) # split along different axis


    #print("Field shape:", field.shape)
    #print("Slabs shape:", [slab.shape for slab in slabs])


    def fft_slab(slab):
        return jnp.fft.ifftn(jnp.array(slab), axes=[axis])

    # Use shard_map to apply FFT to each slab in parallel
    #mesh = Mesh(jax.devices()[:num_cores], ('i',))
    #f_shmapped = shard_map(fft_slab, mesh, in_specs=P('i'), out_specs=P('i'))

    # Concatenate the results back into a single array

    #fft_slabs = f_shmapped(slabs)

    fft_slabs = [fft_slab(slab) for slab in slabs]

    #print("FFT Slabs shape:", [slab.shape for slab in fft_slabs])
    fft_field = jnp.array(fft_slabs).reshape(field.shape)
    fft_field = fft_field.transpose(axes[0], axes[1], axis)
    #print("FFT Field shape:", fft_field.shape)
    return fft_field

@partial(jax.jit, static_argnums=(1,2))
def fft_pencil_decomposition(field, axis=2, num_cores=2):
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

    axes = [0, 1, 2]
    axes.remove(axis)

    N1 = field.shape[axes[0]]
    N2 = field.shape[axes[1]]
    # Assume field is a 3D array with shape (Nx, Ny, Nz)
    pencils = jnp.split(field, N1, axis=axes[0])
    pencils = [jnp.split(pencil, N2, axis=axes[1]) for pencil in pencils]
    pencils = [pencil for sublist in pencils for pencil in sublist]  # Flatten the list of lists

    #print("Field shape:", field.shape)
    #print("Pencils shape:", [pencil.shape for pencil in pencils])
    def fft_pencil(pencil, axis=axis):
        return jnp.fft.fftn(jnp.array(pencil), axes=[axis])

    # Use shard_map to apply FFT to each pencil in parallel
    mesh = Mesh(jax.devices()[:num_cores], ('i',))
    f_shmapped = shard_map(fft_pencil, mesh, in_specs=P('i'), out_specs=P('i'))

    fft_pencils = f_shmapped(pencils)
    # Concatenate the results back into a single array

    # Reshape the FFT pencils back into the original 3D field shape
    fft_field = jnp.array(fft_pencils).reshape(field.shape)
    fft_field = fft_field.transpose(axes[0], axes[1], axis)
    return fft_field