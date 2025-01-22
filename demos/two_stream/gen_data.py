import jax.numpy as jnp
from jax import random
if __name__ == "__main__":
    # Spatial Grid
    Nx = 50
    Ny = 50
    Nz = 50
    x_wind = 1e-3
    y_wind = 1e-3
    z_wind = 1e-3

    # Electrons
    N_electrons = 10000

    y = jnp.zeros(N_electrons)
    z = jnp.zeros(N_electrons)
    vy = jnp.zeros(N_electrons)
    vz = jnp.zeros(N_electrons)

    vmax = 2000
    alternating_ones = jnp.array( [ (-1)**i for i in range(N_electrons) ] )
    vx = vmax * alternating_ones
    # every other electron has a negative velocity
    key = random.PRNGKey(758493)  # Random seed is explicit in JAX
    # vx = vx*( 1 + 0.1*random.uniform(key, shape=(N_electrons,)) )
    vx = vx*( 1 + 0.1*jnp.sin(2*jnp.pi*jnp.linspace(0, 1, N_electrons)) )
    # add some random perturbations to the velocity

    x = jnp.linspace(-x_wind/2, x_wind/2, N_electrons)
    # uniformly placed along x

    # Save the data
    jnp.save('electron_x_positions.npy', x)
    jnp.save('electron_y_positions.npy', y)
    jnp.save('electron_z_positions.npy', z)
    jnp.save('electron_x_velocities.npy', vx)
    jnp.save('electron_y_velocities.npy', vy)
    jnp.save('electron_z_velocities.npy', vz)
