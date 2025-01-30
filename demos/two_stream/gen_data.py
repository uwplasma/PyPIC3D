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

    vmax = 4000
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




# weight = (
#     constants['eps']
#     *me
#     *constants['C']**2
#     /q_e**2
#     *world['Nx']**2
#     /world['x_wind']
#     /(2*N_particles)
#     *plasma_parameters['Thermal Velocity']**2
#     /plasma_parameters['dx per debye length']**2).item()

# particles[0].set_weight(weight)
# # set the weight of the particles
# me = particles[0].get_mass()

# x_wind, y_wind, z_wind = world['x_wind'], world['y_wind'], world['z_wind']
# electron_x, electron_y, electron_z = particles[0].get_position()
# ev_x, ev_y, ev_z = particles[0].get_velocity()
# alternating_ones = (-1)**jnp.array(range(0,N_particles))
# relative_drift_velocity = 0.5*jnp.sqrt(3*constants['kb']*Te/me)
# perturbation = relative_drift_velocity*alternating_ones
# perturbation *= (1 + 0.1*jnp.sin(2*jnp.pi*electron_x/x_wind))
# ev_x = perturbation
# ev_y = jnp.zeros(N_particles)
# ev_z = jnp.zeros(N_particles)
# particles[0].set_velocity(ev_x, ev_y, ev_z)

# #electron_x = jnp.zeros(N_particles)
# electron_y = jnp.zeros(N_particles)# jnp.ones(N_particles) * y_wind/4*alternating_ones
# electron_z = jnp.zeros(N_particles)
# particles[0].set_position(electron_x, electron_y, electron_z)
# #put electrons with opposite velocities in the same position along y

# np.save('electron_x_positions.npy', electron_x)
# np.save('electron_y_positions.npy', electron_y)
# np.save('electron_z_positions.npy', electron_z)
# np.save('electron_x_velocities.npy', ev_x)
# np.save('electron_y_velocities.npy', ev_y)
# np.save('electron_z_velocities.npy', ev_z)
# exit()