import jax.numpy as jnp
import jax


N_electrons = 50000

x_wind = 1e-2
y_wind = 1e-2
z_wind = 1e-2
# window size
x = jnp.linspace( -x_wind/2, x_wind/2, N_electrons )
y = jnp.linspace( -y_wind/2, y_wind/2, N_electrons )
z = jnp.linspace( -z_wind/2, z_wind/2, N_electrons )
# initially uniformly distributed electrons

perturbation_wavenumber = 1.0
perturbation_amplitude = 1e-3

x_perturbation = perturbation_amplitude * jnp.sin( perturbation_wavenumber*x)
# create a perturbation in the x direction

x = x + x_perturbation
# add the perturbation to the x coordinate

jnp.save( 'x.npy', x )
jnp.save( 'y.npy', y )
jnp.save( 'z.npy', z )