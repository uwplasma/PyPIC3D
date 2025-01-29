import jax.numpy as jnp
import jax


N_electrons = 10000

x_wind = 1e-2
y_wind = 1e-2
z_wind = 1e-2

key = jax.random.PRNGKey( 0 )
key1 = jax.random.PRNGKey( 1 )
key2 = jax.random.PRNGKey( 2 )
x = jax.random.uniform( key, minval=-x_wind/2, maxval=x_wind/2, shape=(N_electrons,) )
y = jax.random.uniform( key1, minval=-y_wind/2, maxval=y_wind/2, shape = (N_electrons,) )
z = jax.random.uniform( key2, minval=-z_wind/2, maxval=z_wind/2, shape = (N_electrons,) )
# initially uniformly distributed electrons

perturbation_wavenumber = 1.0
perturbation_amplitude = 1e-1

x_perturbation = perturbation_amplitude * jnp.sin( 2*jnp.pi*perturbation_wavenumber*x/x_wind )
# create a perturbation in the x direction

x = x + x_perturbation
# add the perturbation to the x coordinate

jnp.save( 'x.npy', x )
jnp.save( 'y.npy', y )
jnp.save( 'z.npy', z )