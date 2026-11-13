"""Write quiet charged dust and common-time physical Yee D inputs.

Run after initial_metric.py. Initial_v files store covariant momenta u_i.
"""
from pathlib import Path
from itertools import product
import jax
import jax.numpy as jnp
from scipy.stats import qmc
from initial_metric import NX, HALF_WIDTH, OUTPUT_DIR

PARTICLE_COUNT = 262144


def main():
    jax.config.update('jax_enable_x64',True)
    data = jnp.load(OUTPUT_DIR/'radial_profile.npz')
    r,psi = data['r'],data['psi']
    count = PARTICLE_COUNT//8
    sample = jnp.asarray(qmc.Sobol(3,scramble=False).random_base2(count.bit_length()-1))+.5/count
    R = jnp.interp(sample[:,0]*data['seed_mass'][-1],data['seed_mass'],data['seed_R'])
    radius = jnp.interp(R,jnp.concatenate((jnp.zeros(1),r*psi**2)),jnp.concatenate((jnp.zeros(1),r)))
    cosine = sample[:,1]
    sine = jnp.sqrt(1-cosine**2)
    phi = .5*jnp.pi*sample[:,2]
    octant = radius[:,None]*jnp.stack((sine*jnp.cos(phi),sine*jnp.sin(phi),cosine),axis=-1)
    signs = jnp.asarray(tuple(product((-1.,1.),repeat=3)))
    positions = (signs[:,None,:]*octant[None,:,:]).reshape((-1,3))
    for i,axis in enumerate('xyz'):
        jnp.save(OUTPUT_DIR/f'charged_dust_initial_{axis}.npy',positions[:,i])
        jnp.save(OUTPUT_DIR/f'charged_dust_initial_v{axis}.npy',jnp.zeros(PARTICLE_COUNT))
    dx = 2*HALF_WIDTH/NX
    nodes = -HALF_WIDTH+jnp.arange(NX)*dx
    W = jnp.load(OUTPUT_DIR/'conformal_factor.npy')
    for i,axis in enumerate('xyz'):
        coordinates = jnp.meshgrid(*(nodes+.5*dx if j==i else nodes for j in range(3)),indexing='ij')
        radius = jnp.sqrt(sum(x*x for x in coordinates))
        density = jnp.interp(radius,r,psi**2*data['E_r'])*coordinates[i]/radius
        # Native C-to-face interpolation, with constant outer metric halos.
        upper = jnp.take(W,jnp.minimum(jnp.arange(NX)+1,NX-1),axis=i)
        w_face = .5*(W+upper)
        jnp.save(OUTPUT_DIR/f'physical_D_{axis}.npy',density*w_face**3)
    print(f'Saved {PARTICLE_COUNT} particles; mass={float(data["mass_factor"]):.17g}, weight={float(data["seed_mass"][-1])/PARTICLE_COUNT:.17g}')


if __name__ == '__main__':
    main()
