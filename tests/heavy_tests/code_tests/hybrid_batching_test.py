"""Active-only hybrid batches preserve slots, species and checked execution."""
import jax
import jax.numpy as jnp
import numpy as np
import unittest
from tests.support.particle_metric_fixtures import make_runtime
from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push


class TestHybridBatching(unittest.TestCase):
    def test_sparse_partial_batches_match_full_batch(self):
        for shape in (1,2):
            with self.subTest(shape=shape):
                s,d,m,D,B=make_runtime('spherical',16,32)
                s=s._replace(shape_factor=shape)
                x=np.tile([2.,.7,.2],(1,1,1,2,9,1))
                x[...,1,1]=.42; x[...,7,1]=np.pi-.42
                u=np.tile([.01,.02,.003],(1,1,1,2,9,1))
                active=np.zeros((1,1,1,2,9),bool)
                active[...,0,[1,5,7]]=True; active[...,1,[0,6]]=True
                x[~active]=np.nan;u[~active]=np.nan
                particles=TiledParticles(jnp.array(x),jnp.array(u),jnp.array(active))
                species=SpeciesConfig(jnp.array([-1.,1.]),jnp.array([1.,2.]),jnp.ones(2),
                                      jnp.array([[True,True,True],[True,False,True]]))
                def execute(p,metric,batch):
                    return jax.jit(lambda p:hybrid_boris_geodesic_push(
                        p,species,D,B,metric,s._replace(particle_batch_size=batch),d))(p)
                expected=execute(particles,m,18)
                actual=execute(particles,m,4)
                for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
                    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True)
                for result in actual:
                    np.testing.assert_array_equal(np.asarray(result.x)[~active],x[~active])
                    np.testing.assert_array_equal(np.asarray(result.u)[~active],u[~active])
                # An all-inactive tile runs no batches, so a poisoned metric is
                # never sampled and the stored slots come back untouched.
                poison=m._replace(center=m.center._replace(gamma=jnp.full_like(m.center.gamma,jnp.nan)))
                empty=particles._replace(active=jnp.zeros_like(particles.active))
                for result in execute(empty,poison,4):
                    np.testing.assert_array_equal(np.asarray(result.x),x)


if __name__ == "__main__":
    unittest.main()
