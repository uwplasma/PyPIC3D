"""Active-only hybrid batches preserve slots, species and checked execution."""
import jax
import jax.numpy as jnp
import numpy as np
import unittest
from tests.support.particle_metric_fixtures import make_runtime
from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push


class TestHybridBatching(unittest.TestCase):
    def test_explicit_single_tile_matches_vmapped_tiles(self):
        s,d,m,D,B=make_runtime('spherical',16,32)
        s=s._replace(particle_batch_size=2)
        particles=TiledParticles(jnp.array([2.,.42,.2]).reshape(1,1,1,1,1,3),
                                jnp.array([.01,.02,.003]).reshape(1,1,1,1,1,3),
                                jnp.ones((1,1,1,1,1),bool))
        species=SpeciesConfig(jnp.ones(1),jnp.ones(1),jnp.ones(1),jnp.ones((1,3),bool))
        def duplicate(tree):
            return jax.tree.map(lambda value: jnp.concatenate((value,value),axis=0)
                                if hasattr(value,'ndim') and value.ndim >= 3 else value, tree)
        def execute(p,dd,mm,ee,bb):
            return jax.jit(lambda p:hybrid_boris_geodesic_push(p,species,ee,bb,mm,s,dd))(p)
        single=execute(particles,d,m,D,B)
        # Duplicate supplied tile geometry so both lanes represent the same local
        # problem; the two-tile branch exercises the existing vmapped kernel.
        doubled=execute(duplicate(particles),duplicate(d),duplicate(m),duplicate(D),duplicate(B))
        for a,b in zip(jax.tree.leaves(single),jax.tree.leaves(doubled)):
            np.testing.assert_allclose(a,b[:1],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(a,b[1:],rtol=1e-12,atol=1e-12)

    def test_tiles_with_unequal_active_counts_match_isolated_tiles(self):
        s,d,m,D,B=make_runtime('spherical',16,32)
        s=s._replace(particle_batch_size=2)
        species=SpeciesConfig(jnp.ones(1),jnp.ones(1),jnp.ones(1),jnp.ones((1,3),bool))
        def make(active_slots):
            x=np.tile([2.,.7,.2],(1,1,1,1,4,1))
            x[...,1]+=.05*np.arange(4)
            u=np.tile([.01,.02,.003],(1,1,1,1,4,1))
            active=np.zeros((1,1,1,1,4),bool); active[...,active_slots]=True
            x[~active]=np.nan; u[~active]=np.nan
            return TiledParticles(jnp.array(x),jnp.array(u),jnp.array(active))
        def duplicate(tree):
            return jax.tree.map(lambda value: jnp.concatenate((value,value),axis=0)
                                if hasattr(value,'ndim') and value.ndim >= 3 else value, tree)
        def execute(p,dd,mm,ee,bb):
            return jax.jit(lambda p:hybrid_boris_geodesic_push(p,species,ee,bb,mm,s,dd))(p)
        # Three active particles in tile 0 and one in tile 1 need two batches and
        # one; each lane of the vmapped batch loop has to stop on its own count.
        crowded,sparse=make([0,1,2]),make([0])
        stacked=jax.tree.map(lambda a,b: jnp.concatenate((a,b),axis=0),crowded,sparse)
        together=execute(stacked,duplicate(d),duplicate(m),duplicate(D),duplicate(B))
        for lane,alone in ((0,execute(crowded,d,m,D,B)),(1,execute(sparse,d,m,D,B))):
            for a,b in zip(jax.tree.leaves(alone),jax.tree.leaves(together)):
                np.testing.assert_allclose(a,b[lane:lane+1],rtol=1e-12,atol=1e-12,equal_nan=True)


if __name__ == "__main__":
    unittest.main()
