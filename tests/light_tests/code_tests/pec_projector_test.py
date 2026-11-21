"""One-tile API contracts."""

from tests.support.pec_projector_fixtures import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    StaggeredProjectorsFixtures,
    apply_tiled_pec_boundary,
    densitize_fields,
    jnp,
    make_setup,
    np,
    physical_fields,
    refresh_fields,
    unittest,
)


class TestStaggeredProjectors(StaggeredProjectorsFixtures, unittest.TestCase):


    def test_initial_normal_trace_ignores_stale_exterior(self):
        s,d,m=make_setup(gamma=[[1.25,.5,0.],[.5,1.,0.],[0.,0.,1.]])
        g=s.guard_cells;n=s.tile_shape[0]
        normal=jnp.full(m.center.lapse.shape,2.)
        normal=normal.at[:,:,:,:g].set(123.).at[:,:,:,g+n:].set(-456.)
        D=refresh_fields((normal,jnp.zeros_like(normal),jnp.zeros_like(normal)),s,D_FIELD_LOCATIONS,'D',m)
        for wall in (g,g+n):self.assertAlmostEqual(float(D[1][0,0,0,wall,g+2,g]),-1.)


    def test_auxiliary_refresh_preserves_all_computed_exterior_faces(self):
        s,d,m=make_setup((4,4,1),2,(1,3,0))
        shape=m.center.lapse.shape
        x=jnp.arange(shape[3])[None,None,None,:,None,None]
        y=jnp.arange(shape[4])[None,None,None,None,:,None]
        value=jnp.broadcast_to(x+10*y,shape).astype(float)
        for locs in (D_FIELD_LOCATIONS,B_FIELD_LOCATIONS):
            result=refresh_fields((value,)*3,s,locs)
            for component in result:
                np.testing.assert_array_equal(component[0,0,0,:,:,2],value[0,0,0,:,:,2])


if __name__ == "__main__":
    unittest.main()
