"""One-tile numerical and physical checks."""

from tests.support.pec_projector_fixtures import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    EdgeCoupledProjectionFixtures,
    StaggeredProjectorsFixtures,
    _pec_setup,
    _prepare_normal_D,
    _project_native_nodes,
    apply_tiled_pec_boundary,
    build_yee_metric,
    compute_covariant_E,
    compute_covariant_H,
    copy_densities,
    coupled_setup,
    divergence,
    jnp,
    make_setup,
    np,
    partial,
    reconstruct_vector,
    refresh_fields,
    unittest,
    update_B,
    update_D,
)


class TestStaggeredProjectors(StaggeredProjectorsFixtures, unittest.TestCase):
    def test_offdiagonal_normal_D_and_normal_shift_auxiliary(self):
        gamma=np.array([[1.25,.5,0.],[.5,1.,0.],[0.,0.,1.]])
        s,d,m=make_setup(gamma=gamma,shift=(.3,0.,0.))
        shape=m.center.lapse.shape;g=s.guard_cells;n=s.tile_shape[0]
        D=tuple(jnp.full(shape,v) for v in (2.,7.,8.))
        B=tuple(jnp.full(shape,v) for v in (3.,0.,4.))
        Dp=refresh_fields(D,s,D_FIELD_LOCATIONS,'D',m)
        Bp=refresh_fields(B,s,B_FIELD_LOCATIONS,'B',m)
        for x in (g,g+n):
            idx=(0,0,0,x,g+2,g)
            self.assertAlmostEqual(float(Dp[1][idx]),-1.)
            self.assertEqual(float(Dp[2][idx]),0.)
            self.assertEqual(float(Bp[0][idx]),0.)
        E=compute_covariant_E(Dp,Bp,m)
        self.assertGreater(abs(float(E[1][0,0,0,g,g+2,g])),.1)
        Er=refresh_fields(E,s,D_FIELD_LOCATIONS)
        np.testing.assert_array_equal(Er[1][0,0,0,:g,g+2,g],E[1][0,0,0,:g,g+2,g])
        for i in range(3):
            np.testing.assert_array_equal(D[i],(2.,7.,8.)[i]);np.testing.assert_array_equal(B[i],(3.,0.,4.)[i])
        with self.assertRaisesRegex(ValueError,'requires a Yee metric'):
            refresh_fields(D,s,D_FIELD_LOCATIONS,'D')


    def test_reflection_uses_surface_metric_not_ghost_metric(self):
        s,d,_=make_setup((8,4,1),3)
        def metric_at(p):
            shear=.2+.4*p[0]
            gamma=jnp.array([[1+shear**2,shear,0.],[shear,1.,0.],[0.,0.,1.]])
            return 1.,jnp.zeros(3),gamma,jnp.linalg.inv(gamma),1.
        m=build_yee_metric(d,metric_at);g=s.guard_cells;n=s.tile_shape[0]
        v=tuple(jnp.full(m.center.lapse.shape,x) for x in (2.,5.,0.))
        D=refresh_fields(v,s,D_FIELD_LOCATIONS,'D',m)
        for wall,ghost in ((g,g-1),(g+n,g+n+1)):
            x=float(d.grids.tiled_center_grid[0][0,0,0,wall])
            shear=.2+.4*x
            self.assertAlmostEqual(float(D[1][0,0,0,wall,g+1,g]),-2*shear)
            self.assertAlmostEqual(float(D[1][0,0,0,ghost,g+1,g]),-4*shear-5.)


if __name__ == "__main__":
    unittest.main()
