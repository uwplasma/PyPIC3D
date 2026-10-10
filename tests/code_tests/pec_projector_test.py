"""Single-device numerical tests."""

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
    densitize_fields,
    divergence,
    jnp,
    make_setup,
    np,
    partial,
    physical_fields,
    reconstruct_vector,
    refresh_fields,
    unittest,
    update_B,
    update_D,
)


class TestEdgeCoupledProjection(EdgeCoupledProjectionFixtures, unittest.TestCase):
    def test_refresh_is_idempotent_with_coupled_edges(self):
        cases = (((16, 8, 1), (16, 8, 1), (1, 1, 0)),    # x/y walls
                 ((8, 6, 8), (8, 6, 8), (1, 1, 1)))     # x/y/z walls
        rng = np.random.default_rng(0)
        for n, tiles, bc in cases:
            s, d, m = coupled_setup(n, tiles, bc)
            shape = m.center.lapse.shape
            for kind, locations in (('D', D_FIELD_LOCATIONS), ('B', B_FIELD_LOCATIONS)):
                with self.subTest(walls=bc, field=kind):
                    vector = tuple(jnp.asarray(rng.normal(size=shape)) for _ in range(3))
                    once = refresh_fields(vector, s, locations, kind, m)
                    twice = refresh_fields(once, s, locations, kind, m)
                    for first, second in zip(once, twice):
                        np.testing.assert_allclose(second, first, rtol=0, atol=1e-12 * float(jnp.abs(first).max()))


    def test_D_edge_rows_hold_simultaneously(self):
        """Recomputing the x-wall normal row of D^y from the refreshed y-wall D^x reproduces it."""
        s, d, m = coupled_setup((16, 8, 1), (16, 8, 1), (1, 1, 0))
        vector = tuple(jnp.asarray(np.random.default_rng(1).normal(size=m.center.lapse.shape)) for _ in range(3))
        refreshed = refresh_fields(vector, s, D_FIELD_LOCATIONS, 'D', m)
        axes, exchange = _pec_setup(refreshed[0].shape, s, D_FIELD_LOCATIONS, m)
        volumes = tuple(sample.sqrt_gamma for sample in m.D)
        prepare = partial(_prepare_normal_D, static=s, axes=axes)
        snapshot = exchange(copy_densities(prepare, exchange(refreshed), volumes))
        one_pass = _project_native_nodes(snapshot, s, D_FIELD_LOCATIONS, 'D', m, axes)
        g = s.guard_cells
        for component in (0, 1):
            np.testing.assert_allclose(one_pass[component][0, 0, 0, g:g+2, g:g+2, g],
                                       refreshed[component][0, 0, 0, g:g+2, g:g+2, g], atol=1e-13)



class TestStaggeredProjectors(StaggeredProjectorsFixtures, unittest.TestCase):
    def test_endpoints_all_axes_guards_and_narrow_tiles(self):
        for g in (1,2,3):
            for axis in range(3):
                n=[1,1,1];n[axis]=2
                bc=[0,0,0];bc[axis]=1
                s,d,m=make_setup(tuple(n),g,tuple(bc))
                original=tuple(jnp.full(m.center.lapse.shape,i+1.) for i in range(3))
                for locs,kind in ((D_FIELD_LOCATIONS,'D'),(B_FIELD_LOCATIONS,'B')):
                    result=refresh_fields(original,s,locs,kind,m)
                    for i,loc in enumerate(locs):
                        index=[0,0,0,g,g,g]
                        if loc[axis]=='C':
                            for wall in (g,g+n[axis]):
                                index[axis+3]=wall
                                self.assertEqual(float(result[i][tuple(index)]),0.)
                            index[axis+3]=g+n[axis]-1
                            self.assertEqual(float(result[i][tuple(index)]),i+1.)
                        # Diagonal metric reflection is exactly the C/V parity.
                        index[axis+3]=g-1; owner=index.copy()
                        owner[axis+3]=g if loc[axis]=='V' else g+1
                        parity=(1 if i==axis else -1) if kind=='D' else (-1 if i==axis else 1)
                        self.assertEqual(float(result[i][tuple(index)]),parity*float(result[i][tuple(owner)]))


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


    def test_corners_intersect_and_yee_last_interior_is_free(self):
        s,d,m=make_setup((4,4,4),2,(1,1,1),gamma=[[2.,.4,.2],[.4,1.5,.3],[.2,.3,1.]])
        values=tuple(jnp.full(m.center.lapse.shape,i+1.) for i in range(3))
        D=refresh_fields(values,s,D_FIELD_LOCATIONS,'D',m)
        E=apply_tiled_pec_boundary(values,s)
        for i,loc in enumerate(D_FIELD_LOCATIONS):
            others=[a for a in range(3) if a!=i]
            for lo in (2,6):
                for hi in (2,6):
                    idx=[0,0,0,3,3,3];idx[others[0]+3]=lo;idx[others[1]+3]=hi
                    self.assertEqual(float(D[i][tuple(idx)]),0.)
            idx=[0,0,0,3,3,3];idx[others[0]+3]=5
            self.assertEqual(float(E[i][tuple(idx)]),i+1.)
            idx[others[0]+3]=6;self.assertEqual(float(E[i][tuple(idx)]),0.)


    def test_varying_metric_boundary_residual_converges(self):
        errors=[]
        for n in (8,16,32):
            s,d,_=make_setup((n,n,1),2)
            def metric_at(p):
                shear=.3+.15*jnp.cos(2*jnp.pi*p[1])+.1*p[0]
                gamma=jnp.array([[1+shear**2,shear,0.],[shear,1.,0.],[0.,0.,1.]])
                return 1.,jnp.zeros(3),gamma,jnp.linalg.inv(gamma),1.
            m=build_yee_metric(d,metric_at)
            v=tuple(jnp.ones_like(m.center.lapse)*(i+1) for i in range(3))
            result=refresh_fields(v,s,D_FIELD_LOCATIONS,'D',m)
            collocated=jnp.stack(reconstruct_vector(result,D_FIELD_LOCATIONS,('C',)*3),axis=-1)/m.center.sqrt_gamma[...,None]
            residual=jnp.einsum('...ij,...j->...i',m.center.gamma,collocated)[...,1:]
            errors.append(float(jnp.max(jnp.abs(residual[0,0,0,2,3:2+n-1,2]))))
        for coarse,fine in zip(errors,errors[1:]):self.assertGreater(coarse/fine,3.5,errors)


    def test_updated_stages_sponge_and_projected_divergence(self):
        """Uniform fields, flat metric, x walls: projection adds no div D and only wall div B."""
        s,d,m=make_setup((6,6,1),2)
        s=s._replace(supergaussian_active=True,supergaussian_layers=((1,1,2,4.,2.),))
        shape=m.center.lapse.shape;g=s.guard_cells;n=s.tile_shape[0]
        fields=tuple(jnp.ones(shape)*(i+1.) for i in range(3));zero=(jnp.zeros(shape),)*3
        B=update_B(zero,fields,m,s,d,.01)
        D=update_D(fields,zero,zero,m,s,d,.01)
        for wall in (g,g+n):
            np.testing.assert_array_equal(B[0][0,0,0,wall,g:-g,g],0.)
            for i in (1,2):np.testing.assert_array_equal(D[i][0,0,0,wall,g:-g,g],0.)
        projected_B=refresh_fields(fields,s,B_FIELD_LOCATIONS,'B',m)
        projected_D=refresh_fields(fields,s,D_FIELD_LOCATIONS,'D',m)
        # Zeroing the uniform tangential D^y, D^z on the walls leaves Gauss's law unchanged.
        gauss=divergence(projected_D,d)-divergence(fields,d)
        np.testing.assert_allclose(gauss[0,0,0,g:-g,g:-g,g],0.,atol=1e-12)
        # Zeroing normal B^x=1 on the wall nodes changes div B by +-B^x/dx in the
        # first and last interior cells only.
        change=divergence(projected_B,d,forward=True)-divergence(fields,d,forward=True)
        expected=np.zeros((n,s.tile_shape[1]))
        expected[0],expected[-1]=1./d.dx,-1./d.dx
        np.testing.assert_allclose(change[0,0,0,g:-g,g:-g,g],expected,rtol=1e-12,atol=1e-12)


    def test_initial_normal_trace_ignores_stale_exterior(self):
        s,d,m=make_setup(gamma=[[1.25,.5,0.],[.5,1.,0.],[0.,0.,1.]])
        g=s.guard_cells;n=s.tile_shape[0]
        normal=jnp.full(m.center.lapse.shape,2.)
        normal=normal.at[:,:,:,:g].set(123.).at[:,:,:,g+n:].set(-456.)
        D=refresh_fields((normal,jnp.zeros_like(normal),jnp.zeros_like(normal)),s,D_FIELD_LOCATIONS,'D',m)
        for wall in (g,g+n):self.assertAlmostEqual(float(D[1][0,0,0,wall,g+2,g]),-1.)


    def test_particle_gather_and_previous_levels_are_projected(self):
        from unittest.mock import patch
        from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
        s,d,m=make_setup()
        value=jnp.ones(m.center.lapse.shape);z=(value*0,)*3
        # the loop expects every incoming D and B to be refreshed already
        vD=refresh_fields((value,)*3,s,D_FIELD_LOCATIONS,'D',m)
        vB=refresh_fields((value,)*3,s,B_FIELD_LOCATIONS,'B',m)
        fields=densitize_fields((vD,vB,z,value*0,value*0,(vD,vB),m,(vD,vB),False))
        captured=[]
        def push(particles,species,D,B,*args):
            captured.append((D,B));return particles,particles
        module='PyPIC3D.solvers.GR_yee.time_loop.'
        with patch(module+'hybrid_boris_geodesic_push',side_effect=push), \
             patch(module+'GR_direct_deposition',return_value=z), \
             patch(module+'refresh_tiled_particle_tiles',side_effect=lambda p,*args:(p,False)):
            _,result=time_loop_static_metric(None,None,fields,s,d)
        result = physical_fields(result)
        for D,B in (*captured,(result[0],result[1]),result[7]):
            for wall in (s.guard_cells,s.guard_cells+s.tile_shape[0]):
                idx=(0,0,0,wall,s.guard_cells+2,s.guard_cells)
                self.assertEqual(float(B[0][idx]),0.)
                self.assertEqual(float(D[1][idx]),0.)
                self.assertEqual(float(D[2][idx]),0.)


    def test_flat_stages_agree_with_yee_for_compatible_fields(self):
        from PyPIC3D.solvers.yee.first_order_yee import update_E as yee_update_E, update_B as yee_update_B
        s,d,m=make_setup((8,8,1),2,(1,1,0))
        shape=m.center.lapse.shape
        x=d.grids.tiled_center_grid[0][..., :,None,None]
        y=d.grids.tiled_center_grid[1][...,None,:,None]
        z=jnp.zeros(shape)
        E=(z,z,jnp.broadcast_to(jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y),shape))
        E=apply_tiled_pec_boundary(E,s);D=refresh_fields(E,s,D_FIELD_LOCATIONS,'D',m)
        By=(z,)*3;Bg=By;J=By
        for _ in range(3):
            By,_=yee_update_B(E,By,s,d)
            Bg=update_B(compute_covariant_E(D,Bg,m),Bg,m,s,d,d.dt/2)
            E,_=yee_update_E(E,By,J,s,d)
            D=update_D(D,compute_covariant_H(D,Bg,m),J,m,s,d,d.dt)
            for a,b in zip((*E,*By),(*compute_covariant_E(D,Bg,m),*Bg)):
                np.testing.assert_allclose(a[:,:,:,2:-2,2:-2,2],b[:,:,:,2:-2,2:-2,2],atol=1e-12)


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
