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
        from PyPIC3D.particles.particle_class import TiledParticles
        # Nonempty storage selects the mocked particle push/deposition branch.
        position=jnp.zeros((1,1,1,1,1,3))
        particles=TiledParticles(position,position,jnp.ones(position.shape[:-1],dtype=bool))
        module='PyPIC3D.solvers.GR_yee.time_loop.'
        with patch(module+'hybrid_boris_geodesic_push',side_effect=push), \
             patch(module+'GR_direct_deposition',return_value=z), \
             patch(module+'refresh_tiled_particle_tiles',side_effect=lambda p,*args:(p,False)):
            _,result=time_loop_static_metric(particles,None,fields,s,d)
        result = physical_fields(result)
        for D,B in (*captured,(result[0],result[1]),result[7]):
            for wall in (s.guard_cells,s.guard_cells+s.tile_shape[0]):
                idx=(0,0,0,wall,s.guard_cells+2,s.guard_cells)
                self.assertEqual(float(B[0][idx]),0.)
                self.assertEqual(float(D[1][idx]),0.)
                self.assertEqual(float(D[2][idx]),0.)

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
