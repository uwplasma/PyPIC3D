"""Focused cross-device regression tests."""

from tests.support.pec_projector_fixtures import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    StaggeredProjectorsFixtures,
    jax,
    jnp,
    make_setup,
    np,
    refresh_fields,
    unittest,
)


class TestStaggeredProjectors(StaggeredProjectorsFixtures, unittest.TestCase):
    def test_transverse_endpoint_exchange_matches_single_device(self):
        if jax.device_count()<4:raise RuntimeError('requires four CPU devices')
        gamma=[[1.25,.5,0.],[.5,1.,0.],[0.,0.,1.]]
        results=[]
        tiled_results=[]
        for tile in ((4,4,1),(2,2,1)):
            s,d,m=make_setup((4,4,1),3,(1,1,0),tile,gamma)
            x=d.grids.tiled_center_grid[0][..., :,None,None]
            y=d.grids.tiled_center_grid[1][...,None,:,None]
            fields=tuple(jnp.broadcast_to((i+1)*(1+x+2*y),m.center.lapse.shape) for i in range(3))
            pair=[]
            tile_pair=[]
            for locs,kind in ((D_FIELD_LOCATIONS,'D'),(B_FIELD_LOCATIONS,'B')):
                values=refresh_fields(fields,s,locs,kind,m)
                tile_pair.extend(np.asarray(v) for v in values)
                joined=[]
                for value,loc in zip(values,locs):
                    rows=[]
                    for tx in range(value.shape[0]):
                        cols=[]
                        for ty in range(value.shape[1]):
                            ex=int(tx==value.shape[0]-1 and loc[0]=='C')
                            ey=int(ty==value.shape[1]-1 and loc[1]=='C')
                            cols.append(np.asarray(value)[tx,ty,0,3:3+tile[0]+ex,3:3+tile[1]+ey,3])
                        rows.append(np.concatenate(cols,axis=1))
                    joined.append(np.concatenate(rows,axis=0))
                pair.extend(joined)
            results.append(pair)
            tiled_results.append(tile_pair)
        for a,b in zip(*results):np.testing.assert_allclose(a,b,atol=1e-13)
        # Check endpoint planes in transverse halos as well as owned nodes.
        for single,distributed in zip(*tiled_results):
            for tx in range(2):
                for ty in range(2):
                    np.testing.assert_allclose(distributed[tx,ty,0],
                        single[0,0,0,2*tx:2*tx+8,2*ty:2*ty+8,:],atol=1e-13)


if __name__ == "__main__":
    unittest.main()
