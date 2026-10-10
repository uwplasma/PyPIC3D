"""Small supplied-metric fixtures for production regression tests."""
from types import SimpleNamespace
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from PyPIC3D.utilities.parameters import build_static_parameters, build_dynamic_parameters
from PyPIC3D.utilities.grids import build_yee_grid, build_tiled_yee_grids
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, Metric, build_yee_metric
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.relativity.metrics.flat import initialize_flat_cartesian_metric, initialize_flat_spherical_metric
from PyPIC3D.pusher.hybrid_boris_geodesic import magnetic_boris_rotation, gather_vector
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from tests.kernel_fixtures import kernel_parameters, empty_tiled_vector

PERIOD = 2*np.pi*np.sqrt(1.16)


def consumer_runtime(shape_factor=1, tile_shape=None):
    """Small numerical metric and off-node particles; no demo evolution."""
    s, d = kernel_parameters(
        Nx=8, Ny=8, Nz=1, x_min=1., y_min=.5, z_min=0.,
        x_wind=1., y_wind=1., z_wind=1., dt=.005,
        solver='static_metric', particle_pusher='hybrid_boris_geodesic',
        metric='numerical', shape_factor=shape_factor, tile_shape=tile_shape,
    )

    def provider(position):
        x, y, _ = position
        gamma = jnp.array([[2.+.2*jnp.sin(x), .2, .1],
                           [.2, 3.+.1*jnp.cos(y), -.1], [.1, -.1, 1.5]])
        lapse = 1.+.03*jnp.sin(x)+.02*jnp.cos(y)
        shift = jnp.array([.04*x*y, .02*y, .01*x])
        return lapse, shift, gamma, jnp.linalg.inv(gamma), jnp.sqrt(jnp.linalg.det(gamma))

    metric = build_yee_metric(d, provider)
    # The first particle's midpoint crosses x=1.5, the seam in the two-tile fixture.
    x = jnp.array([[1.4999, .82, 0.], [1.63, 1.12, 0.]])
    u = jnp.array([[.6, .2, -.1], [-.2, .1, .3]])
    ntx = 8 // s.tile_shape[0]
    slots = 2 // ntx
    shape = (ntx, 1, 1, 1, slots)
    particles = TiledParticles(x.reshape(shape+(3,)), u.reshape(shape+(3,)), jnp.ones(shape, bool))
    species = SpeciesConfig(jnp.ones(1), jnp.ones(1), jnp.ones(1), jnp.ones((1,3), bool))
    zeros = empty_tiled_vector(s, d)
    D = tuple(a+v for a, v in zip(zeros, (.02, .01, 0.)))
    B = tuple(a+v for a, v in zip(zeros, (0., 0., .4)))
    return s, d, metric, D, B, particles, species


def manufactured(offset=0.0):
    """Smooth, non-diagonal numerical metric on a 15x15x7 grid (radial origin shifted by offset)."""
    grid = (jnp.arange(-3, 12, dtype=float)*0.1 + offset,
            jnp.arange(-3, 12, dtype=float)*0.1,
            jnp.arange(7, dtype=float))
    x, y, z = jnp.meshgrid(*grid, indexing='ij')
    g = jnp.broadcast_to(jnp.array([[2.0, 0.2, 0.1], [0.2, 3.0, -0.1], [0.1, -0.1, 1.5]]), x.shape + (3, 3))
    g = g.at[..., 0, 0].add(0.2*x + 0.1*y*y).at[..., 0, 1].add(0.03*x*y).at[..., 1, 0].add(0.03*x*y)
    lapse = 1.0 + 0.03*x + 0.02*y*y
    shift = jnp.stack((0.04*x*y, 0.02*y, 0.01*x), axis=-1)
    return grid, Metric(lapse, shift, g, jnp.zeros_like(g), jnp.zeros(x.shape))

@lru_cache(maxsize=1)
def make_runtime(chart, nr=64, ntheta=128):
    spherical=chart=='spherical'
    nx,ny=(nr,ntheta) if spherical else (ntheta,ntheta)
    cfg=dict(Nx=nx,Ny=ny,Nz=1,x_min=1. if spherical else -4.,y_min=.4 if spherical else -4.,
             z_min=0. if spherical else -4.,x_wind=3. if spherical else 8.,
             y_wind=np.pi-.8 if spherical else 8.,z_wind=2*np.pi if spherical else 8.,dt=PERIOD/4096)
    cfg.update(dx=cfg['x_wind']/nx,dy=cfg['y_wind']/ny,dz=cfg['z_wind'])
    s=build_static_parameters(dict(**cfg,solver='static_metric',metric='flat_spherical' if spherical else 'flat_cartesian',
        metric_mass=0.,metric_spin=0.,particle_pusher='hybrid_boris_geodesic',current_deposition='esirkepov',
        current_filter='none',shape_factor=1,guard_cells=3,tile_shape=(nx,ny,1),
        boundary_conditions=(3,3,0) if spherical else (3,3,3),
        particle_boundary_conditions=(2,2,0) if spherical else (2,2,2)))
    center,vertex=build_yee_grid(SimpleNamespace(**cfg))
    tc,tv=build_tiled_yee_grids(s,SimpleNamespace(**cfg,grids=SimpleNamespace(center=center,vertex=vertex)))
    d=build_dynamic_parameters(dict(**cfg,grids=dict(center=center,vertex=vertex,tiled_center_grid=tc,tiled_vertex_grid=tv)))
    m=(initialize_flat_spherical_metric if spherical else initialize_flat_cartesian_metric)(s,d)
    z=jnp.zeros_like(m.center.sqrt_gamma);B=[]
    for i,loc in enumerate(B_FIELD_LOCATIONS):
        rg=(tc if loc[0]=='C' else tv)[0][...,:,None,None]
        tg=(tc if loc[1]=='C' else tv)[1][...,None,:,None]
        b=(jnp.cos(tg) if i==0 else -jnp.sin(tg)/rg if i==1 else z) if spherical else (z+1 if i==2 else z)
        B.append(jnp.broadcast_to(b,z.shape))
    # The analytic uniform physical field includes regular metric halos.
    B = tuple(B)
    return s,d,m,(z,)*3,B


def tile_grids(d):
    return (tuple(a[0,0,0] for a in d.grids.tiled_center_grid),
            tuple(a[0,0,0] for a in d.grids.tiled_vertex_grid))


def sample_metric(q,m,s,d):
    center=jax.tree.map(lambda a:a[0,0,0],m.center)
    return interpolate_metric(center,q,tile_grids(d)[0],s.metric,(True,True,False),(3,3,3))


def gather_B(q,B,s,d):
    return gather_vector(tuple(b[0,0,0] for b in B),B_FIELD_LOCATIONS,q,*tile_grids(d),
                         s.shape_factor,(True,True,False),(3,3,3))


def norm(u,m):
    return jnp.einsum('...i,...ij,...j->...',u,m.gamma_inv,u)


def sampled_rotation_checks(ntheta=128):
    """Boris-rotation invariants in the interior and near both regular chart boundaries, for both shape orders."""
    rows=[];s,d,m,D,B=make_runtime('spherical',64,ntheta)
    for order in (1,2):
        s=s._replace(shape_factor=order)
        for label,t in [('away',.8),('lower_half',.4+float(d.dy)/2),('lower_close',.4+float(d.dy)*.05),
                        ('upper_half',np.pi-.4-float(d.dy)/2),('upper_close',np.pi-.4-float(d.dy)*.05)]:
            q=jnp.array([[2.37,t,.1]]);met=sample_metric(q,m,s,d)
            b=gather_B(q,B,s,d);u=jnp.einsum('...ij,j->...i',jnp.linalg.cholesky(met.gamma),jnp.array([.4,.3,.2]))
            for charge in (-1.,1.):
                for dt in (.01,.5,5.,50.):
                    v=magnetic_boris_rotation(u,b,met,jnp.array([charge]),dt)
                    back=magnetic_boris_rotation(v,b,met,jnp.array([charge]),-dt)
                    scale=jnp.sqrt(norm(u,met)*jnp.einsum('...i,...ij,...j->...',b,met.gamma,b))
                    rows.append(dict(shape=order,location=label,charge=charge,dt=dt,
                        inverse_defect=float(jnp.linalg.norm(met.gamma@met.gamma_inv-jnp.eye(3))),
                        norm_error=float(jnp.max(jnp.abs(norm(v,met)/norm(u,met)-1))),
                        parallel_error=float(jnp.max(jnp.abs(jnp.sum((v-u)*b,axis=-1))/scale)),
                        reversal_error=float(jnp.linalg.norm(back-u)/jnp.linalg.norm(u))))
    return rows
