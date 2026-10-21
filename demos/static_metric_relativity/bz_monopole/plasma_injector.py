"""Neutral thermal pair injection into fixed-capacity tiled particle storage."""
import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
from PyPIC3D.particles.velocity_distribution import maxwell_juttner_momentum
from PyPIC3D.relativity.pair_injection import insert_pairs


def empty_particles(p, static):
    shape = (p.devices, 1, 1, 2, p.slots_per_species)
    # Inactive positions must also have a finite metric: JAX evaluates masked lanes.
    x = jnp.broadcast_to(jnp.array([p.r_min+0.5*p.dr, 0.5*(p.theta_start+p.theta_end), 0.]), shape+(3,))
    particles = TiledParticles(x=x, u=jnp.zeros_like(x), active=jnp.zeros(shape, bool))
    species = SpeciesConfig(charge=jnp.array([-1., 1.]), mass=jnp.ones(2),
                            weight=jnp.full(2, p.weight), update_x=jnp.ones((2, 3), bool))
    return shard_tiled_particles(particles, static), species


def sample_position(key, rlo, rhi, tlo, thi, spin):
    """Sample n(r)*proper-volume with n proportional to r^-2, inside a cell."""
    bound=(1+spin**2/rlo**2)*jnp.sqrt(1+2/rlo)
    def draw(state):
        key,_,_=state;key,k=jax.random.split(key)
        u=jax.random.uniform(k,(4,),dtype=jnp.float64)
        r=rlo+(rhi-rlo)*u[0]
        mu=jnp.cos(tlo)+(jnp.cos(thi)-jnp.cos(tlo))*u[1]
        edge=jnp.nextafter(jnp.float64(1),jnp.float64(0))
        theta=jnp.arccos(jnp.clip(mu,-edge,edge))
        sig=r*r+spin*spin*mu*mu
        accept=u[3]*bound <= sig*jnp.sqrt(1+2*r/sig)/(r*r)
        return key,jnp.array([r,theta,2*jnp.pi*u[2]]),accept
    _,position,_=jax.lax.while_loop(lambda s:~s[2],draw,(key,jnp.array([rlo,(tlo+thi)/2,0.]),jnp.array(False)))
    return position


def inject_pairs(particles, species, magnetization, D, B, metric, static, dynamic,
                 parameters, key, step):
    """Insert complete pairs; call inside a JIT closing over static/parameters.

    Each event proposes n0/r^2 particles per proper volume (both species combined).
    Stochastic rounding preserves this expectation with fixed macroparticle weights.
    Capacity is per species, not per pair. The caller must fail if rejected != 0.
    D and B are the native densities the leapfrog seed pushes with.
    """
    p = parameters
    g = static.guard_cells
    nr, nt, _ = static.tile_shape
    nt += 1  # reflecting endpoint charge nodes are available for injection
    slots = particles.active.shape[-1]
    candidate_capacity = min(slots, nr*nt*(p.pairs_per_cell+2))
    interior = (slice(None),)*3 + (slice(g,-g),slice(g,-g+1),slice(g,-g))
    rgrid, tgrid, _ = dynamic.grids.tiled_center_grid
    sigma = magnetization.sigma[interior]
    valid = magnetization.valid[interior]
    next_key, event_key = jax.random.split(key)
    event_key = jax.random.fold_in(event_key, step)

    def one_tile(sig, valid, rline, tline, tile_id):
        rr, tt = jnp.meshgrid(rline[g:-g], tline[g:-g+1], indexing="ij")
        rr, tt = rr.reshape(-1), tt.reshape(-1)
        cells = jnp.arange(nr*nt)
        global_cells = tile_id*nr*nt+cells
        cell_keys = jax.vmap(lambda i: jax.random.fold_in(event_key, i))(global_cells)
        # Injection samples clipped charge-control volumes. Radial boundary
        # truncation is integrated with the same 8-point geometry quadrature.
        rlo=jnp.maximum(rr-p.dr/2,p.r_min); rhi=jnp.minimum(rr+p.dr/2,p.sponge_start)
        tlo=jnp.maximum(tt-p.dtheta/2,p.theta_start)
        thi=jnp.minimum(tt+p.dtheta/2,p.theta_end)
        from numpy.polynomial.legendre import leggauss
        nodes,weights=leggauss(8)
        integral=jnp.zeros_like(rr)
        for a,wa in zip(nodes,weights):
            radii=(rlo+rhi)/2+(rhi-rlo)*a/2
            for b,wb in zip(nodes,weights):
                angles=(tlo+thi)/2+(thi-tlo)*b/2
                geom_sig=radii**2+p.spin**2*jnp.cos(angles)**2
                integral+=wa*wb*geom_sig*jnp.sqrt(1+2*radii/geom_sig)*jnp.sin(angles)/radii**2
        expected = p.n0*integral*jnp.maximum(rhi-rlo,0)*(thi-tlo)*2*jnp.pi/4/(2*species.weight[0])
        eligible = valid.reshape(-1) & (sig.reshape(-1)>p.sigma_threshold) & (rr<p.sponge_start)
        rounding = jax.vmap(lambda k: jax.random.uniform(k, dtype=jnp.float64))(cell_keys)
        counts = jnp.where(eligible, jnp.floor(expected+rounding).astype(jnp.int32), 0)
        cumulative = jnp.cumsum(counts)
        requested = jnp.sum(counts)
        ranks = jnp.arange(candidate_capacity)
        cell = jnp.minimum(jnp.searchsorted(cumulative, ranks, side="right"), nr*nt-1)
        offset = ranks-jnp.where(cell>0, cumulative[jnp.maximum(cell-1, 0)], 0)
        keys = jax.vmap(jax.random.fold_in)(cell_keys[cell], offset)
        # Cell and ordinal keys make samples independent of the tile decomposition.
        positions=jax.vmap(lambda k,rl,rh,tl,th:sample_position(jax.random.fold_in(k,1),rl,rh,tl,th,p.spin))(
            keys,rlo[cell],rhi[cell],tlo[cell],thi[cell])
        def sample_momentum(key):
            # The BZ temperature is dimensionless: m = kb = c = 1.
            u = maxwell_juttner_momentum(
                1, 1., p.temperature, p.temperature, p.temperature, 1.,
                jax.random.fold_in(key, 2))
            return jnp.stack(u, axis=-1)[0]

        momenta = jax.vmap(sample_momentum)(keys)
        return positions, momenta, requested

    positions, momenta, requested = jax.vmap(one_tile)(
        sigma[:, 0, 0], valid[:, 0, 0], rgrid[:, 0, 0],
        tgrid[:, 0, 0], jnp.arange(p.devices))
    result, counts = insert_pairs(
        particles, species, positions[:, None, None], momenta[:, None, None],
        requested[:, None, None], D, B, metric, static, dynamic)
    return result, next_key, counts
