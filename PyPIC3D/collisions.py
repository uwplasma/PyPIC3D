"""
Particle collisions on tiled particle storage.

``binary``         Coulomb collisions between particles sharing a cell (intra-
                   and inter-species): Takizuka-Abe pairing and variance, with
                   Nanbu's angular distribution as in Smilei (Perez 2012), which
                   stays accurate for large per-step scattering.
``background_mcc`` Monte Carlo collisions with a uniform Maxwellian neutral gas
                   (elastic, back, charge_exchange, excitation, ionization) as
                   in WarpX's background MCC module.

Velocities are treated non-relativistically; ``particles.u`` holds 3-velocity.
"""

import jax
import jax.numpy as jnp
import numpy as np

E_CHARGE = 1.602176634e-19
MCC_PROCESSES = ("elastic", "back", "charge_exchange", "excitation", "ionization")


def load_collisions_from_toml(config, species_names):
    """Translate ``[collisions.<name>]`` tables into hashable static specs."""
    index = {name: i for i, name in enumerate(species_names)}
    specs = []
    for name, c in config.get("collisions", {}).items():
        species = [index[s] for s in np.atleast_1d(c["species"])]
        if c["type"] == "binary":
            specs.append(("binary", species[0], species[-1], float(c.get("coulomb_log", 10.0))))
        elif c["type"] == "background_mcc":
            processes = []
            for p in (p for p in MCC_PROCESSES if p in c):
                energy, sigma = _cross_section(c[p]["cross_section"])
                products = [index[s] for s in c[p].get("product_species", [])] + [-1, -1]
                processes.append((p, float(c[p].get("energy", 0.0)), energy, sigma, products[0], products[1]))
            specs.append(("mcc", species[0], float(c["background_density"]),
                          float(c.get("background_temperature", 0.0)), float(c["background_mass"]), tuple(processes)))
        else:
            raise ValueError(f"Unknown collision type for {name}: {c['type']}")
    return tuple(specs)


def _cross_section(value):
    """Constant cross section [m^2], a two-column (eV, m^2) file, or an inline [[eV, m^2], ...] table."""
    if isinstance(value, (int, float)):
        return (0.0, 1.0e30), (float(value), float(value))
    table = np.loadtxt(value, comments="#", ndmin=2) if isinstance(value, str) else np.asarray(value, float)
    return tuple(table[:, 0].tolist()), tuple(table[:, 1].tolist())


def _isotropic(key, n):
    d = jax.random.normal(key, (n, 3))
    return d / jnp.linalg.norm(d, axis=-1, keepdims=True)


def _rotate(g, cos_t, key):
    """Rotate vectors g by polar angle acos(cos_t) about g, with uniform azimuth."""
    gn = jnp.linalg.norm(g, axis=-1, keepdims=True)
    n = g / jnp.maximum(gn, 1e-300)
    a = jnp.where(jnp.abs(n[:, :1]) < 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    e1 = jnp.cross(n, a)
    e1 = e1 / jnp.maximum(jnp.linalg.norm(e1, axis=-1, keepdims=True), 1e-300)
    e2 = jnp.cross(n, e1)
    phi = jax.random.uniform(key, cos_t.shape, maxval=2 * jnp.pi)[:, None]
    sin_t = jnp.sqrt(jnp.maximum(1 - cos_t**2, 0.0))[:, None]
    return gn * (cos_t[:, None] * n + sin_t * (jnp.cos(phi) * e1 + jnp.sin(phi) * e2))


def _nanbu_cos(s, r):
    """Sample cos(theta) from Nanbu's distribution with <cos theta> = exp(-s) (Perez 2012 fits)."""
    inv_a = 0.0056958 + s * (0.9560202 + s * (-0.508139 + s * (0.47913906 + s * (-0.12788975 + s * 0.02389567))))
    a = jnp.where(s < 3, 1 / jnp.maximum(inv_a, 1e-12), 3 * jnp.exp(-s))
    cos_t = 1 + jnp.log(r + (1 - r) * jnp.exp(-2 * a)) / a
    cos_t = jnp.where(s < 0.1, 1 + s * jnp.log(r), jnp.where(s < 6, cos_t, 2 * r - 1))
    return jnp.clip(cos_t, -1.0, 1.0)


def _binary(key, x, u, active, spec, species_config, dyn):
    """Takizuka-Abe collisions for one tile and one species pair (Smilei-style pairing)."""
    _, sa, sb, coulomb_log = spec
    intra = sa == sb
    pick = (lambda v: v[sa]) if intra else (lambda v: jnp.concatenate((v[sa], v[sb])))
    xs, us, act = pick(x), pick(u), pick(active)
    tag = jnp.arange(xs.shape[0]) >= u.shape[1]
    L = xs.shape[0]
    cell = jnp.floor(xs / jnp.stack((dyn.dx, dyn.dy, dyn.dz))).astype(jnp.int32)
    k_sort, k_round = jax.random.split(key)
    order = jnp.lexsort((jax.random.uniform(k_sort, (L,)), tag, cell[:, 2], cell[:, 1], cell[:, 0], ~act))
    # sort active particles by cell, then species, then randomly within each cell
    c, t, a = cell[order], tag[order], act[order]
    new = jnp.concatenate((jnp.array([True]), jnp.any(c[1:] != c[:-1], -1) | (a[1:] != a[:-1])))
    seg, pos = jnp.cumsum(new) - 1, jnp.arange(L)
    start = jax.ops.segment_min(pos, seg, L)
    na = jax.ops.segment_sum((a & ~t).astype(jnp.int32), seg, L)
    nb = jax.ops.segment_sum((a & t).astype(jnp.int32), seg, L)
    if intra:
        npairs = na // 2 + 2 * ((na % 2 == 1) & (na >= 3))
    else:
        npairs = jnp.where(jnp.minimum(na, nb) > 0, jnp.maximum(na, nb), 0)
    cum = jnp.cumsum(npairs)
    s = jnp.minimum(jnp.searchsorted(cum, pos, side="right"), L - 1)
    valid = pos < cum[-1]
    p, n_a, n_b, st = pos - (cum[s] - npairs[s]), na[s], nb[s], start[s]
    # enumerate pairs cell by cell: pair p of a cell maps to positions (i, j) and a round
    wa, wb = species_config.weight[sa], species_config.weight[sb]
    ma, mb = species_config.mass[sa], species_config.mass[sb]
    volume = dyn.dx * dyn.dy * dyn.dz
    if intra:
        m, last = n_a // 2, n_a - 1
        i = jnp.where(p < m, 2 * p, jnp.where(p == m, last - 1, last))
        j = jnp.where(p < m, 2 * p + 1, jnp.where(p == m, last, last - 2))
        rnd = jnp.where(p < m, 0, p - m + 1)
        density = wa * n_a / volume * jnp.where((n_a % 2 == 1) & (p >= m - 1), 0.5, 1.0)
        # odd cells collide their last three particles pairwise at half weight (Takizuka-Abe)
    else:
        lo, hi = jnp.maximum(jnp.minimum(n_a, n_b), 1), jnp.maximum(jnp.maximum(n_a, n_b), 1)
        i, j, rnd = p % jnp.maximum(n_a, 1), n_a + p % jnp.maximum(n_b, 1), p // lo
        density = jnp.maximum(wa, wb) * n_a * n_b / (hi * volume)
        # each particle of the larger group collides once; weights corrected by rejection below
    i = order[jnp.where(valid, st + i, 0)]
    j = order[jnp.where(valid, st + j, 0)]
    mu = ma * mb / (ma + mb)
    coef = (species_config.charge[sa] * species_config.charge[sb]) ** 2 * density * coulomb_log * dyn.dt / (
        8 * jnp.pi * dyn.eps**2 * mu**2)
    wmax = jnp.maximum(wa, wb)

    def round_(k, us):
        k1, k2, k3 = jax.random.split(jax.random.fold_in(k_round, k), 3)
        g = us[i] - us[j]
        var = coef / jnp.maximum(jnp.linalg.norm(g, axis=-1), 1e-300) ** 3
        cos_t = _nanbu_cos(2 * var, jax.random.uniform(k1, var.shape, minval=1e-300))
        # s = 2 <delta^2> is the Takizuka-Abe small-angle limit of <1 - cos(theta)>
        dg = _rotate(g, cos_t, k2) - g
        r = jax.random.uniform(k3, var.shape)
        sel = valid & (rnd == k)
        us = us.at[i].add(jnp.where(sel & (r < wb / wmax), mb / (ma + mb), 0.0)[:, None] * dg)
        return us.at[j].add(-jnp.where(sel & (r < wa / wmax), ma / (ma + mb), 0.0)[:, None] * dg)

    us = jax.lax.fori_loop(0, jnp.max(jnp.where(valid, rnd + 1, 0)), round_, us)
    return u.at[sa].set(us[: u.shape[1]]) if intra else u.at[sa].set(us[: u.shape[1]]).at[sb].set(us[u.shape[1]:])


def _insert(x, u, active, s, mask, xn, un):
    """Place masked new particles into free slots of species s; report overflow."""
    free_first = jnp.argsort(active[s], stable=True)
    rank = jnp.cumsum(mask) - 1
    ok = mask & (rank < jnp.sum(~active[s]))
    slot = jnp.where(ok, free_first[jnp.clip(rank, 0, mask.shape[0] - 1)], mask.shape[0])
    x = x.at[s, slot].set(xn, mode="drop")
    u = u.at[s, slot].set(un, mode="drop")
    return x, u, active.at[s, slot].set(True, mode="drop"), jnp.any(mask & ~ok)


def _mcc(key, x, u, active, spec, species_config, dyn):
    """Background-gas Monte Carlo collisions for one tile and one projectile species."""
    _, s, n_gas, t_gas, M, processes = spec
    S, m = u.shape[1], species_config.mass[s]
    keys = jax.random.split(key, 5 + 2 * len(processes))
    vn = jnp.sqrt(dyn.kb * t_gas / M) * jax.random.normal(keys[0], (S, 3))
    g = u[s] - vn
    gn = jnp.linalg.norm(g, axis=-1)
    mu = m * M / (m + M)
    energy = 0.5 * mu * gn**2 / E_CHARGE
    nu = jnp.stack([n_gas * jnp.interp(energy, jnp.asarray(e), jnp.asarray(sig)) * gn for _, _, e, sig, _, _ in processes])
    nu_cum = jnp.cumsum(nu, axis=0)
    hit = active[s] & (jax.random.uniform(keys[1], (S,)) < -jnp.expm1(-nu_cum[-1] * dyn.dt))
    which = jnp.sum(jax.random.uniform(keys[2], (S,)) * nu_cum[-1] > nu_cum, axis=0)
    # choose one process per colliding particle with probability nu_k / nu_total
    vcm = (m * u[s] + M * vn) / (m + M)
    new_u, overflow, births = u[s], jnp.asarray(False), []
    for k, (ptype, cost, _, _, se, si) in enumerate(processes):
        sel = hit & (which == k)
        g_after = jnp.sqrt(jnp.maximum(gn**2 - 2 * cost * E_CHARGE / mu, 0.0))[:, None]
        if ptype == "charge_exchange":
            v = vn
        elif ptype == "back":
            v = vcm - M / (m + M) * g
        elif ptype == "ionization":
            g_after = g_after / jnp.sqrt(2.0)
            # remaining energy shared equally by the scattered and ejected electrons
            v = vn + g_after * _isotropic(keys[5 + 2 * k], S)
            births += [(se, sel, vn + g_after * _isotropic(keys[6 + 2 * k], S)), (si, sel, vn)]
        else:
            v = vcm + M / (m + M) * g_after * _isotropic(keys[5 + 2 * k], S)
            # elastic and excitation scatter isotropically in the centre-of-mass frame
        new_u = jnp.where(sel[:, None], v, new_u)
    u = u.at[s].set(new_u)
    for target, mask, un in births:
        x, u, active, full = _insert(x, u, active, target, mask, x[s], un)
        overflow = overflow | full
    return x, u, active, overflow


def apply_collisions(particles, species_config, static_parameters, dynamic_parameters):
    """Apply every configured collision operator to all tiles; returns (particles, overflow)."""
    specs = static_parameters.collisions
    if not specs:
        return particles, jnp.asarray(False)
    if particles.key is None:
        raise ValueError("Collisions need TiledParticles.key (a jax.random.PRNGKey).")
    key, sub = jax.random.split(particles.key)
    tiles = particles.x.shape[:3]
    keys = jax.random.split(sub, int(np.prod(tiles))).reshape(tiles + (-1,))

    def tile(k, x, u, active):
        overflow = jnp.asarray(False)
        for n, spec in enumerate(specs):
            kn = jax.random.fold_in(k, n)
            if spec[0] == "binary":
                u = _binary(kn, x, u, active, spec, species_config, dynamic_parameters)
            else:
                x, u, active, full = _mcc(kn, x, u, active, spec, species_config, dynamic_parameters)
                overflow = overflow | full
        return x, u, active, overflow

    x, u, active, overflow = jax.vmap(jax.vmap(jax.vmap(tile)))(keys, particles.x, particles.u, particles.active)
    return particles._replace(x=x, u=u, active=active, key=key), jnp.any(overflow)
