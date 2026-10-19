# Static-metric pair injection

`PyPIC3D.relativity.pair_injection.insert_pairs` inserts neutral pairs into existing
inactive particle slots. It only supports `solver="static_metric"`. The source
model chooses where, how often, and how much to inject; the shared routine handles
capacity, covariant birth momentum, and leapfrog initialization.

Candidates have shape `(tile_x, tile_y, tile_z, candidates, 3)` and must belong to
their supplied tiles. Positions are chart coordinates; momenta are orthonormal
spatial four-velocities at the integer birth time, before backward half-step
seeding. Requested pair counts have shape `(tile_x, tile_y, tile_z)`. Requests may
exceed candidate storage: the difference is reported as rejected, never silently
lost. The returned `counts = (requested, inserted)` contains two int32 arrays.
The newborn mask is internal to insertion and leapfrog seeding.

The selected species must have equal positive masses and weights and opposite
nonzero charges. Incompatible species reject all requested pairs, including
inside JIT. Both partners start at the same position with the same physical
momentum, so injection adds no deposited charge. Their seeded half-step momenta
can differ because their charges differ. Existing particles are unchanged.

Here is a small source that inserts one cold pair per tile per event:

```python
import jax
import jax.numpy as jnp
from PyPIC3D.relativity.pair_injection import insert_pairs
from PyPIC3D.__main__ import run_PyPIC3D


def inject(particles, species, fields, static, dynamic, key, event):
    g = static.guard_cells
    # An interior point of each tile; source models may instead sample a volume.
    point = jnp.stack([
        axis[..., g] + .5 * (axis[..., 1] - axis[..., 0])
        for axis in dynamic.grids.tiled_center_grid
    ], axis=-1)
    positions = point[..., None, :]
    momenta = jnp.zeros_like(positions)
    requested = jnp.ones(particles.active.shape[:3], dtype=jnp.int32)
    particles, counts = insert_pairs(
        particles, species, positions, momenta, requested,
        fields[0], fields[1], fields[6], static, dynamic,
        species_indices=(0, 1),
    )
    return particles, key, counts  # This deterministic source consumes no RNG.


# config is the usual loaded configuration dictionary. Configure two compatible
# species and enough fixed particle capacity for the intended experiment.
run_PyPIC3D(config, pair_injector=inject,
            injection_key=jax.random.PRNGKey(42), injection_interval_steps=10)
```

For random sources, split the key once per event and return the next key. Fold in
`event` and global cell/candidate indices for reproducible sampling. Use
`maxwell_juttner_momentum` from `PyPIC3D.particles.velocity_distribution` for
orthonormal spatial four-velocities. It shares the particle module's sampler
with `maxwell_juttner`, which returns three-velocities. For dimensionless
isotropic temperature T, pass `mass=kb=c=1` and `Tx=Ty=Tz=T`; all-zero
temperatures give cold momenta. The BZ model uses this shared sampler, so its
fixed-seed samples differ from the former demo rejection sampler.

A custom runner can pass the same callback directly to `time_loop_static_metric`:

```python
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric

particles, fields, key, counts = time_loop_static_metric(
    particles, species, fields, static, dynamic,
    pair_injector=inject, injection_key=key, step=step,
    injection_interval_steps=10,
)
```

Close over the callable and positive integer interval when using `jax.jit`, and
carry `step` and `key` as dynamic inputs. `jax.lax.cond` injects at step zero and
every interval thereafter; skipped steps preserve the key and return two zero count arrays.
With `return_diagnostics=True`, the output is
`(particles, fields, boundary_diagnostics, key, counts)`. Without an injector the
original solver output contract is unchanged. Custom runners must unpack
`requested, inserted = jax.device_get(counts)` and fail when any
`requested - inserted` is nonzero; `run_PyPIC3D` and the BZ runner already do this.

Injection precedes the original solver step and its old-position capture for
charge-conserving deposition. Supply the intended native densitized D/B fields
for seeding (including external fields if the source model requires them). The BZ
example retains its original entry-time D and staggered B inputs. This extraction
does not change the solver's field centering or deposition method.

The BZ model remains in `demos/static_metric_relativity/bz_monopole/plasma_injector.py`.
It retains Kerr geometry, magnetization gating, proper-volume sampling, and its
radial tiling restriction; these restrictions do not apply to the shared inserter.
