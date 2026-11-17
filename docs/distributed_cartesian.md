# Coupled Cartesian runs across nodes

The distributed `dynamic_metric` path uses one logical field/particle/BSSN tile
per device. The three tile counts must match the global JAX mesh, and tile
widths must divide the physical cell counts. Both packages use the same mesh
object and `tile_x`, `tile_y`, `tile_z` axes. Finite differences and metric/source
transfers operate on local tiles; output copies only addressable device shards.
Multiprocess simulation initialization currently supports `dynamic_metric` only.
The same output routines also handle single-process runs on one or several
devices; their use does not add multiprocess support to other solvers.

PyPIC3D processes never exchange anything during output:
each one writes its own openPMD series, and
[`PyPIC3D.diagnostics.merge_openpmd`](#assembling-a-multiprocess-run) joins them
afterwards.

## Runtime and launch

Use the same Python/JAX versions on every node and double precision. The
ordinary openPMD PyPI wheel is all that is needed.

Install both updated checkouts into that same execution environment (the tested
JAX/JAXlib version is 0.9.2):

```bash
python -m pip install -e /path/to/updated/JAX_BSSN
python -m pip install -e /path/to/updated/PyPIC3D
```

Replace these example paths with the actual checkouts. PyPIC3D's existing
`relativity` extra points to the upstream GitHub package, which can predate
these changes; installing that extra alone does not ensure that
`JAX_BSSN.parallel` is available.

Set `PYPIC_DISTRIBUTED=1` before importing PyPIC3D: package startup initializes
JAX's distributed runtime before arrays are constructed. JAX detects Slurm and
Open MPI launches on its own, so an allocation on Perlmutter with two GPU nodes
needs no coordinator settings:

```bash
export PYPIC_DISTRIBUTED=1
export PYPIC_LOCAL_DEVICE_IDS=0
export JAX_ENABLE_X64=true
export PYTHONNOUSERSITE=1
srun --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 \
    python -m PyPIC3D --config coupled.toml
```

The scheduler must expose one selected GPU to each task for this command's
`PYPIC_LOCAL_DEVICE_IDS=0`. If all local GPUs are visible, omit that variable and
JAX selects the local-rank device.

Where JAX cannot detect the launcher, describe it explicitly. Every process needs
the same `PYPIC_COORDINATOR_ADDRESS` and `PYPIC_NUM_PROCESSES`, and its own
`PYPIC_PROCESS_ID`:

```bash
PYPIC_DISTRIBUTED=1 PYPIC_COORDINATOR_ADDRESS=10.0.0.1:12345 \
    PYPIC_NUM_PROCESSES=2 PYPIC_PROCESS_ID=0 JAX_ENABLE_X64=true \
    python -m PyPIC3D --config coupled.toml
```

The default initialization timeout is 120 seconds; `PYPIC_INITIALIZATION_TIMEOUT`
overrides it. Run all processes through the same simulation and output schedule.

## Native inputs and local callbacks

Existing field `.npy` files keep shape `(Nx, Ny, Nz)` without guards. Each process
memory-maps these files and reads only its addressable tile interiors. The
ordinary boundary preparation supplies physical wall values and tile halos.
Metric files keep the existing global C-node shape, including owned upper
endpoints where configured; they are likewise read by addressable tile.

Python callers can supply `field_initializers` to `initialize_simulation` or
`run_PyPIC3D`. Keys are `Ex`, `Ey`, `Ez`, `Bx`, `By`, `Bz`. A value is a callable
`initializer(x, y, z)` returning a scalar or broadcastable local array. Arguments
are coordinate lines broadcast to three dimensions at the component's native
Yee location, including its guards. Callbacks replace the corresponding file
initialization and are evaluated only for addressable tiles. They supply the
same physical vectors as native field inputs, before GR densitization/seeding.

Distributed particles currently require all six native one-dimensional files:
`initial_x`, `initial_y`, `initial_z`, `initial_vx`, `initial_vy`, `initial_vz`.
Their lengths must equal `N_particles`. Every process reads every bounded block
in file order and keeps only the particles whose spatial tile it owns, so tile
occupancy, capacity and the derived temperature come out identical everywhere
without an exchange. The cost is that each process reads the whole file.
Capacity and weights retain the serial convention. Built-in random particle
generation is explicitly unsupported for multiprocess initialization;
pre-generate the same native files when a random realization is required. Empty
particle storage is supported.

## Sharded field and particle output

`plot_openpmd_fields` writes `data/fields.h5`, and `plot_openpmd_particles` writes
`data/particles.h5`. A single-process run writes exactly those names, as before.
With N processes, each one writes its own `data/fields_p000.h5` …
`data/fields_pNNN.h5`; the same `_pNNN` suffix applies to the initial dumps.
Each process copies only its addressable device shards to host buffers, then
stores their chunks through bounded asynchronous queues. It does not assemble
global field or particle arrays on a host or device, and it does not wait on any
other process. `openpmd_field_queue_size` and `openpmd_particle_queue_size`
bound the pending local snapshots per process.

Every mesh component keeps the full `(Nx, Ny, Nz)` shape in every per-process
file, including singleton axes and mixed periodic/conducting boundaries, so each
file is a valid whole-domain mesh that ordinary readers open directly. Only the
tiles a process owns are stored; field records use tile-shaped HDF5 chunks, so a
file costs space proportional to the tiles it actually holds. Writers strip
guards and retain the existing physical-interior output selection and
component-position metadata. Read any file with the ordinary
`io.Series(path, io.Access.read_only)` constructor.

Particle records are sized to the particles a process actually holds and are
kept in global tile order within each file. Each file records which global tiles
it owns, and per species how many particles each of those tiles holds, as
`pypic3dOwnedTiles` / `pypic3dOwnedTileCounts` attributes. The HDF5 backend
reports every dataset as one whole chunk, so this metadata is the only way a
reader can tell which regions a file wrote. Scalar diagnostics use solver
reductions and are written by process zero.

`dump_fields` writes `data/initial_fields/initial_fields.h5`, and `dump_particles`
writes one `data/initial_particles/<species>_initial_particles.h5` file per
species. Initial particle dumps preserve the configured positions and velocities
before the leapfrog seed, with no diagnostic half-step position shift. These
initial dumps also work with `Nt=0`. Single-process initial histogram plots use
local shard extrema and 50-bin counts without flattening all particles.

Runtime snapshots retain the simulation step as the iteration identifier and
the existing solver time-centering and record time offsets. Output cadence and
the solver's final-endpoint rules are unchanged. These files are scientific
diagnostics, not complete restart checkpoints.

## Assembling a multiprocess run

A single-process run needs no assembly. After a multiprocess run, join each
group into the unsuffixed file every analysis script already reads:

```bash
python -m PyPIC3D.diagnostics.merge_openpmd outputs/data
```

The installed console script `PyPIC3D-merge-openpmd` does the same. Point it at
a data directory, one group member, or a `.pmd` pointer; `--output` names the
merged series, `--iterations` restricts it to listed steps, and `--overwrite`
replaces an existing merged file.

The merge rejects a group whose process indices are not exactly
`0 .. pypic3dProcessCount - 1`, which catches stale files left by an earlier run
with a different process count, and it checks that the owned tiles partition the
global tile grid exactly once. It then overlays each file's owned tiles into one
global-extent mesh, concatenates the particle records back into global tile order
independent of which process wrote what, and rewrites the `.pmd` pointer. A
multiprocess run leaves the pointer to the merge rather than writing one that
dangles until then.

## Validation

Run local input/chunk and initial histogram tests on two logical CPU devices:

```bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2 JAX_ENABLE_X64=true \
    python -m unittest tests.distributed_tests.distributed_io_test \
    tests.code_tests.sharded_histogram_test
```

Per-process naming, ownership metadata and the merge round trip run on one
process and need no launcher:

```bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=8 JAX_ENABLE_X64=true \
    python -m unittest tests.distributed_tests.merge_openpmd_test \
    tests.code_tests.sharded_openpmd_test
```

A real two-process check needs two shells sharing a coordinator address and a
directory visible to both, with `PYPIC_PROCESS_ID` 0 and 1. Confirm the run
wrote `fields_p000.h5` and `fields_p001.h5` and no bare `fields.h5`, then merge
and read the result with an unmodified analysis script.

