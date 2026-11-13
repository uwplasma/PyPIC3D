# Coupled Cartesian runs across nodes

The distributed `dynamic_metric` path uses one logical field/particle/BSSN tile
per device. The three tile counts must match the global JAX mesh, and tile
widths must divide the physical cell counts. Both packages use the same mesh
object and `tile_x`, `tile_y`, `tile_z` axes. Finite differences and metric/source
transfers operate on local tiles; output copies only addressable device shards.
Multiprocess simulation initialization currently supports `dynamic_metric` only.
The same output routines also handle single-process runs on one or several
devices; their use does not add multiprocess support to other solvers.

## Runtime and launch

Use the same Python/JAX versions on every node, double precision, and an
MPI-enabled openPMD installation with parallel HDF5. `mpi4py`, openPMD and HDF5
must link to the same MPI implementation. The `distributed` optional dependency
installs mpi4py; the ordinary openPMD PyPI wheel can be serial and is not sufficient.
Check `openpmd_api.variants['mpi']` before allocating a production job. Disable a
user site that would shadow the intended cluster packages.

Install both updated checkouts into that same execution environment (the tested
JAX/JAXlib version is 0.9.2):

```bash
python -m pip install -e /path/to/updated/JAX_BSSN
python -m pip install -e '/path/to/updated/PyPIC3D[distributed]'
```

Replace these example paths with the actual checkouts containing the parallel
implementation. PyPIC3D's existing `relativity` extra points to the upstream
GitHub package, which can predate these changes; installing that extra alone
does not ensure that `JAX_BSSN.parallel` is available. Select the MPI-enabled
openPMD build after dependency installation so a serial wheel does not shadow it.

An allocation on Perlmutter with two GPU nodes can launch four ranks per node:

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
JAX selects the MPI local-rank device. Set `PYPIC_DISTRIBUTED` before importing
PyPIC3D: package startup initializes MPI/JAX before arrays are constructed.
The default initialization timeout is 120 seconds; the environment variable
`PYPIC_INITIALIZATION_TIMEOUT` overrides it. Run all ranks through the same
simulation and output schedule. MPI output is synchronous on the main thread;
single-process output uses the same chunk writers through bounded asynchronous
queues. No background writer thread enters MPI.

### Building an MPI-enabled openPMD runtime

Use the cluster's MPI wrappers and parallel HDF5 installation, with mpi4py
compiled against that same MPI. For openPMD 0.17.1, an isolated prefix can be
built as follows (set `HDF5_ROOT` to the parallel installation first):

```bash
curl -L https://github.com/openPMD/openPMD-api/archive/refs/tags/0.17.1.tar.gz \
    -o /tmp/openpmd-api-0.17.1.tar.gz
tar -xf /tmp/openpmd-api-0.17.1.tar.gz -C /tmp
cmake -S /tmp/openPMD-api-0.17.1 -B /tmp/openpmd-mpi-build \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/openpmd-mpi-install \
    -DopenPMD_USE_MPI=ON -DopenPMD_USE_HDF5=ON -DopenPMD_USE_ADIOS2=OFF \
    -DopenPMD_USE_PYTHON=ON -DopenPMD_BUILD_TESTING=OFF \
    -DopenPMD_BUILD_EXAMPLES=OFF -DopenPMD_BUILD_CLI_TOOLS=OFF \
    -DPython_EXECUTABLE="$(command -v python)" -DHDF5_ROOT="$HDF5_ROOT" \
    -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_CXX_COMPILER="$(command -v mpic++)"
cmake --build /tmp/openpmd-mpi-build --parallel 2
cmake --install /tmp/openpmd-mpi-build
```

Put that prefix's `lib/python<version>/site-packages` first on `PYTHONPATH` and
its `lib` first on `LD_LIBRARY_PATH`; include the MPI/HDF5 library directory as
required by the selected runtime. On a cluster, use an equivalent shared prefix
accessible from every node. Verify the selected module rather than relying on
the package name:

```bash
python -c 'import openpmd_api as io; print(io.__file__, io.variants); assert io.variants["mpi"]'
```

The historical local two-process validation described below used Python 3.14, JAX/JAXlib 0.9.2,
OpenMPI 5.0.10, parallel HDF5 1.14.6, and the above openPMD 0.17.1 build.
The build/install directories were `/tmp/openpmd-mpi-build` and
`/tmp/openpmd-mpi-install`; `/tmp/pypic-mpi-overlay` supplied the existing
MPI-matched mpi4py package. Those temporary paths are validation artifacts.
The ordinary installed openPMD package remained serial (`mpi=False`); a
production environment must select its own MPI-enabled build explicitly.

## Native inputs and local callbacks

Existing field `.npy` files keep shape `(Nx, Ny, Nz)` without guards. Each rank
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
Their lengths must equal `N_particles`. Bounded blocks are read once by assigned
input ranks, routed to the owning spatial tiles, and ordered by original file
index. Capacity and weights retain the serial convention. Built-in random
particle generation is explicitly unsupported for multiprocess initialization;
pre-generate the same native files when a random realization is required. Empty
particle storage is supported.

## Sharded field and particle output

`plot_openpmd_fields` writes `data/fields.h5`, and `plot_openpmd_particles` writes
`data/particles.h5`. These paths and record names are the same for one process
and for MPI runs. Each rank copies only its addressable device shards to host
buffers, then stores their chunks at the corresponding global offsets. It does
not assemble global field or particle arrays on a host or device. Single-process
queues retain at most the configured number of pending local snapshots;
`openpmd_field_queue_size` and `openpmd_particle_queue_size` do not limit file
counts and are not used by synchronous MPI output.

Every mesh component has shape `(Nx, Ny, Nz)`, including singleton axes and
mixed periodic/conducting boundaries. Writers strip guards and retain the
existing physical-interior output selection and component-position metadata.
The writer does not change component extents or add upper endpoint planes.
Read the resulting files with the ordinary
`io.Series(path, io.Access.read_only)` constructor.

Active particle records follow global tile order, independently of rank order
or local shard enumeration. Only integer per-tile/species counts are exchanged
to determine global record sizes and offsets; particle coordinates and momenta
remain local. Ranks with no active particles still participate in collective
metadata, flush, and close operations. Scalar diagnostics use solver reductions
and are written by rank zero.

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

MPI output requires an MPI-enabled openPMD build with parallel HDF5 and fails
early when that support is absent. A rank-local output failure aborts the MPI
communicator so other ranks cannot remain waiting in collective I/O. All ranks
must use the same output flags and cadence.

## Validation

Run local input/chunk and initial histogram tests on two logical CPU devices:

```bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2 JAX_ENABLE_X64=true \
    python -m unittest tests.distributed_tests.distributed_io_test \
    tests.code_tests.sharded_histogram_test
```

The MPI output acceptance worker requires an MPI-enabled openPMD build and a
directory visible to all ranks. Run it with one and then two local devices per
rank, using separate fresh directories:

```bash
PYPIC_DISTRIBUTED=1 JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=1 JAX_ENABLE_X64=true \
    timeout 120s mpiexec -n 2 python -m tests.distributed_tests.sharded_openpmd_worker \
    --directory /tmp/pypic-mpi-output-one-device

PYPIC_DISTRIBUTED=1 JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2 JAX_ENABLE_X64=true \
    timeout 120s mpiexec -n 2 python -m tests.distributed_tests.sharded_openpmd_worker \
    --directory /tmp/pypic-mpi-output-two-devices
```

The worker checks fields-only, particles-only and combined output, repeated
iterations, original initial-dump paths, unequal particle counts, an empty
species and a species absent on one rank. It also permutes local snapshot shard
lists, rejects transfers of global distributed arrays, and reads the files with
ordinary openPMD readers. The field fixture includes a singleton axis and mixed
boundary conditions.

Check failure propagation separately. This command must print
`WRITER_ABORT rank=1` and terminate promptly with a nonzero MPI exit status.
The marker verifies the writer's failure handler ran. An exit status of `124`
means the timeout fired and the failure test did not pass:

```bash
PYPIC_DISTRIBUTED=1 JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2 JAX_ENABLE_X64=true \
    timeout 120s mpiexec -n 2 python -m tests.distributed_tests.sharded_openpmd_worker \
    --directory /tmp/pypic-mpi-output-failure --fail-rank 1
```

On a two-node Perlmutter allocation, exercise the output worker with one GPU on
each node and a directory visible to both nodes:

```bash
export PYPIC_DISTRIBUTED=1 PYPIC_LOCAL_DEVICE_IDS=0 JAX_ENABLE_X64=true
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda
srun --nodes=2 --ntasks-per-node=1 --gpus-per-task=1 --gpu-bind=single:1 \
    python -m tests.distributed_tests.sharded_openpmd_worker \
    --directory "$SCRATCH/pypic-two-node-output-test"
```

An output worker does not establish coupled-evolution accuracy or scaling, and
a local MPI run does not establish two-node GPU operation. Record hostnames,
global/local device ownership and readback results when running on two real nodes.

### Unified writer validation

The unified writers passed 43 focused serial/distributed regression tests on
logical CPU devices, including initial and runtime GR time metadata. Actual
MPI output passed on two ranks with both one and two local devices per rank:
initial and runtime readback, unequal counts, empty species/ranks, reordered
shards, and fields-only/particles-only output. The injected rank-local failure
printed `WRITER_ABORT rank=1` and exited with status 1 before the timeout.

The updated `multihost_io_worker --require-mpi-output --evolve` also completed
two coupled steps with initial and runtime output. An otherwise identical run
with output disabled produced bitwise-identical final particle, field and BSSN
arrays on each rank. The BSSN state retained one addressable tile and 373,248
local bytes per rank. This used the MPI-enabled openPMD 0.17.1 runtime described
above and the current local JAX_BSSN checkout. A five-minute cold-compilation
limit was insufficient; the successful comparison used a 15-minute limit and
four separate physical CPU cores per rank. No multi-node or GPU validation was
performed for this writer change.

### Historical distributed-evolution validation

The earlier MPI draft passed a local run on 2026-11-10 with two CPU processes
on one host. That draft used `data/simulation.h5` and component-specific upper
endpoint extents; its output-format results do not validate the current unified
writers. The following measurements describe that earlier execution only. The
actual `initialize_simulation` / `run_PyPIC3D` path advanced two steps on an
`8 x 4 x 4` periodic mesh split into two `4 x 4 x 4` tiles, starting from flat
geometry, `metric_backreaction=True`, seven neutral unit-mass native-file
particles, and an analytic electric field. One rank initially owned all
particles. Matter sourcing produced nonzero `trace_K`; all final BSSN leaves
remained finite and distributed, with one addressable tile per process. This
fixture does not solve the initial Einstein constraints and tests execution and
communication rather than physical accuracy. The shared
series contained all 20 EM/BSSN records (30 components) at both saved iterations,
with finite values and the expected particle count. Separate analytic MPI
snapshots checked the physical upper endpoint planes and native E-component
extents. The final scalar particle-count metadata also passed a two-process
round trip. A separate two-process initialization/output check enabled both
`plotvelocities` and `plotchargedensity`; all four optional output components
were finite, retained local shards, and passed the shared-series readback.

For that tiny CPU case, explicit step compilation took 180.027 seconds using
the slowest process time. An MPI barrier preceded each timed step, and all
returned arrays were synchronized before stopping its timer. The first step
took 0.059005 seconds and the second 0.057823 seconds, again using the slowest
process time. The compiled per-device memory estimate was 1,358,745 bytes of
arguments, 2,055,766 bytes of outputs, and 20,896,936 bytes of temporary storage
(zero alias bytes). The local BSSN state occupied 373,248 bytes per process.
These figures exclude initialization and diagnostics, and do not establish
production scaling, accuracy, or GPU memory requirements. The ordinary
installed serial openPMD runtime was not changed.
