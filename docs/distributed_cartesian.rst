Multi-node Simulations
===================================

The distributed ``dynamic_metric`` path uses one logical field/particle/BSSN
tile per device. The three tile counts must match the global JAX mesh, and tile
widths must divide the physical cell counts. Both packages use the same mesh
object and ``tile_x``, ``tile_y``, ``tile_z`` axes. Finite differences and
metric/source transfers operate on local tiles; output copies only addressable
device shards. Multiprocess simulation initialization currently supports
``dynamic_metric`` only. The same output routines also handle single-process
runs on one or several devices; their use does not add multiprocess support to
other solvers.

PyPIC3D processes never exchange anything during output: each one writes its own
openPMD series, and ``PyPIC3D.diagnostics.merge_openpmd`` joins them afterwards,
as described in `Assembling a multiprocess run`_.

Runtime and launch
------------------

Use the same Python/JAX versions on every node and double precision. The
ordinary openPMD PyPI wheel is all that is needed.

Install both updated checkouts into that same execution environment (the tested
JAX/JAXlib version is 0.9.2):

.. code-block:: bash

   python -m pip install -e /path/to/updated/JAX_BSSN
   python -m pip install -e /path/to/updated/PyPIC3D

Replace these example paths with the actual checkouts. PyPIC3D's existing
``relativity`` extra points to the upstream GitHub package, which can predate
these changes; installing that extra alone does not ensure that
``JAX_BSSN.parallel`` is available.

Set ``PYPIC_DISTRIBUTED=1`` before importing PyPIC3D: package startup
initializes JAX's distributed runtime before arrays are constructed. JAX detects
Slurm and Open MPI launches on its own, so an allocation on Perlmutter with two
GPU nodes needs no coordinator settings:

.. code-block:: bash

   export PYPIC_DISTRIBUTED=1
   export PYPIC_LOCAL_DEVICE_IDS=0
   export JAX_ENABLE_X64=true
   export PYTHONNOUSERSITE=1
   srun --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 \
       python -m PyPIC3D --config coupled.toml

The scheduler must expose one selected GPU to each task for this command's
``PYPIC_LOCAL_DEVICE_IDS=0``. If all local GPUs are visible, omit that variable
and JAX selects the local-rank device.

Where JAX cannot detect the launcher, describe it explicitly. Every process
needs the same ``PYPIC_COORDINATOR_ADDRESS`` and ``PYPIC_NUM_PROCESSES``, and
its own ``PYPIC_PROCESS_ID``:

.. code-block:: bash

   PYPIC_DISTRIBUTED=1 PYPIC_COORDINATOR_ADDRESS=10.0.0.1:12345 \
       PYPIC_NUM_PROCESSES=2 PYPIC_PROCESS_ID=0 JAX_ENABLE_X64=true \
       python -m PyPIC3D --config coupled.toml

The default initialization timeout is 120 seconds;
``PYPIC_INITIALIZATION_TIMEOUT`` overrides it. Run all processes through the
same simulation and output schedule.


Assembling a multiprocess run
-----------------------------

A single-process run needs no assembly. After a multiprocess run, join each
group into the unsuffixed file every analysis script already reads:

.. code-block:: bash

   python -m PyPIC3D.diagnostics.merge_openpmd outputs/data

The installed console script ``PyPIC3D-merge-openpmd`` does the same. Point it
at a data directory, one ``.h5`` or ``.bp`` group member, or a ``.pmd`` pointer
to a member. BP series directories are recognized directly, including through a
pointer. ``--output`` names the merged series, ``--iterations`` restricts it to
listed steps, and ``--overwrite`` replaces an existing merged file.

The ``.pmd`` pointer is written beside the merged series with the same basename:
``--output /results/custom.h5`` creates ``/results/custom.pmd`` containing
``custom.h5``. Existing source-directory pointers are left untouched when the
destination has a different pointer path. Without ``--output``, the usual
``data/fields.pmd`` still points to ``fields.h5``. Use ``--no-pmd`` to leave all
pointers untouched.

Validation
----------

Run local input/chunk and initial histogram tests on two logical CPU devices:

.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2 JAX_ENABLE_X64=true \
       python -m unittest tests.distributed_tests.distributed_io_test \
       tests.distributed_tests.sharded_histogram_test

Per-process naming, ownership metadata and the merge round trip run on one
process and need no launcher:

.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=8 JAX_ENABLE_X64=true \
       python -m unittest tests.distributed_tests.merge_openpmd_test \
       tests.distributed_tests.sharded_openpmd_test

A real two-process check needs two shells sharing a coordinator address and a
directory visible to both, with ``PYPIC_PROCESS_ID`` 0 and 1. Confirm the run
wrote ``fields_p000.h5`` and ``fields_p001.h5`` and no bare ``fields.h5``, then
merge and read the result with an unmodified analysis script.
