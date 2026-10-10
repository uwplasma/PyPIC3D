Architecture
============

PyPIC3D separates compile-time numerical choices from dynamic scalar and array
state. The production timestep operates on tile-major fields and particles;
global arrays are constructed only at explicit diagnostic and output
boundaries.

PyPIC3D contains a single run loop that advances the simulation timestep,
the intermittently copies tile-local snapshots to asynchronous openPMD 
writers and computes energy and momentum diagnostics. The timestep is JIT-
compiled with static parameters and the JAX device mesh, while dynamic
parameters and particle state are passed as JAX array objects.

``dynamic_parameters`` contains the timestep, grid spacing and size, 
physical constants, and global/tiled coordinate arrays. These values 
are JAX array leaves and can pass through compiled kernels.

``static_parameters`` contains choices that control compiled branches 
or array layout, including the solver, pusher, deposition method, 
shape factor, tile shape, guard depth, boundary codes, and JAX device 
mesh.

``TiledParticles`` contains dynamic tile/species/slot arrays for particle
positions, momenta, and active flags. ``SpeciesConfig`` contains metadata
stored once per species, including charge, mass, weight, and flags for
whether to update positions and momenta.

The ordinary Yee and electrostatic solvers use this field-state tuple:

.. code-block:: text

   (E, B, J, rho, phi, external_fields, pml_state, overflow)

``E``, ``B``, and ``J`` are three-component tiled fields. ``rho`` and ``phi``
are tiled scalar fields. ``external_fields`` contains prescribed electric and
magnetic fields used by the particle push and energy diagnostics but excluded
from Maxwell evolution. ``pml_state`` is ``None`` unless PML is active, and
``overflow`` reports a failed fixed-capacity particle retile.

The dark photon solver preserves that prefix and inserts its dark state before
the overflow flag. With PML active, its ``pml_state`` is
``(maxwell_pml, dark_pml)``; otherwise it is ``None``. The two sectors share wall
settings but maintain independent histories:

.. code-block:: text

   (E, B, J, rho, phi, external_fields, pml_state, dark_fields, overflow)

``dark_fields`` is ``(E_prime, A_prime, phi_prime)`` with ``E_prime`` and
``phi_prime`` at integer time and ``A_prime`` half a step behind. Output and
particle gathering reconstruct ``A_prime`` and its curl at integer time.


See :doc:`tiling` for the array shapes, guard ownership, and sharding structure.

Boundary code separates mesh transport from physical policy.
``boundary_conditions.halo_exchange`` implements distributed refresh/folding,
and ``ownership`` defines staggered endpoints and reflected indices.
``ghost_cells`` retains the public field/particle adapters; ``sources`` supplies
staggered source parity. ``staggered`` stages horizon extrapolation and field
refresh, while ``pec`` handles metric-aware wall projections. Transport and
ownership do not depend on the solver or PEC implementation.

Execution Flow
--------------

1. ``PyPIC3D.__main__.main`` enables 64-bit JAX, reports the backend and
   devices selected by the environment, and parses ``--config``.
2. ``initialization.initialize_simulation`` merges TOML values with defaults,
   builds global and tiled grids, creates the JAX device mesh, initializes
   particles and fields, and selects the timestep function.
3. ``run_PyPIC3D`` closes over the static parameters, JIT-compiles the selected
   timestep, advances the simulation, checks particle-retile overflow, and
   schedules diagnostics.
4. Asynchronous openPMD writers consume tile-local snapshots without changing
   the live solver state.

Core Module Map
---------------

- ``PyPIC3D/__main__.py``: CLI, run loop, writer scheduling, and diagnostics.
- ``PyPIC3D/initialization.py``: configuration defaults, grid/device setup,
  particle and field initialization, and solver selection.
- ``PyPIC3D/parameters.py``: ``StaticParameters``, ``DynamicParameters``, and
  grid parameter tuples.
- ``PyPIC3D/particles/``: fixed-capacity particle state, initialization, and
  cross-tile communication.
- ``PyPIC3D/pusher/``: field interpolation and particle pushers.
- ``PyPIC3D/deposition/``: direct current, Esirkepov current, charge density,
  and particle shape functions.
- ``PyPIC3D/solvers/``: algorithm-specific time loops and numerical kernels
  for electrostatic, Yee, and prescribed-metric GR evolution.
- ``PyPIC3D/boundary_conditions/``: field/current ghost cell communications,
  conducting PEC walls, grid stencils, and PML.
- ``PyPIC3D/diagnostics/``: field maps, fluid velocity, energy/momentum, and
  openPMD output.
- ``PyPIC3D/utilities/``: grid construction and numerical filters.
