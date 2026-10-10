Domain Decomposition
====================

PyPIC3D uses one shared Cartesian decomposition for fields and particles. Each
logical tile owns a physical interior, surrounding guard cells, fixed-capacity
particle slots, and one position in the JAX device mesh.

.. figure:: images/new_tiled_field.png
   :alt: Six simulation tiles with active interiors surrounded by ghost-cell regions
   :align: center
   :width: 80%

   A tiled domain decomposition. Each active tile interior is surrounded by
   guard cells used for field stencils and cross-tile deposition.

Tile Geometry
-------------

The input values ``particle_tile_nx``, ``particle_tile_ny``, and
``particle_tile_nz`` are physical cells per tile. Particle tile shape
is:

.. code-block:: text

   tile_shape = (tile_nx, tile_ny, tile_nz)
   ntx = Nx / tile_nx
   nty = Ny / tile_ny
   ntz = Nz / tile_nz

Each tile width must divide its physical grid dimension exactly. If tile widths
are omitted, initialization uses ``(Nx, Ny, Nz)``, so the complete domain is
one tile. Electrostatic and electrodynamic runs both preserve an explicitly
configured multi-tile layout.

Field and Grid Storage
----------------------

For guard depth ``g``, every scalar tile has shape:

.. code-block:: text

   (ntx, nty, ntz,
    tile_nx + 2*g, tile_ny + 2*g, tile_nz + 2*g)

Vector fields are tuples of three arrays with this shape. The tiled center and
vertex coordinate arrays use the same leading tile structure and local ghost
cells. The default is two guard cells, while the nearest-neighbor field
stencils also support an explicit one-cell guard region:

.. code-block:: text

   g = guard_cells >= 1

The physical interior of each tile is ``g:-g`` along its local spatial axes.

Communication and Deposition
----------------------------

PyPIC3D uses two complementary operations:

- **Ghost refresh** copies neighboring owner-interior values into tile ghost
  cells before a stencil or interpolation reads them.
- **Ghost folding** adds current, charge, or fluid-moment contributions
  deposited in a guard region back into the neighboring tiles. Ghost cells
  are refreshed again after the fold.

Communication is performed with ``jax.shard_map`` and ``jax.lax.ppermute`` over
the named ``tile_x``, ``tile_y``, and ``tile_z`` mesh axes. Reduced one-cell
dimensions remain in the tile layout and are handled without removing axes or
changing array rank.

Particle Boundary Folds
-----------------------

At a reflecting particle wall, every solver folds particle deposits by the
nodal method of images about the wall nodes, in ``boundary_conditions/sources.py``
and ``halo_exchange._staggered_conducting_axis``:

- Charge and fluid moments sit on collocated C nodes. The upper wall node at
  ``g+n`` is an owned physical node. A deposit in the exterior guard region is
  added to its mirror owner, and a wall node receives its coincident image,
  so its point density doubles. Integrals of these nodal densities use
  trapezoid end weights on the wall nodes (see
  ``diagnostics.static_metric.node_weights``).
- Currents sit at the Yee E-component locations. Their images are odd in the
  component normal to the wall and even in the two tangential components.

Halos are then refreshed with the same images. No metric volume factors enter
the fold.

Device Topology
---------------

The current runtime assigns exactly one logical tile to each JAX device. A
layout with ``ntx * nty * ntz`` tiles therefore needs at least that many
devices, and the leading tile structure must match the device mesh.

The CLI enables 64-bit calculations but leaves backend selection to JAX. Use
``JAX_PLATFORMS`` when a run must fail instead of falling back to another
backend, and restrict the visible accelerators before starting Python. A
one-GPU CUDA run uses:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
     PyPIC3D --config path/to/config.toml

A two-GPU run uses:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda \
     PyPIC3D --config path/to/config.toml

Confirm the two devices are available to JAX before launching a long run:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda \
     python -c "import jax; print(jax.default_backend(), jax.devices())"

The ``GPUs`` configuration setting is accepted but not used for device
selection and is not stored in runtime parameters. The JAX environment
variables above determine the backend and visible devices.

The tile topology must also contain two tiles. For a ``300 x 1 x 300`` x-z
domain, splitting the periodic x direction keeps the full current-sheet
profile on each device:

.. code-block:: toml

   [simulation_parameters]
   Nx = 300
   Ny = 1
   Nz = 300
   particle_tile_nx = 150
   particle_tile_ny = 1
   particle_tile_nz = 300
   particle_tile_capacity_factor = 1.25

This creates a ``(2, 1, 1)`` mesh: the first visible GPU owns the first x
tile and the second visible GPU owns the second. The capacity factor provides
free particle slots for cross-tile motion; increase it if the runtime reports
a tile-capacity overflow.

For CPU-only development, expose the required number of logical CPU devices
before Python imports JAX. A six-tile example is:

.. code-block:: bash

   XLA_FLAGS=--xla_force_host_platform_device_count=6 \
     PyPIC3D --config path/to/config.toml

The one-tile default works with the ordinary single-device CPU setup.

Particle Storage and Retiling
-----------------------------

Particle arrays have shapes:

.. code-block:: text

   x, u   : (ntx, nty, ntz, species, max_particles_per_tile, 3)
   active : (ntx, nty, ntz, species, max_particles_per_tile)

``TiledParticles`` stores only these dynamic arrays. ``SpeciesConfig`` stores
charge, mass, weight, and position/velocity update masks once per species.

During initialization, PyPIC3D finds the largest active tile/species
population and multiplies it by ``particle_tile_capacity_factor`` to choose a
single static slot capacity. A value greater than one leaves room for later
particle motion. Inactive slots remain allocated so array shapes stay fixed
under JIT compilation.

The particle pusher constructs a temporary fixed-shape active-index list in
each tile and processes it in ``particle_batch_size`` chunks. The six Yee field
gathers and the selected velocity pusher therefore skip inactive slots, while
``x``, ``u``, and ``active`` retain their fixed-capacity layout. Because tiles
are mapped together, a compiled multi-tile push can execute up to the largest
active batch count among the mapped tiles; no global particle compaction or
cross-device gather is performed.

When ``particle_batch_size`` is omitted, the batch shape is resolved after
particle packing and remains static for the run. A one-tile run uses its full
active population in one batch, regardless of spare inactive capacity. A
multi-tile CPU run estimates the available compute from process CPU affinity
and the number of local mesh devices, targeting 128 particles per available
logical thread and never exceeding the largest initial active tile population.
Accelerators use a conservative target of 1024 particles. This is a
deterministic throughput heuristic rather than an exact hardware-performance
model; an explicit positive ``particle_batch_size`` remains the override for
machine-specific tuning.

After a position update, particles are sent to adjacent owner tiles and packed
into free slots. A destination without enough slots, or a particle crossing
more than one tile in a single refresh, sets the overflow flag. The Python
driver then raises a hard error rather than dropping particles.

Runtime and Output Boundaries
-----------------------------

Particle pushing, direct and Esirkepov current deposition, Yee curls, charge
deposition, filtering, and fluid-velocity deposition are all sharded across 
tiles.

Runtime openPMD writers transfer sharded tile snapshots to host memory and
write physical tile interiors into their global mesh offsets. The live solver
arrays remain tiled. Global assembly helpers are reserved for synchronous
output adapters, tests, and validation references. The production
electrostatic Poisson path remains in tiled storage and communicates only
through scalar halo refreshes.
