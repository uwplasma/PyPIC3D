Architecture
============

This page summarizes how PyPIC3D is organized and how data flows through a run.

Execution Flow
--------------

1. ``PyPIC3D.__main__.main`` parses ``--config`` and enables JAX settings.
2. ``initialization.initialize_simulation`` builds defaults, loads TOML,
   computes derived world parameters, builds grids, initializes particles and
   fields, and selects loop functions.
3. ``run_PyPIC3D`` executes the timestep loop, writes diagnostics/outputs, and
   dumps run metadata.

Core Module Map
---------------

- ``PyPIC3D/__main__.py``: CLI and top-level run loop.
- ``PyPIC3D/initialization.py``: parameter defaults, config merge, world/grid
  setup, and mode selection.
- ``PyPIC3D/evolve.py``: JIT-compiled per-step loops.
- ``PyPIC3D/particle.py``: particle species model, initialization loaders,
  particle boundary handling.
- ``PyPIC3D/J.py``: current deposition kernels.
- ``PyPIC3D/rho.py``: charge deposition.
- ``PyPIC3D/solvers/``: field update operators and electrostatic Poisson
  helpers.
- ``PyPIC3D/diagnostics/``: phase-space plots and openPMD output.
- ``PyPIC3D/utils.py``: config handling, filters, energy calculations,
  serialization helpers.

State Model
-----------

Main runtime objects:

- ``particles``: list of particle species objects.
- ``fields``:

  - electrostatic: ``(E, B, J, rho, phi, external_fields)``
  - electrodynamic: ``(E, B, J, rho, phi, external_fields, pml_state)``

- ``world``: spatial/temporal metadata, grid spacing, grid arrays, encoded field
  boundary conditions.
- ``constants``: physical constants and filter coefficients.

Tiled State Contract
--------------------

The ``electrodynamic_yee`` path uses one shared tile shape for fields and particles.
The existing tile-size configuration is interpreted as this common
``tile_shape = (tile_nx, tile_ny, tile_nz)``; there are not separate field-tile
and particle-tile dimensions in the current contract.  Each tile width must
divide the corresponding physical grid size exactly, so the leading tile axes
are ``(ntx, nty, ntz)`` with ``ntx * tile_nx = Nx`` and similarly for ``y`` and
``z``.

Tiled fields store each vector component as compact per-tile arrays with
leading tile axes followed by a one-cell halo around the tile-local physical
interior:
``(ntx, nty, ntz, tile_nx + 2, tile_ny + 2, tile_nz + 2)``.  The halo cells
carry neighbor-tile values or exterior field boundary conditions so tiled Yee
curls can be evaluated on the physical interior without assembling a global
field.

Tiled particles use the same leading tile axes, followed by species and fixed
slot axes.  Positions and velocities have shape
``(ntx, nty, ntz, species, max_particles_per_tile, 3)``; the active mask has
shape ``(ntx, nty, ntz, species, max_particles_per_tile)``.  Species-level
metadata such as charge, mass, weight, and position/velocity update masks is
stored once in ``SpeciesConfig`` and broadcast over tile slots inside the tiled
kernels.  The slot capacity is set when tiled particles are initialized.  Empty
slots remain inactive so the array shape stays static during JAX updates.

Retiling preserves this fixed-capacity layout.  If later particle motion would
place more active particles in a tile/species block than its slot capacity can
hold, the tiled refresh reports overflow and the Python driver treats that as a
hard error.  This avoids silently dropping active particles.

Data and Output Flow
--------------------

- Diagnostics and metadata are written under ``<output_dir>/data``.
- Text outputs include energy and momentum traces.
- Optional outputs include matplotlib phase-space arrays and openPMD files.
- ``output.toml`` captures simulation stats, resolved runtime parameters,
  particle summaries, and package versions.
