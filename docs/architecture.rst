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
- ``PyPIC3D/diagnostics/``: phase-space plots, VTK output, openPMD output.
- ``PyPIC3D/utils.py``: config handling, filters, energy calculations,
  serialization helpers.

State Model
-----------

Main runtime objects:

- ``particles``: list of particle species objects.
- ``fields``:

  - electrodynamic/electrostatic: ``(E, B, J, rho, phi)``
  - vector potential path: ``(E, B, J, rho, phi, A2, A1, A0)``

- ``world``: spatial/temporal metadata, grid spacing, grid arrays, encoded field
  boundary conditions.
- ``constants``: physical constants and filter coefficients.

Data and Output Flow
--------------------

- Diagnostics and metadata are written under ``<output_dir>/data``.
- Text outputs include energy and momentum traces.
- Optional outputs include VTK files and openPMD files.
- ``output.toml`` captures simulation stats, resolved runtime parameters,
  particle summaries, and package versions.
