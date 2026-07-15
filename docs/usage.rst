Usage
=====

This page documents the current PyPIC3D runtime workflow and TOML configuration
schema used by the CLI entrypoint in ``PyPIC3D.__main__``.

Run Command
-----------

.. code-block:: bash

    PyPIC3D --config path/to/config.toml

The command loads the config, initializes the simulation state, and writes data
under ``simulation_parameters.output_dir`` (default: current working directory)
in the ``data`` subdirectory.

Required Top-Level Sections
---------------------------

PyPIC3D expects these config sections:

- ``[simulation_parameters]``: required
- ``[plotting]``: required
- At least one ``[particleX]`` section: required
- ``[constants]``: optional
- ``[fieldX]`` sections: optional external field loading

Minimal Working Example
-----------------------

.. code-block:: toml

    [simulation_parameters]
    name = "minimal"
    solver = "electrodynamic_yee"
    relativistic = true
    current_calculation = "j_from_rhov"   # j_from_rhov or esirkepov
    filter_j = "bilinear"                 # bilinear, digital, none
    Nx = 64
    Ny = 1
    Nz = 1
    x_wind = 1.0
    y_wind = 1.0
    z_wind = 1.0
    t_wind = 1e-8
    cfl = 0.9
    shape_factor = 1                       # 1 or 2
    x_bc = "periodic"                     # periodic or conducting
    y_bc = "periodic"
    z_bc = "periodic"
    particle_x_bc = "periodic"            # periodic, reflecting, or absorbing
    particle_y_bc = "periodic"
    particle_z_bc = "periodic"
    output_dir = "./outputs"

    [plotting]
    plotting_interval = 10
    plot_phasespace = false
    plot_openpmd_particles = false
    plot_openpmd_fields = false
    openpmd_field_queue_size = 2
    openpmd_particle_queue_size = 2
    dump_particles = false
    dump_fields = false

    [constants]
    eps = 8.85418782e-12
    mu = 1.25663706e-6
    C = 2.99792458e8
    kb = 1.380649e-23
    alpha = 1.0

    [particle1]
    name = "electrons"
    N_particles = 5000
    charge = -1.602e-19
    mass = 9.1093837e-31
    temperature = 1.0

Key Notes
---------

- ``[plotting]`` is required even when most flags are ``false``.
- ``dt`` and ``Nt`` are optional:

  - If ``dt`` is omitted, it is computed from CFL and grid spacing.
  - If ``Nt`` is omitted, it is derived from ``t_wind / dt``.
  - If both are provided, ``t_wind`` is updated to ``dt * Nt``.

- Field boundary conditions use ``simulation_parameters.x_bc/y_bc/z_bc`` with
  values ``periodic`` or ``conducting``.
- Particle boundary conditions are global simulation parameters:
  ``particle_x_bc/particle_y_bc/particle_z_bc`` with values ``periodic``,
  ``reflecting``, or ``absorbing``.
- The smoothing coefficient is ``constants.alpha`` (not
  ``simulation_parameters.alpha``).

External Field Injection
------------------------

Use ``[fieldX]`` entries to add arrays to initial field components:

.. code-block:: toml

    [field1]
    name = "initial_bias"
    type = 0
    path = "inputs/initial_Ex.npy"

``type`` mapping:

- ``0``: Ex
- ``1``: Ey
- ``2``: Ez
- ``3``: Bx
- ``4``: By
- ``5``: Bz
- ``6``: Jx
- ``7``: Jy
- ``8``: Jz

Arrays must match the simulation grid shape ``(Nx, Ny, Nz)``.

Outputs
-------

At ``plotting_interval`` cadence, PyPIC3D writes diagnostics like:

- ``total_energy.txt``
- ``energy_error.txt``
- ``electric_field_energy.txt``
- ``magnetic_field_energy.txt``
- ``kinetic_energy.txt``
- ``total_momentum.txt``

Depending on flags, it also writes matplotlib phase-space arrays and openPMD
files plus ``data/output.toml`` metadata. Runtime openPMD field and particle
output use bounded queues; ``openpmd_field_queue_size`` and
``openpmd_particle_queue_size`` cap pending batches so output cannot grow memory
without bound.
