Usage
=====

PyPIC3D uses TOML configuration files to define the simulation 
parameters.

Run Command
-----------

.. code-block:: bash

   PyPIC3D --config path/to/config.toml

The output directory is ``simulation_parameters.output_dir`` (the current
working directory by default). PyPIC3D creates a ``data`` subdirectory there.

Configuration Sections
----------------------

``[simulation_parameters]`` is the input section for solver,
geometry, runtime, and physical-constant values.

``[plotting]`` contains write parameters used during simulations 
to write output data. Particle and external-field entries use numbered 
sections such as ``[particle1]`` and ``[field1]``.

All sections contain built-in defaults. A particle section is needed for a PIC
run, but field-only electrodynamic configurations are also supported.

Minimal PIC Example
-------------------

This example uses one tile covering the complete grid and therefore needs only
one JAX device:

.. code-block:: toml

   [simulation_parameters]
   name = "minimal"
   output_dir = "./outputs"
   solver = "electrodynamic_yee"
   particle_pusher = "boris"
   relativistic = true

   Nx = 64
   Ny = 1
   Nz = 1
   x_wind = 1.0
   y_wind = 1.0
   z_wind = 1.0
   t_wind = 1.0e-8
   cfl = 0.9

   particle_tile_nx = 64
   particle_tile_ny = 1
   particle_tile_nz = 1
   guard_cells = 2
   particle_tile_capacity_factor = 1.25

   shape_factor = 1
   current_calculation = "j_from_rhov"
   filter_j = "bilinear"
   alpha = 1.0

   x_bc = "periodic"
   y_bc = "periodic"
   z_bc = "periodic"
   particle_x_bc = "periodic"
   particle_y_bc = "periodic"
   particle_z_bc = "periodic"

   eps = 8.85418782e-12
   mu = 1.25663706e-6
   C = 2.99792458e8
   kb = 1.380649e-23

   [plotting]
   plotting_interval = 10
   plot_openpmd_particles = false
   plot_openpmd_fields = false
   plotvelocities = false
   plotchargedensity = false
   openpmd_field_queue_size = 2
   openpmd_particle_queue_size = 2
   dump_particles = false
   dump_fields = false

   [particle1]
   name = "electrons"
   N_particles = 5000
   charge = -1.602e-19
   mass = 9.1093837e-31
   temperature = 1.0

Time and Grid Parameters
------------------------

- ``Nx``, ``Ny``, and ``Nz`` are the physical cell counts.
- ``x_wind``, ``y_wind``, and ``z_wind`` are the physical domain widths. The
  domain is centered on zero.
- If ``dt`` is omitted, it is computed from ``cfl`` and the active grid
  spacing.
- If ``Nt`` is omitted, it is computed as ``int(t_wind / dt)``.
- If ``Nt`` is provided, it controls the number of iterations directly.
- Tile widths default to the full physical grid, giving one tile. Multi-tile
  configuration and device requirements are described in :doc:`tiling`.

Numerical Choices
-----------------

- ``solver`` is ``electrodynamic_yee`` or ``electrostatic``.
- ``particle_pusher`` is ``boris`` or ``higuera_cary``. The ``relativistic``
  switch selects relativistic or non-relativistic Boris; Higuera-Cary uses its
  relativistic update.
- ``shape_factor`` is ``1`` or ``2``.
- ``current_calculation`` is ``j_from_rhov`` or ``esirkepov``.
- ``filter_j`` is ``none``, ``digital``, or ``bilinear`` for direct current.
  The selected filter is applied to deposited current and to an evolved electric
  copy used for particle interpolation. Esirkepov requires
  ``filter_j = "none"``.
- ``alpha`` is the digital filter coefficient and belongs in a
  recognized simulation or dynamic parameter section.
- ``electrostatic_schwarz_tol`` is the maximum absolute Poisson residual in
  the interface-adjacent owned cells (default ``1e-6``).
- ``electrostatic_schwarz_max_iterations`` caps nearest-neighbor Schwarz
  iterations per electrostatic timestep (default ``500``).
- ``electrostatic_local_cg_tol`` is the tile-local CG residual-norm tolerance
  (default ``1e-6``).
- ``electrostatic_local_cg_max_iterations`` caps CG iterations within each
  local tile solve (default ``500``).

Boundary Conditions and PML
---------------------------

Field boundaries use ``x_bc``, ``y_bc``, and ``z_bc`` with values ``periodic``
or ``conducting``. Particle boundaries are global settings shared by all
species:

- ``particle_x_bc``
- ``particle_y_bc``
- ``particle_z_bc``

Their supported values are ``periodic``, ``reflecting``, and ``absorbing``.

Coordinate-stretched PML is available only with ``electrodynamic_yee``. Add one
entry for each absorbing wall:

.. code-block:: toml

   [[pml]]
   wall = "-x"
   thickness = 8
   order = 3.0
   target_reflection = 1.0e-8

   [[pml]]
   wall = "+x"
   thickness = 8
   order = 3.0
   sigma_max = 60.0

``wall`` is one of ``-x``, ``+x``, ``-y``, ``+y``, ``-z``, or ``+z``.
``sigma_max`` can be supplied directly; otherwise it is derived from
``target_reflection``. A periodic field boundary on a PML-active axis is
changed to a non-wrapping conducting halo contract during initialization.

External Fields
---------------

Use ``[fieldX]`` entries to load ``.npy`` arrays into field components:

.. code-block:: toml

   [field1]
   name = "prescribed_Bz"
   type = 5
   path = "inputs/Bz.npy"
   evolve = false

The component mapping is:

- ``0``, ``1``, ``2``: ``Ex``, ``Ey``, ``Ez``
- ``3``, ``4``, ``5``: ``Bx``, ``By``, ``Bz``
- ``6``, ``7``, ``8``: ``Jx``, ``Jy``, ``Jz``

Arrays must have physical-interior shape ``(Nx, Ny, Nz)``. ``evolve = true``
is the default and adds the array to the self-consistent field state.
``evolve = false`` is available for electric and magnetic components only; it
keeps the prescribed field outside Maxwell evolution while including it in
particle pushes and energy diagnostics.

Outputs
-------

At ``plotting_interval`` cadence, the driver appends:

- ``total_energy.txt``
- ``energy_error.txt``
- ``electric_field_energy.txt``
- ``magnetic_field_energy.txt``
- ``kinetic_energy.txt``
- ``total_momentum.txt``

Set ``plot_openpmd_fields = true`` or
``plot_openpmd_particles = true`` for asynchronous runtime output. The 
number of openPMD files is limited by ``openpmd_field_queue_size`` and
``openpmd_particle_queue_size``. The queue sizes determine how many 
snapshots are retained in memory before being written to disk. If the queue
sizes are too small, the simulation may stall while waiting for disk writes to
complete. If the queue sizes are too large, the simulation may run out of memory.


Electrodynamic and static-metric field output contains ``E``, ``B``, and ``J``.
For these solvers, ``plotchargedensity = true`` deposits the scalar
charge-density mesh ``rho`` from the current particle positions. Electrostatic
field output instead contains ``rho``, ``phi``, and ``E`` by default; its zero
``B`` and ``J`` fields are omitted. ``plotvelocities = true`` adds the
particle-weighted vector mesh ``fluid_velocity`` for every solver.
``dump_fields`` and ``dump_particles`` write initial-state openPMD files during
initialization. Final run parameters, species metadata, and timing statistics
are written to ``data/output.toml``.
