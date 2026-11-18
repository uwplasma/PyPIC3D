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
   particle_batch_size = 1250

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
  domain is centered on zero unless explicit bounds are given.
- ``x_min``/``x_max`` (and the ``y`` and ``z`` pairs) place the domain
  explicitly instead of centering it. The GR demos use this form.
- If ``dt`` is omitted, it is computed from ``cfl`` and the active grid
  spacing.
- If ``Nt`` is omitted, it is computed as ``int(t_wind / dt)``.
- If ``Nt`` is provided, it controls the number of iterations directly.
- Tile widths default to the full physical grid, giving one tile. Multi-tile
  configuration and device requirements are described in :doc:`tiling`.

Numerical Choices
-----------------

- ``solver`` is ``electrodynamic_yee``, ``electrostatic``, ``static_metric``,
  ``dynamic_metric``, or ``dark_matter_yee``. See :doc:`solvers` for the update
  sequences, :doc:`dark_photon` for the dark sector, and
  :doc:`dynamical_relativity` for evolving geometry.
- ``metric`` selects the spacetime chart for the GR solvers: ``flat_cartesian``
  (default), ``flat_cylindrical``, ``flat_spherical``,
  ``kerr_schild_cartesian``, ``kerr_schild_spherical``, or ``numerical``. The
  Kerr-Schild charts also read ``metric_mass`` (default ``1.0``) and
  ``metric_spin`` (default ``0.0``). See :doc:`dynamical_relativity`.
- ``particle_pusher`` is ``boris`` or ``higuera_cary``. The ``relativistic``
  switch selects relativistic or non-relativistic Boris; Higuera-Cary uses its
  relativistic update.
- ``static_metric`` uses ``hybrid_boris_geodesic`` and the shared Hermite
  particle-metric reconstruction described in :doc:`solvers`. Its particle
  sampler requires ``guard_cells >= 3`` (the default for this configuration).
- ``particle_batch_size`` is the positive, static number of active particles
  processed at once inside each tile. An explicit value is reduced to the tile
  slot capacity when necessary. If omitted, initialization uses every active
  particle for a one-tile run. For a multi-tile CPU run it targets 128
  particles per scheduler-visible logical CPU thread assigned to each local
  mesh device, capped by the largest active tile population. Other accelerator
  backends use a conservative target of 1024 particles. The resolved value is
  printed before JIT compilation.
- ``shape_factor`` is ``1`` or ``2`` for particle shapes and electromagnetic
  field gathering. It does not select the particle-metric reconstruction.
- ``current_calculation`` is ``j_from_rhov`` or ``esirkepov`` for the Yee
  solvers, or ``GR_direct_deposition`` or ``esirkepov`` for
  ``static_metric``. ``esirkepov`` is charge conserving and preserves
  Gauss's law to round-off; see :doc:`chargeconservation`.
- ``filter_j`` is ``none``, ``digital``, or ``bilinear`` for direct current.
  The selected filter is applied to deposited current and to an evolved electric
  copy used for particle interpolation, including the initial backward
  half-kick. External fields and the magnetic gather are unfiltered. On the Yee
  solvers Esirkepov requires ``filter_j = "none"``. The ``static_metric`` and
  ``dynamic_metric`` solvers instead support a matched filter that is applied to
  the deposited current, the charge density, the gathered physical D/B and the
  particle metric with one common periodic stencil, so Esirkepov continuity is
  preserved; it requires a Cartesian metric and periodic field and particle
  boundaries. Because of those constraints both GR solvers default ``filter_j``
  to ``none`` rather than inheriting the Yee default, so matched filtering is
  opt-in.
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

``dark_matter_yee`` additionally appends ``dark_field_energy.txt`` and prints a
``Dark Field Energy`` line. ``dynamic_metric`` instead writes the constraint
norms ``hamiltonian_l2.txt`` and ``momentum_l2.txt``; flat-space energy
conservation reports do not apply to a dynamical spacetime.

Set ``plot_openpmd_fields = true`` to write ``data/fields.h5`` and
``plot_openpmd_particles = true`` to write ``data/particles.h5``. Writers copy
only addressable device shards and store their chunks at global offsets,
without assembling full field or particle arrays. Every run uses bounded
asynchronous queues, including when it spans several devices or processes.
``openpmd_field_queue_size`` and ``openpmd_particle_queue_size`` limit pending
host snapshots, not the number of output files. A full queue waits for disk
writes; larger queues retain more host memory.


Field output contains ``E``, ``B``, and ``J``. ``plotchargedensity = true``
deposits the scalar charge-density mesh ``rho`` from the current particle
positions, and ``plotvelocities = true`` adds the particle-weighted vector mesh
``fluid_velocity``. ``dump_fields`` writes
``data/initial_fields/initial_fields.h5``, and ``dump_particles`` writes
``data/initial_particles/<species>_initial_particles.h5`` during initialization,
including runs with ``Nt = 0``. Initial particle dumps retain the configured
positions and velocities before leapfrog seeding, with no diagnostic half-step
position shift. Initial histogram plots in single-process runs accumulate
50-bin counts from local shards and retain the existing image names and limits.
Final run parameters, species metadata, and timing statistics are written to
``data/output.toml``. These diagnostics are not complete restart checkpoints.

Dark Photon Initial Data
------------------------

``dark_matter_yee`` adds ``[dark_fieldN]`` blocks for the dark sector and the
``sin_chi`` / ``dark_mu`` static parameters. It also overrides the ordinary
Courant condition with the Proca stability limit. See :doc:`dark_photon`.

Dynamical Geometry
------------------

``dynamic_metric`` adds a ``[metric]`` table carrying the initial BSSN
geometry, the gauge controls and the six gravitational boundary codes, and it
forces ``C = eps = mu = 1`` with double precision. See
:doc:`dynamical_relativity`.
