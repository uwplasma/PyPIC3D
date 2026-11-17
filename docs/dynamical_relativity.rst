Dynamical Relativity
====================

``solver = "dynamic_metric"`` evolves the spacetime geometry with BSSN numerical
relativity while PyPIC3D keeps particles, Maxwell evolution, and the
stress-energy projection. The plasma curves the spacetime, and the evolving
spacetime moves the plasma.

This page covers the configuration surface and the operational contract. For the
full coupling specification --- unit conventions, time levels, source filtering,
and the work-energy diagnostic --- see :doc:`einstein_maxwell`. For the
prescribed-metric solver, see the GR Maxwell section of :doc:`solvers`.

Requirements
------------

The BSSN equations are **not** part of this repository. They live in the
optional ``JAX_BSSN`` dependency:

.. code-block:: bash

   pip install -e '.[relativity]'

Importing ordinary PyPIC3D solvers does not import JAX BSSN, and the dynamical
test suite skips entirely when it is absent.

The coupled solver rejects configurations that violate any of these; each is a
hard error at initialization, not a warning:

- **Geometrized Gaussian units**: ``C = eps = mu = 1``. The CLI sets these for
  this solver.
- **Double precision**: ``jax_enable_x64 = True``. The CLI enables it.
- **Uniform spacing**: all resolved Cartesian axes must share one grid spacing,
  and the BSSN and PIC ``dx``/``dt`` must agree.
- **Fixed timestep**: ``dt`` must not change during a run.
- Prescribed external fields, PML, and non-Cartesian charts are not supported.
- Matched GR filtering additionally requires all six BSSN boundaries periodic
  and a Cartesian chart.

The particle path supports periodic Cartesian tiles on a shared device mesh,
with one tile per device. BSSN and EM use identical tile ownership; BSSN retains
wider local guards for its stencils and for the C-to-Yee geometry transfer.

Formulation
-----------

The evolved variables are the standard BSSN set, collocated on ``(C,C,C)``
nodes:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Symbol
     - Meaning
   * - ``conformal_metric``
     - :math:`\tilde\gamma_{ij}`
     - symmetric 3x3, unit determinant
   * - ``conformal_factor``
     - :math:`W`
     - :math:`\gamma_{ij}=\tilde\gamma_{ij}/W^{2}`
   * - ``traceless_K``
     - :math:`\tilde A_{ij}`
     - symmetric 3x3, traceless
   * - ``trace_K``
     - :math:`K`
     - scalar
   * - ``conformal_connection``
     - :math:`\tilde\Gamma^{i}`
     - 3-vector
   * - ``lapse``
     - :math:`\alpha`
     - scalar
   * - ``shift``
     - :math:`\beta^{i}`
     - 3-vector

The conformal factor convention is **W**, so the physical spatial metric is
:math:`\gamma_{ij}=\tilde\gamma_{ij}/W^{2}`. The extrinsic-curvature convention
is :math:`K_{ij}=-\tfrac{1}{2}\mathcal L_n\gamma_{ij}`.

Matter enters only the :math:`\tilde A_{ij}`, :math:`K` and
:math:`\tilde\Gamma^{i}` equations; the conformal metric, conformal factor,
lapse and shift have no matter source. Algebraic constraints ---
:math:`\det\tilde\gamma=1` and :math:`\mathrm{tr}\,\tilde A=0` --- are projected
at every stage.

Gauge conditions
~~~~~~~~~~~~~~~~

The lapse uses either harmonic or 1+log slicing, selected by ``gauge``:

.. math::

   \text{harmonic:}\quad \partial_t\alpha &= -\alpha^{2}K
     + \beta^{i}\partial_i\alpha, \\
   \text{1+log:}\quad \partial_t\alpha &= -2\alpha K
     + \beta^{i}\partial_i\alpha.

The shift uses a single-variable Gamma driver:

.. math::

   \partial_t\beta^{i} = g\,\tilde\Gamma^{i}
     + \beta^{j}\partial_j\beta^{i} - \eta\,\beta^{i}.

Both add sixth-order Kreiss--Oliger dissipation with coefficient
:math:`\nu/64`, applied only on resolved axes. Setting ``zero_shift = true``
**freezes the supplied shift**; it does not replace it with zero, so a nonzero
initial shift stays nonzero and stays constant.

Time integration
~~~~~~~~~~~~~~~~

One macro step is PIC(``dt``/2), BSSN(``dt``), PIC(``dt``/2). The BSSN interval
uses RK4 with gauge callbacks and algebraic projection applied at every
intermediate stage, but the coupled Strang split is **second order** overall.
Each gravity stage resamples the particle geometry and recomputes the particle
stress from fixed midpoint positions and covariant momenta.

Selecting the chart
-------------------

The ``metric`` key under ``[simulation_parameters]`` chooses the spacetime
chart. It is distinct from the ``[metric]`` table described below.

.. list-table::
   :header-rows: 1
   :widths: 32 20 48

   * - ``metric``
     - Solver
     - Notes
   * - ``flat_cartesian``
     - either
     - default
   * - ``flat_cylindrical``
     - ``static_metric``
     -
   * - ``flat_spherical``
     - ``static_metric``
     -
   * - ``kerr_schild_cartesian``
     - ``static_metric``
     - uses ``metric_mass``, ``metric_spin``
   * - ``kerr_schild_spherical``
     - ``static_metric``
     - horizon-penetrating; uses ``metric_mass``, ``metric_spin``
   * - ``numerical``
     - ``dynamic_metric``
     - geometry comes from the evolved BSSN state

``metric_mass`` defaults to ``1.0`` and ``metric_spin`` to ``0.0``; both apply
only to the Kerr-Schild charts. Dynamical runs must use ``numerical`` or
``flat_cartesian``. Matched GR filtering requires ``flat_cartesian`` or
``numerical``.

The ``[metric]`` table
----------------------

``[metric]`` supplies the initial geometry, the gauge and evolution controls,
and the gravitational boundary codes. It is the recommended initialization
path, and the one every shipped dynamical demo uses.

Initial-data arrays
~~~~~~~~~~~~~~~~~~~

Each state entry is either a ``.npy`` filename or the literal ``false``:

.. code-block:: toml

   [metric]
   conformal_metric = "initial_data/conformal_metric.npy"
   conformal_factor = "initial_data/conformal_factor.npy"
   lapse = "initial_data/lapse.npy"
   # The momentarily static slice: these four are identically zero.
   traceless_K = false
   trace_K = false
   conformal_connection = false
   shift = false
   time = 0.0

Files contain **global C-node arrays without guards**, with tensor components
first. In a multiprocess run each process reads only its own owned tiles from
the same files.

.. warning::

   An omitted or ``false`` entry creates a **zero** array, not an identity. A
   missing ``conformal_metric`` is therefore a zero matrix, not
   :math:`\delta_{ij}`, and a missing ``conformal_factor`` is zero rather than
   one. Both are rejected by the validation below, but the failure mode is worth
   knowing: supply geometry explicitly rather than relying on defaults.

Loading validates that every array is finite, that ``conformal_factor``
contains positive :math:`W`, and that ``conformal_metric`` is symmetric and
positive definite. These checks run before determinant normalization and
inversion, so an omitted geometry file cannot be masked by a floor or a NaN.

Gauge and evolution controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Key
     - Default
     - Meaning
   * - ``gauge``
     - ``"harmonic"``
     - ``"harmonic"`` or ``"1+log"`` slicing
   * - ``zero_shift``
     - ``true``
     - boolean; freezes the supplied shift rather than zeroing it
   * - ``eta``
     - ``2.0``
     - Gamma-driver shift damping
   * - ``g``
     - ``0.75``
     - Gamma-driver source coefficient
   * - ``nu``
     - ``0.002``
     - sixth-order Kreiss--Oliger dissipation coefficient
   * - ``kappa``
     - ``0.0``
     - momentum-constraint damping
   * - ``mad_q``
     - ``1.0``
     - D4 weight in the MAD first/second derivatives
   * - ``time``
     - ``0.0``
     - initial coordinate time

Gauge values are strings here; the underlying numerical-relativity parameter is
an integer, but ``[metric]`` translates for you.

Gravitational boundaries
~~~~~~~~~~~~~~~~~~~~~~~~

``xl_bc``, ``xr_bc``, ``yl_bc``, ``yr_bc``, ``zl_bc`` and ``zr_bc`` accept
``"periodic"`` or ``"sommerfeld"``:

.. code-block:: toml

   xl_bc = "sommerfeld"
   xr_bc = "sommerfeld"
   yl_bc = "sommerfeld"
   yr_bc = "sommerfeld"
   zl_bc = "sommerfeld"
   zr_bc = "sommerfeld"

.. important::

   These six gravitational codes are **independent of the three
   electromagnetic** ``x_bc``/``y_bc``/``z_bc`` settings, and the numeric codes
   do not mean the same thing. Gravitational code ``1`` is Sommerfeld;
   electromagnetic code ``1`` is conducting.

Rejected keys
~~~~~~~~~~~~~

``dx``, ``dt``, ``x_min``, ``y_min`` and ``z_min`` are derived from the PIC grid
and timestep, and are rejected if supplied. Any unrecognized key is also
rejected rather than silently ignored.

``[metric]`` cannot be combined with the legacy ``[bssn_parameters]`` table, nor
with the ``bssn_state`` or ``bssn_evolution_parameters`` Python arguments; pick
one initialization source.

Minimal configuration
---------------------

.. code-block:: toml

   [simulation_parameters]
   name = "collapse"
   solver = "dynamic_metric"
   metric = "numerical"
   output_dir = "outputs"

   Nx = 64
   Ny = 64
   Nz = 64
   x_min = -8.0
   x_max = 8.0
   y_min = -8.0
   y_max = 8.0
   z_min = -8.0
   z_max = 8.0

   [metric]
   conformal_metric = "initial_data/conformal_metric.npy"
   conformal_factor = "initial_data/conformal_factor.npy"
   lapse = "initial_data/lapse.npy"
   gauge = "1+log"
   zero_shift = false
   eta = 2.0
   g = 0.75
   nu = 0.002
   xl_bc = "sommerfeld"
   xr_bc = "sommerfeld"
   yl_bc = "sommerfeld"
   yr_bc = "sommerfeld"
   zl_bc = "sommerfeld"
   zr_bc = "sommerfeld"

   [particle1]
   name = "dust"
   N_particles = 100000

Normal particle blocks and evolved ``fieldN`` blocks may be added. Initial
fields are physical contravariant D/B at native Yee sites, and initial particle
momentum is covariant :math:`u_i` at the position time. Initialization performs
densitization and the leapfrog offset once.

Defaults, and the legacy table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without a ``[metric]`` table the initial geometry is exactly flat, with harmonic
slicing and a frozen zero shift. **That is not a constraint solution for nonzero
plasma**; supply BSSN variables explicitly whenever matter is present.

The older ``[bssn_parameters]`` table is still honored when ``[metric]`` is
absent. It takes raw numerical-relativity values --- integer ``gauge``,
integer ``zero_shift`` --- and has no initial-data array support. Prefer
``[metric]`` for new configurations.

Forced defaults
---------------

Selecting ``dynamic_metric`` forces several choices, overriding the ordinary
defaults rather than inheriting them:

- ``particle_pusher = "hybrid_boris_geodesic"``
- ``current_calculation = "esirkepov"``
- ``filter_j = "none"`` (matched filtering is opt-in, and constrained)
- ``C = eps = mu = kb = 1``
- ``guard_cells`` at least 3

Options reachable only from Python
----------------------------------

Three documented capabilities have no TOML surface, because ``run_PyPIC3D`` does
not forward them to the coupled loop:

- ``source_filter="binomial"`` isolated-electrovacuum source filtering
- ``prescribed_gauge(state, time)`` callbacks
- ``gravity_boundary(state, time)`` callbacks

Reaching any of them requires calling ``time_loop_dynamical_relativity``
directly from a custom driver. See :doc:`einstein_maxwell` for their contracts.

Runtime state and metric changes
--------------------------------

``dynamic_parameters.bssn_parameters`` holds ``state``,
``evolution_parameters`` and ``time``. The dynamic loop returns
``particles, fields, dynamic_parameters``; other solver signatures are
unchanged.

Field slot 7 stores ``(D_previous, B_previous, B_synchronized, half_dt,
absorber)``. Particle momentum and ``B`` are half a native step behind the
integer position/D time. A metric change corrects these lagged samples without
redensitizing the common-time fields or resetting the leapfrog. That correction
assumes the metric changes smoothly: abrupt replacement, or changing ``dt``,
requires a new initialization.

Generic EM-only runs retain conducting and constant exterior boundaries and
reference sponges. ``make_reference_sponge`` in
``boundary_conditions/absorbing_sponge.py`` accepts fixed native D/B reference
arrays, the static and dynamic parameters, and a coordinate ``width``. All
particle box boundaries remain periodic in this coupling; the optional
``lapse_deleting`` interior boundary can be used alongside them.

The four coupling modules are:

* ``relativity/bssn_parameters.py``: ``BSSNParameters`` and its constructor.
* ``relativity/dynamical_relativity.py``: metric/grid transfer, initialization,
  common-time views, and history corrections after a metric change.
* ``relativity/energy_momentum_tensor.py``: Gaussian EM moments and PIC
  deposition of physical Eulerian ``rho``, ``S_i``, and ``S_ij``.
* ``solvers/GR_yee/time_loop_dynamical_relativity.py``: the complete coupled
  step, calling the static-metric loop for each PIC half-step.

Charged-black-hole setup, old checkpoints, and their migration are deferred.

Diagnostics
-----------

Constraint norms are computed from synchronized common-time sources, printed at
``Initial`` and ``Final``, and appended at ``plotting_interval`` cadence:

- ``data/hamiltonian_l2.txt``
- ``data/momentum_l2.txt``

The electromagnetic constraints --- the densitized Gauss residual
:math:`\nabla\cdot D-4\pi\rho` and :math:`\nabla\cdot B` --- are reported
alongside them, using the retained synchronized ``B``.

The optional work-energy budget is enabled with
``gr_work_energy_theorem = true``. It is off by default because it reconstructs
synchronized matter sources every macro step. Its full balance equations,
conventions and output names are in :doc:`einstein_maxwell`.

Flat-space total-energy conservation reports are not applied to dynamical
spacetimes.

Output includes the BSSN fields and the physical common-time D/B. The existing
``E`` mesh name denotes the physical contravariant displacement, as in the
static GR solver. ``J`` retains the last half-step current and is marked with a
time offset of ``-dt/4``; charge budgets across a full macro step must use the
average of the two substep currents.

Particle deletion inside horizons
---------------------------------

``lapse_deleting = true`` deactivates particles where the interpolated lapse
falls below ``lapse_threshold`` (default ``0.1``). Both PIC half-steps apply the
static-GR checks. Dynamical initialization deletes particles before the initial
current deposition and momentum seeding. After a committed metric change,
deletion is applied against the new native lapse before momentum
synchronization. Midpoint positions, momenta and active masks remain fixed
throughout the BSSN RK4 stages, so intermediate stage lapses do not delete
particles. See :doc:`particles` for configuration and the resulting source
sinks.

Running across nodes
--------------------

``dynamic_metric`` is currently the only solver with multiprocess simulation
initialization. See :doc:`distributed_cartesian` for launch commands,
environment variables, and how to merge the per-process openPMD series.

Demos
-----

See :doc:`demos` for Oppenheimer--Snyder dust collapse, charged stellar
collapse, and a fully relativistic two-stream instability.
