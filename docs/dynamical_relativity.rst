Dynamical Relativity
====================

``solver = "dynamic_metric"`` evolves the spacetime geometry with BSSN numerical
relativity while PyPIC3D keeps particles, Maxwell evolution, and the
stress-energy projection. The plasma curves the spacetime, and the evolving
spacetime moves the plasma.

The BSSN equations are **not** part of this repository. They live in the
optional ``JAX_BSSN`` dependency:

.. code-block:: bash

   pip install -e '.[relativity]'

BSSN and EM use identical tile ownership; BSSN retains
wider local guards for its stencils and for the C-to-Yee geometry transfer.

Formulation
-----------

The evolved variables are the standard BSSN set, collocated on ``(C,C,C)``
nodes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symbol
     - Variable
   * - :math:`\tilde\gamma_{ij}`
     - ``conformal_metric``
   * - :math:`W`
     - ``conformal_factor``
   * - :math:`\tilde A_{ij}`
     - ``traceless_K``
   * - :math:`K`
     - ``trace_K``
   * - :math:`\tilde\Gamma^{i}`
     - ``conformal_connection``
   * - :math:`\alpha`
     - ``lapse``
   * - :math:`\beta^{i}`
     - ``shift``

The conformal factor convention is **W**, so the physical spatial metric is
:math:`\gamma_{ij}=\tilde\gamma_{ij}/W^{2}`. The extrinsic-curvature convention
is :math:`K_{ij}=-\tfrac{1}{2}\mathcal L_n\gamma_{ij}`.


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

Selecting the metric
--------------------

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

Diagnostics
-----------

Constraint norms are computed from synchronized common-time sources, printed at
``Initial`` and ``Final``, and appended at ``plotting_interval`` cadence:

- ``data/hamiltonian_l2.txt``
- ``data/momentum_l2.txt``

The electromagnetic constraints --- the densitized Gauss residual
:math:`\nabla\cdot D-4\pi\rho` and :math:`\nabla\cdot B` --- are reported
alongside them, using the retained synchronized ``B``.


A generalization of the work energy theorem in curved spacetime is 
used to track the conservation of energy in the simulations (see 
equation 39 of Chabonov et al. 2021).


.. math::

    P_K &= \int_V \alpha K_{ij}S^{ij}\sqrt{\gamma}\,d^3x, \\
    P_\alpha &= -\int_V S^i\partial_i\alpha\sqrt{\gamma}\,d^3x, \\
    F_{\rm out} &= \oint_{\partial V}
      (\alpha S^i-\beta^i\rho)s_i\,dA, \\
    E(t) &= E(t=0) + \int_{t_0}^t(P_K+P_\alpha-F_{\rm out})\,dt'



Particle deletion inside horizons
---------------------------------

``lapse_deleting = true`` deactivates particles where the interpolated lapse
falls below ``lapse_threshold`` (default ``0.1``). Both PIC half-steps apply the
static-GR checks. See :doc:`particles` for configuration.