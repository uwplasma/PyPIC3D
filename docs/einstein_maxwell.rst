Einstein--Maxwell and PIC coupling
==================================

Ownership and units
-------------------

JAX BSSN supplies numerical relativity: BSSN variables, generic matter right-hand
sides, algebraic projection, gauge evolution, gravitational boundaries and
constraint diagnostics. PyPIC3D owns particles, Maxwell evolution, native Yee
geometry, matter deposition and the coupled time loop. There is no
``JAX_BSSN.EM`` package or runtime compatibility shim.

The generic NR interface remains::

    compute_bssn_rhs_with_matter(state, evolution_parameters, rho, S_i, S_ij)
    enforce_algebraic_constraints(state)

``rho`` is Eulerian energy density, ``S_i`` covariant momentum density and
``S_ij`` covariant spatial stress. Their layouts are ``(nx,ny,nz)``,
``(3,nx,ny,nz)`` and ``(3,3,nx,ny,nz)``. BSSN applies matter sources before its
Sommerfeld face equations. These interfaces do not import PyPIC3D.

The native coupling uses geometrized units with ``C=eps=mu=1`` and Gaussian
Maxwell fields. Ampere's law uses ``-4*pi*J``; flat-space electromagnetic energy
is ``(D**2+B**2)/(8*pi)``. ``gravity_G`` multiplies all three matter projections
passed to BSSN. The rationalized field values used by the former prototype
must be multiplied by ``sqrt(4*pi)`` before using this Gaussian convention.
Existing saved campaigns are not rewritten or silently reinterpreted.

Native PyPIC3D interface
------------------------

The implementation lives in four modules::

    PyPIC3D.relativity.bssn_parameters
    PyPIC3D.relativity.dynamical_relativity
    PyPIC3D.relativity.energy_momentum_tensor
    PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity

``dynamic_parameters.bssn_parameters`` is a ``BSSNParameters`` tuple containing
``state``, ``evolution_parameters``, ``gravity_G`` and ``time``. The nested NR
parameters are the existing ``JAX_BSSN.bssn.variables.BSSNParameters`` type;
gravitational boundary codes remain there and are independent of EM codes.
In particular, code 1 means Sommerfeld to BSSN and conducting to PyPIC3D.

The host initialization entry takes common-time physical native fields and
common-time covariant particle momenta::

    from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
    from PyPIC3D.relativity.dynamical_relativity import initialize_dynamical_relativity
    from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity

    bssn = build_bssn_parameters(static_parameters, dynamic_parameters,
                                state=initial_bssn, gravity_G=1.0)
    dynamic_parameters = dynamic_parameters._replace(bssn_parameters=bssn)
    particles, fields, dynamic_parameters = initialize_dynamical_relativity(
        particles, species, fields, static_parameters, dynamic_parameters)
    particles, fields, dynamic_parameters = time_loop_dynamical_relativity(
        particles, species, fields, static_parameters, dynamic_parameters)

The caller JITs the step with static parameters and optional pure
``prescribed_gauge(state,time)`` and ``gravity_boundary(state,time)`` callbacks
closed over. Gauge callbacks return lapse, shift, their time derivatives;
boundary callbacks return BSSN state. Other PIC solvers keep their existing
return signatures.

Default initial geometry is exactly flat, with harmonic slicing and frozen
zero shift. It is not a constraint solution for nonzero plasma. Supply BSSN
variables explicitly when different NR initial data are required. The first
particle coupling is periodic Cartesian evolution on one device, with equal
spacing on resolved axes. Empty-particle EM runs retain periodic, conducting,
constant and absorbing native field boundaries. Prescribed external fields,
PML and non-Cartesian coordinate charts are not part of this coupling.

Time levels and sources
-----------------------

One macro step is PIC(dt/2), BSSN(dt), PIC(dt/2). The BSSN interval uses RK4,
but the coupled Strang method is second order. Every stage uses the same
midpoint particle phase-space data and native EM quadratic moments, contracted
with that stage's metric. Native samples are squared before averaging to
BSSN nodes, preserving unresolved alternating-sign field energy.

The field tuple retains its nine top-level slots. Its history slot is::

    (D_previous, B_previous, B_synchronized, half_dt, absorber)

Geometry exchange averages lapse, shift, W and conformal metric from C to
native Yee locations, then projects the conformal determinant and derives
physical metric, inverse and volume. A V sample is above C by half a cell.
Particle geometry uses the existing consistent Hermite reconstruction of 
these metric samples; Maxwell curls and production particle/current 
operators remain in PyPIC3D.

Isolated electrovacuum source filtering
---------------------------------------

The coupled step, ``compute_energy_momentum_tensor`` and
``bssn_constraint_norms`` accept ``source_filter="none"`` (the default) or
``source_filter="binomial"``. The latter requires empty particle storage,
Cartesian geometry and ``current_filter="none"``. It applies the existing
separable ``[1,2,1]/4`` stencil once to coordinate stress-energy densities at
each RK stage, dividing afterwards by the unfiltered stage volume. Periodic
axes wrap; isolated axes replicate their edge samples. Native EM products
are formed before smoothing, and stored Maxwell fields and BSSN variables
are not smoothed. This option does not change the periodic matched-PIC filter
contract and cannot be combined with that filter.

The charged electrovacuum demo in
``demos/dynamical_relativity/charged_vacuum_puncture`` implements analytic
single-puncture data from arXiv:1311.6483v2 in the native Gaussian convention.
It retains the analytic initial metric and reports constraints with both raw
and filtered sources; filtering changes the gravitational source
approximation rather than solving a new initial constraint problem.

Matter and electromagnetic work-energy theorem
----------------------------------------------

Enable the diagnostic for ``solver = "dynamic_metric"`` in the existing
simulation parameter table::

    [simulation_parameters]
    solver = "dynamic_metric"
    gr_work_energy_theorem = true
    # Optional positive energy scale, in the same units as the measured energy:
    # gr_work_energy_reference = 1.0

It is disabled by default because it reconstructs synchronized matter sources
at every macro step. The diagnostic does not modify the evolution equations,
gauge, particle push, Maxwell updates, or time integration. It monitors the
total particle-plus-Maxwell budget and also reports the separately measured
field and particle energies. It does not form separate field and particle
balance equations.

The runtime state lives in ``dynamic_parameters.work_energy_tracker``. Python
drivers can read its scalar output with::

    from PyPIC3D.diagnostics.gr_work_energy_theorem import tracker_values

    values = tracker_values(dynamic_parameters.work_energy_tracker)

Conventions and synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stored native fields are contravariant densities,
``D^i = sqrt(gamma) E_Eulerian^i`` and
``B^i = sqrt(gamma) B_Eulerian^i``. The lapse- and shift-dependent covectors
used in Maxwell curls are auxiliary fields, not the Eulerian fields used in
the stress-energy tensor. The diagnostic reuses the solver's electromagnetic
moments and stress-energy contractions, including non-diagonal metric terms.
Native field products are formed before transferring them to C nodes.
The physical metric is ``gamma_ij = conformal_metric_ij / W**2``, and the
extrinsic-curvature convention is ``K_ij = -L_n(gamma_ij)/2``.
The physical curvature is reconstructed as
``K_ij = (traceless_K_ij + conformal_metric_ij * trace_K / 3) / W**2``;
stress indices are raised with the full physical inverse metric.
The diagnostic measures physical matter energy without multiplying by the
coupling parameter ``gravity_G``.

Particles store covariant spatial four-velocity ``u_i``. Their Eulerian energy
includes rest energy and macroparticle weight:
``mass * weight * sqrt(1 + gamma^ij u_i u_j)``. Particle energy is evaluated
with the existing metric reconstruction and Lorentz-factor convention.
The stored current is a densitized coordinate transport current. No
``J dot E`` term is added to this total budget: electromagnetic-to-particle
exchange cancels between its two components.

Each sample uses D, the retained synchronized B, synchronized particle
momenta at their position time, and the metric at that same macro-step time.
Geometric power and outward flux are integrated by the endpoint trapezoidal
rule every macro step. This is second-order quadrature for the second-order
PIC/BSSN split; the BSSN RK4 stages do not supply common-time Maxwell and
particle RK4 stages. The diagnostic is a reconstructed continuum energy,
not a leapfrog modified discrete Hamiltonian, and is not expected to remain
constant to roundoff even for a periodic flat-space wave.

The baseline is established at the first synchronized state. On the first
step it is aligned with any prescribed geometry callbacks before evolution;
the driver's initial output row is therefore emitted after that step from
the retained initial sample. Changing ``plotting_interval`` changes only
writing, not integration. The final endpoint is written even when it falls
between output times; a zero-step run writes one initial sample.

Balance, domain and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The monitored region is the full declared, fixed Cartesian computational
domain. Volume and surface quadrature exclude ghost cells and count physical
boundaries once. Periodic axes use the sum of owned C nodes times the grid
spacing; conducting axes use trapezoidal weights with half-weight endpoints.
When a nonperiodic declared
upper endpoint is not represented by a C node, the endpoint values are
reconstructed as ``q_0 = 2*q_1 - q_2`` and
``q_N = 2*q_(N-1) - q_(N-2)``. Reconstructing the lower endpoint too avoids
using its halo-contaminated native field moments. These values enter the
trapezoidal volume quadrature and face flux; the volume is not silently
shortened to the last stored node. Each open axis requires at least three
stored C nodes.
Flux includes both lapse-weighted Eulerian transport and shift advection.
Periodic opposite faces cancel. Collapsed Cartesian axes retain their
physical transverse widths in the three-dimensional volume and face areas.

The diagnostic does not improve the solver's boundary accuracy. In the
constant-shift outgoing-pulse test, constant exterior ghosts produce a
boundary error while the pulse crosses the face. The measured energy
converges at about second order, but the native field and balance residual
do not show a uniform second-order rate over the tested grids. Boundary
transport errors remain visible in the continuum residual.

With covariant momentum density ``S_i`` and covariant stress ``S_ij`` from the
matter source, full metric contractions give

.. math::

    P_K &= \int_V \alpha K_{ij}S^{ij}\sqrt{\gamma}\,d^3x, \\
    P_\alpha &= -\int_V S^i\partial_i\alpha\sqrt{\gamma}\,d^3x, \\
    F_{\rm out} &= \oint_{\partial V}
      (\alpha S^i-\beta^i\rho)s_i\,dA, \\
    E_{\rm budget}(t) &= E_0 + \int_{t_0}^t(P_K+P_\alpha-F_{\rm out})\,dt', \\
    R_E(t) &= E_{\rm measured}(t)-E_{\rm budget}(t).

Positive flux leaves the domain. Positive residual is unaccounted energy
gain; negative residual is unaccounted loss. The absorber's signed modeled
power and integrated work are reported separately and are not subtracted
from this continuum residual. This models the instantaneous energy change
from the damping term, not the exact finite-step energy removal from all
predictor, history, and synchronization operations. A sponge with a nonzero
reference field can supply energy, so its reported work need not be negative.
Filtering and numerical dissipation therefore remain visible as departures
from the original continuum balance.

The fixed normalization is ``abs(residual) / reference``. Its default is the
initial total energy, including particle rest energy. An initially empty,
zero-field domain requires an explicit finite positive
``gr_work_energy_reference``. A positive field or kinetic-energy scale can also
be supplied for a rest-energy-dominated plasma; this does not remove rest
energy from the underlying balance.

The enabled diagnostic requires fully evolving active particle species;
frozen or partially frozen species are rejected because their imposed motion
requires external work accounting. Particle runs retain the solver's periodic
boundary restriction. Empty-particle runs support the native periodic,
conducting, constant and absorbing field boundaries. There is no particle
injection or open-particle-boundary accounting in this coupled solver.
Non-Cartesian spherical/axisymmetric charts, moving regions, horizon masks,
subvolumes, mesh refinement and distributed coupled evolution are unsupported.
A horizon inside the monitored domain contributes no additional boundary.
Checkpoint/restart support is outside this diagnostic's scope.

Scalar output
~~~~~~~~~~~~~

The existing text time series use ``data/work_energy_<name>.txt``, each with
``time, value`` rows. Every name pairs an instantaneous rate ``*_power`` with
its time integral ``*_work``. Names are:

* ``total``, ``electromagnetic``, ``particles`` and ``budget``;
* ``curvature_power``, ``lapse_power``, ``geometry_power`` and ``boundary_power``;
* ``curvature_work``, ``lapse_work``, ``geometry_work`` and ``boundary_work``;
* ``absorbing_boundary_power`` and ``absorbing_boundary_work``;
* ``residual``, ``step_residual``, ``relative_residual`` and ``reference``;
* ``baseline_time`` and ``baseline_energy``.

``step_residual`` belongs to the most recent macro step, not the interval
between output rows. Accumulators are fixed-size JAX state; output cadence
does not require growing histories or per-step scalar host transfers.

This diagnostic measures matter/EM energy-balance error on the numerical
geometry. It neither bounds the complete solution error nor replaces
Einstein/Maxwell constraints, ADM mass, or quasi-local mass diagnostics.
For the geometric-source interpretation of matter continuity equations, see
Katy Clough, `Continuity equations for general matter: applications in
numerical relativity <https://arxiv.org/abs/2104.13420>`_.
