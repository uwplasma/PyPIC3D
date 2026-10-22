Einstein--Maxwell and PIC coupling
=================================

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
-----------------------

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