Current Deposition
==================

PyPIC3D deposits current directly into tile-local Yee arrays. Contributions
that land in tile ghost cells are merged into neighboring tiles, followed by 
a ghost cell update.

Direct ``j_from_rhov``
----------------------

The direct method pushes particles to the centered position
``x + u*dt/2``, updates which tile they belong to, and deposits 
weighted charge times velocity on the staggered current component
grids. It then completes the second half of the position update.

Configure it with:

.. code-block:: toml

   current_calculation = "j_from_rhov"
   filter_j = "bilinear"

Supported filters are:

- ``none``: no current smoothing.
- ``bilinear``: tri-linear smoothing.
- ``digital``: nearest-neighbor digital filtering with coefficient ``alpha``.

The filter runs after ghost cell updates. Current ghosts are refreshed before and
after smoothing so each tile reads completed neighboring values. The same filter 
is applied to the evolved electric field used for interpolation so that the deposition 
and interpolation methods use the same stencil and remain consistent. Charge density 
uses the same digital filter when ``filter_j = "digital"``.

Esirkepov
---------

All electromagnetic solvers call the same ``Esirkepov_current`` method with
explicit old and new particle states. It builds aligned particle-shape stencils
and deposits the charge-conserving current before wrapping, boundary handling,
or changes to particle ownership. Both states must retain matching slots,
activity, and tile ownership.

The Python interface is::

   Esirkepov_current(particles_old, particles_new, species_config, J,
                     static_parameters, dynamic_parameters,
                     *, coordinate_velocity=None)

The optional keyword-only ``coordinate_velocity`` has the same shape as particle
positions and is used only for current along unresolved grid axes. If omitted,
that velocity is derived from endpoint displacement divided by ``dt``. Frozen
species axes deposit no current. Ordinary and dark-matter Yee advance positions
once and supply their coordinate velocity explicitly to avoid subtraction error.

It supports shape factors 1 and 2 and reduced 1D/2D axes. Current filtering is
disabled for Esirkepov because the discrete continuity equation is satisfied
exactly. Configure it with:

.. code-block:: toml

   current_calculation = "esirkepov"
   filter_j = "none"

Initialization rejects Esirkepov combined with ``digital`` or ``bilinear``
current filtering so the discrete continuity construction is not silently
altered.

Fixed-metric schemes
--------------------

The ``static_metric`` solver stores and evolves the native Yee densities
``sqrt(gamma) D^i`` and ``sqrt(gamma) B^i``, including their previous time
levels. Its stored current and current time averages are ``sqrt(gamma) J^i``.
``Esirkepov_current`` and ``GR_direct_deposition`` both return that
density directly, and optional ``current_transform`` callbacks receive and
return it.

``GR_direct_deposition`` is a metric-weighted volume deposit. It samples the
lapse, shift and covariant metric through the shared Hermite reconstruction,
derives the inverse from that tensor, and forms ``alpha v^i - beta^i``.
It supports current filtering. It does not satisfy the discrete continuity
equation, so Gauss's law violation accumulates over a run.

``esirkepov`` is charge conserving. It works because the conformal charge
density carries no metric,

.. math::

   \sqrt{\gamma}\,\rho = q\,S(x) / (\Delta x\,\Delta y\,\Delta z),

so the conformal continuity equation

.. math::

   \partial_t(\sqrt{\gamma}\rho) + \partial_i(\sqrt{\gamma}J^i) = 0

is the flat Esirkepov identity verbatim, and the same density decomposition
applies unchanged. Flat and GR solvers use the same endpoint-based method and
metric-free ``esirkepov_tile_currents`` kernel.

The GR loop retains pre-push positions and supplies the actual post-push
positions. Its ``particles.u`` stores covariant ``u_i``, which the depositor
never reads. GR omits ``coordinate_velocity`` so reduced-axis current uses
``(x^{n+1} - x^n)/dt``. The depositor needs no metric interpolation and returns
the density directly for Maxwell evolution.

Because the backward-difference divergence of the backward-difference curl in
``update_D`` vanishes identically, satisfying discrete continuity
preserves

.. math::

   \partial_i(\sqrt{\gamma}D^i) = 4\pi\sqrt{\gamma}\rho

to round-off with no divergence cleaning. Configure it with:

.. code-block:: toml

   solver = "static_metric"
   current_calculation = "esirkepov"
   particle_pusher = "hybrid_boris_geodesic"
   filter_j = "none"

Current filtering is rejected for ``esirkepov`` for the same reason as the
flat scheme. ``GR_direct_deposition`` remains the default for
``static_metric``.

Reference
---------

Esirkepov, T. Z. (2001). Exact charge conservation scheme for particle-in-cell
simulation with an arbitrary form-factor. *Computer Physics Communications*,
135(2), 144-153.
