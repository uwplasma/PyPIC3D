Field Solvers
=============

PyPIC3D provides these solver names:

- ``electrodynamic_yee``
- ``electrostatic``
- ``static_metric``
- ``dynamic_metric``
- ``dark_matter_yee``

Electrodynamic Yee Step
-----------------------

The electrodynamic timestep keeps both particles and fields tiled. Its order is:

1. Interpolate total electric and magnetic fields to particles and push
   velocity.
2. Deposit current:

   - Direct deposition advances position by ``dt/2``, deposits at the
     centered position, then completes the second ``dt/2``.
   - Esirkepov advances position once by ``dt``, then deposits from the old and
     new endpoints before wrapping particles or refreshing tile ownership.

3. Update ``B`` by a half timestep from the old ``E``.
4. Update ``E`` by a full timestep from the half-step ``B`` and deposited
   ``J``.
5. Update ``B`` by a second half timestep from the new ``E``.

The field equations are:

.. math::

   \mathbf{B}^{n+1/2} =
   \mathbf{B}^{n} - \frac{\Delta t}{2}\nabla\times\mathbf{E}^{n},

.. math::

   \mathbf{E}^{n+1} =
   \mathbf{E}^{n} + \Delta t\left(
   c^2\nabla\times\mathbf{B}^{n+1/2}
   - \frac{\mathbf{J}^{n+1/2}}{\epsilon_0}\right),

.. math::

   \mathbf{B}^{n+1} =
   \mathbf{B}^{n+1/2}
   - \frac{\Delta t}{2}\nabla\times\mathbf{E}^{n+1}.

Dark Photon Maxwell-Proca Solver
--------------------------------

``dark_matter_yee`` evolves an additional massive, kinetically mixed vector
field alongside the ordinary Maxwell fields, using periodic or conducting field
boundaries and optional PML layers. The same boundary configuration applies to
both sectors, which keep independent histories.

Ordinary Maxwell evolution is unchanged; particles feel
``E - sin_chi*dark_E`` and ``B - sin_chi*dark_B`` plus prescribed external
fields. The field equations, the Gauss constraint, the Proca timestep limit,
initial data and diagnostics are in :doc:`dark_photon`.

Electrostatic Step
------------------

The electrostatic timestep:

1. Pushes particle velocity from fields.
2. Advances position by ``dt``.
3. Deposits charge density ``rho``.
4. Solves Poisson in tiled storage with residual-controlled parallel Schwarz iterations.
5. Computes ``E = -grad(phi)``.

Each Schwarz iteration solves the owned potential on every tile with
tile-local conjugate gradient while holding ghost cells fixed.
Converged tiles stop updating through an active mask while other tile solves
continue. CG exits at ``electrostatic_local_cg_tol`` or
``electrostatic_local_cg_max_iterations`` and reduces only over each tile's
three owned spatial axes.

After each local solve, neighboring ``phi`` halos and physical conducting
boundaries are refreshed. The true Poisson residual is then evaluated in the
``guard_cells``-wide owned slabs next to tile interfaces. Schwarz exits at
``electrostatic_schwarz_tol`` or
``electrostatic_schwarz_max_iterations``. The previous timestep's potential is
the warm start, and an already-converged state performs no unnecessary solve.
No global potential, charge field, or Krylov reduction is assembled. The 
existing guard depth is the Schwarz overlap width.

Particle Pushers
----------------

``particle_pusher = "boris"`` selects Boris. With ``relativistic = true`` it
uses the relativistic Boris update; otherwise it uses the non-relativistic
form. ``particle_pusher = "higuera_cary"`` selects the Higuera-Cary relativistic
update.

Particles interpolate the sum of evolved and prescribed external fields.
Maxwell updates use only evolved fields.

GR Maxwell Solver
--------------------------

The static-GR solver evolves only native densities. Slots 0, 1, and 2 of the
runtime tuple store ``sqrt(gamma) D^i``, ``sqrt(gamma) B^i``, and
``sqrt(gamma) J^i`` on their native Yee component locations; the external
D/B pair in slot 5 and the previous D/B in slot 7 are densities too. rho,
phi, the metric, and the overflow flag keep their own conventions.

``update_D`` and ``update_B`` advance coordinate curls without dividing the
increments by the metric volume. The compute auxiliary fields methods use 
averaging to move the fields to the target locations on the Yee grid. 

The ``static_metric`` solver uses ``hybrid_boris_geodesic``. Every particle
metric sample uses the shared ``interpolate_metric`` function: the velocity
update, position midpoint, direct GR current deposition, and particle-birth
momentum conversion all use the same reconstruction. Leapfrog initialization
also uses it through the shared pusher. ``esirkepov`` deposits from particle
endpoints and does not need a separate particle-metric sample.

The sampler reconstructs lapse, shift, and the covariant spatial metric from
``YeeMetric.center`` using Hermite interpolation.

The chart is selected with the ``metric`` key, which accepts
``flat_cartesian``, ``flat_cylindrical``, ``flat_spherical``,
``kerr_schild_cartesian`` and ``kerr_schild_spherical``; the Kerr-Schild charts
also read ``metric_mass`` and ``metric_spin``. See
:doc:`dynamical_relativity` for the full table, including the ``numerical``
chart used by the dynamical solver.

The optional ``lapse_deleting`` particle boundary samples the native center
lapse and deactivates particles below ``lapse_threshold``; see :doc:`particles`
for configuration and the resulting particle-source sinks. Each static-GR
step applies it after optional injection, before storing the old particle
endpoint, and again after current deposition and normal boundary refresh.
The endpoint check therefore uses the destination tile's metric. Both current
methods retain the crossing step's final current, and deletion leaves previous
currents and field histories intact.


Current Deposition
------------------

Select deposition with:

.. code-block:: toml

   current_calculation = "j_from_rhov"

or:

.. code-block:: toml

   current_calculation = "esirkepov"
   filter_j = "none"

See :doc:`chargeconservation` for deposition timing and filter behavior.

Dynamical BSSN-PIC Solver
-------------------------

``dynamic_metric`` evolves geometry with JAX BSSN while keeping Maxwell,
particles, and stress-energy in PyPIC3D. Install JAX BSSN separately or use the
``relativity`` optional dependency; importing ordinary PyPIC3D solvers does not
import it.

One macro step is PIC(dt/2), BSSN(dt), PIC(dt/2). The BSSN interval uses RK4,
but the coupled Strang split is second order overall. Each gravity stage
resamples particle geometry and recomputes particle stress from fixed midpoint
positions and covariant momenta.

Configuration --- the ``[metric]`` table, gauge conditions, gravitational
boundaries, forced defaults and diagnostics --- is in
:doc:`dynamical_relativity`.
