Field Solvers
=============

PyPIC3D provides these solver names:

- ``electrodynamic_yee``
- ``electrostatic``
- ``static_metric``
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
------------------------------

``dark_matter_yee`` evolves an additional massive vector field alongside the
ordinary Maxwell fields, using periodic or conducting field boundaries and
optional PML layers. The same boundary configuration applies to both sectors.
``sin_chi`` is the dimensionless mixing coefficient (default zero, between
-1 and 1); ``dark_mu`` is a dark photon mass (default zero), distinct
from the electromagnetic permeability ``mu``.

With :math:`s=\mathtt{sin\_chi}` and :math:`m=\mathtt{dark\_mu}`, the equations are

.. math::

   \dot{\mathbf E}'=\nabla \times\mathbf B'
      +\mu^2\mathbf A'+sin(\chi_0) \mathbf J,\qquad
   \mathbf B'=\nabla \times\mathbf A',

.. math::

   \dot{\mathbf A}'=-\mathbf E'- \nabla \phi',\qquad
   \dot\phi'=- \nabla \cdot \mathbf A'.





``A`` and ``E`` share componentwise electric-edge locations; ``phi`` lives at
``(C,C,C)`` nodes. Particles feel ``E - sin_chi*dark_E`` and ``B - sin_chi*dark_B``, plus prescribed
external fields. For the integer-time particle push, ``dark_B`` is the curl of
the average of the adjacent half-step vector potentials. The electric gather
uses the same filter as direct-current deposition.

Conducting walls impose homogeneous reflecting Proca conditions: zero scalar
potential, tangential vector potential and electric field, and normal magnetic
field. Across each physical wall, ``phi`` and tangential ``A/E`` have odd
parity, normal ``A/E`` have even parity, and ``B`` has the opposite vector
parities. Reflections use the actual Yee endpoints, including edges and corners.

The existing ``[[pml]]`` wall configuration also absorbs transverse and
longitudinal dark waves. The dark solver stretches both curls, the scalar
gradient, and the vector divergence. Its auxiliary state stores a magnetic
field at half time and 18 derivative histories, separate from the Maxwell PML
histories.

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
