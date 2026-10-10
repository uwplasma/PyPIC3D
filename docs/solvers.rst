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

Dark Photon Maxwell-Proca Step
------------------------------

``dark_matter_yee`` evolves an additional massive vector field alongside the
ordinary Maxwell fields, using periodic or conducting field boundaries and
optional PML layers. The same boundary configuration applies to both sectors.
Constant field walls and supergaussian absorbers are not supported in this mode.
``sin_chi`` is the dimensionless mixing coefficient (default zero, between
-1 and 1); ``dark_mu`` is a dark photon mass (default zero), distinct
from the electromagnetic permeability ``mu``.

With :math:`s=\mathtt{sin\_chi}` and :math:`m=\mathtt{dark\_mu}`, the equations are

.. math::

   \dot{\mathbf E}'=c^2\nabla_h\times\mathbf B'
      +c^2m^2\mathbf A'+s\mathbf J/\epsilon,\qquad
   \mathbf B'=\nabla_h\times\mathbf A',

.. math::

   \dot{\mathbf A}'=-\mathbf E'-G_h\phi',\qquad
   \dot\phi'=-c^2 D_h\mathbf A'.

``A`` and ``E`` share componentwise electric-edge locations; ``phi`` lives at
``(C,C,C)`` nodes. The forward gradient ``G_h`` maps nodes to electric edges;
the backward divergence ``D_h`` maps those edges back to nodes. The live dark
state is ``(E^n, A^(n-1/2), phi^n)``. Each step drifts ``A`` once by ``dt`` and
then kicks both ``E`` and ``phi`` once using the new half-step ``A`` and the
centered deposited current. The positive mass term gives
:math:`\omega^2=c^2(k^2+m^2)` for a free wave.

Particles feel ``E - sin_chi*dark_E`` and ``B - sin_chi*dark_B``, plus prescribed
external fields. For the integer-time particle push, ``dark_B`` is the curl of
the average of the adjacent half-step vector potentials. The electric gather
uses the same filter as direct-current deposition. Zero mixing decouples the
dark field from both particles and Maxwell evolution.

Initial data are supplied at ``t=0``; initialization stores
``A^(-1/2) = A(0) + dt/2 * (E(0) + G_h phi(0))``. For constraint-compatible
data, require ``D_h E' + dark_mu**2 * phi' = -sin_chi * rho / eps`` initially.
The initializer does not solve this constraint. Its discrete preservation with
particles requires charge-conserving current deposition.

Conducting walls impose homogeneous reflecting Proca conditions: zero scalar
potential, tangential vector potential and electric field, and normal magnetic
field. Across each physical wall, ``phi`` and tangential ``A/E`` have odd
parity, normal ``A/E`` have even parity, and ``B`` has the opposite vector
parities. Reflections use the actual Yee endpoints, including edges and corners.
This is an ideal reflecting wall for each sector, not a material model of a
conductor interacting only with the mixed force field.

The existing ``[[pml]]`` wall configuration also absorbs transverse and
longitudinal dark waves. The dark solver stretches both curls, the scalar
gradient, and the vector divergence. Its auxiliary state stores a magnetic
field at half time and 18 derivative histories, separate from the Maxwell PML
histories. Conductivity is interpolated from the existing nodal profiles to the
appropriate Yee derivative locations. Mass and current terms are unchanged.

For a directional derivative ``d``, the dark history satisfies
``psi_dot = -sigma*(psi + d)``. Each stage integrates this equation exponentially
with ``d`` at the stage midpoint and uses the interval-averaged stretched
derivative in the field update. ``A/B`` and their gradient/Faraday histories
advance between half times; ``E/phi`` and the Ampere/divergence histories advance
between integer times. Histories start at zero at physical ``t=0`` before the
backward half-step seed. Nonzero initial fields within the layer therefore
represent a fresh absorber startup, not an established outgoing-wave history.

Particle gathers, field output, and energy diagnostics preview a half drift of
``A/B`` without modifying live histories. Inside PML, ``dark_B`` comes from this
auxiliary magnetic field rather than an ordinary ``curl(A)``. Outside the layer,
the original Proca equations and staggering apply. PML intentionally removes
energy; the physical Gauss constraint is assessed outside the layer and away
from wall surface-charge nodes. Particle boundaries and deposition remain
independent settings. Place sources away from PML when measuring absorption.

The explicit field stability bound is

.. math::

   \Delta t < \frac{2}{c\sqrt{m^2+4\sum_{N_i>1}\Delta x_i^{-2}}}.

Automatic timesteps take the smaller of the Maxwell CFL timestep and
``0.99*cfl`` times this bound. A massless, spatially uniform problem requires
an explicit timestep. The bound concerns field evolution; particle/plasma
timescales may require a smaller timestep.

Field output includes ``dark_E``, ``dark_A``, ``dark_phi``, and ``dark_B``, all
synchronized to integer time. Total energy includes
:math:`\epsilon\int(E'^2+c^2B'^2+c^2m^2A'^2+m^2\phi'^2)\,dV/2`;
the dark contribution is also written to ``dark_field_energy.txt``. This is a
second-order physical-energy diagnostic, not an exactly conserved discrete
leapfrog energy.

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

Static-Metric Maxwell State
--------------------------

The static-GR solver evolves only native densities. Slots 0, 1, and 2 of the
runtime tuple store ``sqrt(gamma) D^i``, ``sqrt(gamma) B^i``, and
``sqrt(gamma) J^i`` on their native Yee component locations; the external
D/B pair in slot 5 and the previous D/B in slot 7 are densities too. rho,
phi, the metric, and the overflow flag keep their own conventions.

``update_D`` and ``update_B`` advance coordinate curls without dividing the
increments by the metric volume. The constitutive ``compute_covariant_E`` and
``compute_covariant_H`` average densities to each target location and divide
by the target ``sqrt_gamma`` to recover physical components. Time centering,
current deposition (``esirkepov`` and ``GR_direct`` both return
``sqrt(gamma) J^i``), any ``current_transform``, and the boundary refresh all
operate on densities. The particle push receives the density sum of evolved
and external fields and divides each component by its grid-node
``sqrt_gamma`` before gathering.

Physical contravariant vectors appear only at the file boundary.
Initialization densitizes TOML D/B/J, external, and previous fields once
(``relativity.field_state.densitize_vector``), and the output adapters write
physical D/B/J through ``physical_fields`` with their existing names,
including the static-GR displacement field written as ``E``. Manually
constructed physical states can pass through ``densitize_fields`` once
before evolution.

This state convention prepares Maxwell evolution for a changing metric;
the solver still prescribes a fixed metric and does not evolve spacetime.

Static-Metric Particle Sampling
--------------------------------

The ``static_metric`` solver uses ``hybrid_boris_geodesic``. Every particle
metric sample uses the shared ``interpolate_metric`` function: the velocity
update, position midpoint, direct GR current deposition, and particle-birth
momentum conversion all use the same reconstruction. Leapfrog initialization
also uses it through the shared pusher. ``esirkepov`` deposits from particle
endpoints and does not need a separate particle-metric sample.

The sampler reconstructs lapse, shift, and the covariant spatial metric from
``YeeMetric.center`` on the base C grid using tensor-product cardinal cubic
Hermite polynomials (Catmull--Rom), with centered nodal slopes. It computes
the inverse and determinant from the reconstructed tensor rather than
interpolating the stored grid inverse or determinant. Analytic metric
providers supply grid values; there is no analytic or lower-order fallback
at particle positions and no interpolation-method setting.

Metric derivatives are derivatives of the same interpolant, including

.. math::

   \partial_k\gamma^{-1}
   = -\gamma^{-1}(\partial_k\gamma)\gamma^{-1}.

``ParticleMetric.grad_lapse[..., k]`` stores ``d_k alpha``,
``grad_shift[..., i, k]`` stores ``d_k beta^i``, and
``grad_gamma_inv[..., k, i, j]`` stores ``d_k gamma^ij``.
Sampling with ``derivatives=False`` returns the same metric values and
``None`` for the derivative fields. Smooth non-polynomial data generally
give third-order values and second-order derivatives; polynomial degree
alone does not imply fourth-order accuracy because the slopes are estimated.

The four-node stencil in each resolved direction requires at least three
guard cells for particle and midpoint sampling before tile migration.
Configurations using the GR particle sampler default to three and reject
smaller values. A globally single-cell direction uses its designated base
node and has zero metric derivative; a single-cell tile in a resolved
direction still interpolates normally.

Unused particle slots are sampled at safe interior positions. Samples outside
the available stencil produce NaNs, and checked execution reports invalid
active samples, nonpositive lapse, or invalid spatial tensors without repairs.
The derived volume factor preserves the chart's signed orientation in
reflected spherical and cylindrical guards.

This contract is independent of ``shape_factor``: electromagnetic field
gathering and particle deposition retain their selected particle shapes.
Output metadata identifies the reconstruction as
``cardinal_cubic_hermite_consistent_v1``; this is descriptive metadata, not
a selectable numerical mode.

Boundary Conditions and PML
---------------------------

Spherical runs use explicit angular bounds, conducting fields, and reflecting
particles on a regular chart. Metric guard nodes must stay away from the axes.

Field boundaries are set with ``x_bc``, ``y_bc``, and ``z_bc``:

- ``periodic``
- ``conducting``

Standard Yee conducting boundaries zero tangential electric components at
``g`` and ``g+n`` on the first and last tiles, so the cavity width is
``N*dx``. Upper C endpoints are owned physical nodes even though stored in a
halo slot; neighbor exchange preserves them, including transverse
communication.

For the static-metric finite-difference solver, ``conducting`` instead imposes
FIDO-field constraints on the evolved contravariant vectors:

.. math::

   G^i{}_j=n^i n_j, \qquad F^i{}_j=\delta^i{}_j-G^i{}_j,
   \qquad D^i\leftarrow G^i{}_jD^j,
   \qquad B^i\leftarrow F^i{}_jB^j.

At a coordinate face ``x^a=constant``, the spatial-metric unit normal is
``n_i=delta_i^a/sqrt(gamma^{aa})`` and
``n^i=gamma^{ia}/sqrt(gamma^{aa})``. A physically normal D can therefore have
nonzero coordinate-tangential components. Surface tensors are taken from the
existing C/V metric samples, with density-weighted component transfers to a
common location before projection.

``enforce_pec_D`` and ``enforce_pec_B`` in ``boundary_conditions/pec.py``
apply these constraints on the native Yee nodes:

- On a wall, each native D component keeps only its row of the normal
  projection, ``D^i <- (gamma^{ia}/gamma^{aa}) D^a``, and each native B
  component has that row removed. D vanishes on nodes where two or more
  walls intersect. Each B component has a single C axis, so it meets at most
  one wall.
- Where an off-diagonal ``gamma^{pq}`` couples the normals of two walls
  meeting at an edge, the D wall rows beside the edge read each other through
  the component transfer. ``solve_D_edges`` solves each coupled pair exactly
  instead of taking a single pass over a common snapshot.
- Exterior ghosts are filled by reflection about the wall nodes, ``2G-I``
  for D and ``2F-I`` for B, composed in x, y, z order at corners. For B, the
  ghost pair beside each edge is coupled the same way; ``solve_B_edges``
  solves it from two reflection sweeps, and a further sweep fills
  triple-corner ghosts when all three axes are conducting.
- Internal tile boundaries are communicated, never projected; halo exchange
  preserves the conducting exterior slabs.

Each conducting axis of a ``static_metric`` run needs at least
``guard_cells + 1`` cells, so that mirror images about one wall do not reach
the opposite wall. Initialization rejects narrower axes.

The projector algebra is exact at the reconstruction points; interpolating
the stored staggered fields to a common surface introduces truncation error.

``staggered.refresh_fields`` takes D/B densities and always requires
``metric``; it freezes any configured horizon layers before the projection.
The boundary policies are defined on the physical vector. Projector rows
are homogeneous, so wall nodes are projected from co-located density
reconstructions directly. Every operation that copies or reflects a value
into another node (horizon freezing, constant extrapolation, PEC exterior
reflections) rescales it by ``sqrt_gamma(target)/sqrt_gamma(source)``.
Owned densities are never touched, and the result equals refreshing the
physical vector to round-off. Volumes may be signed past a polar axis.
Periodic and tile-seam halos join the same physical point and copy
densities directly, which assumes a periodic metric on periodic axes.

D and B are refreshed when they are produced: ``update_D`` and ``update_B``
return refreshed fields, and initialization refreshes the initial and
previous-time-level D and B.
External fields are refreshed once at initialization with
``refresh_fields(..., 'D')`` and ``refresh_fields(..., 'B')``. The time loop
relies on this contract. It does not refresh D or B again, and it does not
refresh the sum of evolved and external fields used for the particle push;
both terms are already refreshed and the refresh is linear. Particle
reflection and source folding do not use these projectors; they follow the
particle-boundary rules in :doc:`tiling`.

The auxiliary E/H routines remain pure constitutive calculations. Their
computed exterior values are preserved during internal halo exchange. The
FIDO constraints apply even with normal shift, where auxiliary tangential E
can be nonzero. Projection can change magnetic divergence and Gauss residuals
for incompatible fields; it does not model conductor surface charges or
currents.

The electrostatic solver extends potential constantly through conducting
exterior guards before taking its gradient.

Coordinate-stretched PML modifies the tile-local spatial derivatives before
the Ampere and Faraday curls are assembled. PML is supported only by
``electrodynamic_yee``. See :doc:`usage` for the TOML form.

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
