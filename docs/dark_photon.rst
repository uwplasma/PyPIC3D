Dark Photon Sector
==================

``solver = "dark_matter_yee"`` evolves a massive, kinetically mixed vector
field alongside the ordinary Maxwell fields. The dark sector is a full
Maxwell--Proca system in its own right: it has its own potentials, its own
boundary treatment, its own PML histories, and its own energy diagnostic. It is
coupled to the visible sector only through the plasma current and the Lorentz
force.

Physical picture
----------------

The model is a hidden ``U(1)`` gauge field :math:`A'^\mu` that mixes
kinetically with ordinary electromagnetism. In the kinetically mixed basis the
Lagrangian is

.. math::

   \mathcal L = -\tfrac{1}{4}F_{\mu\nu}F^{\mu\nu}
     -\tfrac{1}{4}F'_{\mu\nu}F'^{\mu\nu}
     -\tfrac{\varepsilon}{2}F_{\mu\nu}F'^{\mu\nu}
     +\tfrac{1}{2}m^{2}A'_\mu A'^\mu
     +J^\mu A_\mu .

PyPIC3D solves this in the **mass eigenbasis**, reached by the shift
:math:`A=\tilde A-\varepsilon\tilde A'`, :math:`A'=\tilde A'`. That
diagonalizes both the kinetic and the mass terms, leaving :math:`\tilde A`
massless and :math:`\tilde A'` a pure Proca field of mass :math:`m`, and moves
the entire mixing into the matter coupling:

.. math::

   \mathcal L = -\tfrac{1}{4}\tilde F^{2}-\tfrac{1}{4}\tilde F'^{2}
     +\tfrac{1}{2}m^{2}\tilde A'^{2}
     + J^\mu\left(\tilde A_\mu-\varepsilon\tilde A'_\mu\right).

Three consequences follow, and they are exactly what the solver implements:

#. **Ordinary Maxwell evolution is untouched.** The kinetic terms are
   diagonal, so :math:`\partial_\mu\tilde F^{\mu\nu}=J^\nu` with no dark
   back-reaction in the visible Ampere law.
#. **The dark field is sourced directly by the plasma current**, with strength
   :math:`\varepsilon`:
   :math:`\partial_\mu\tilde F'^{\mu\nu}+m^{2}\tilde A'^\nu=-\varepsilon J^\nu`.
#. **Matter couples to the combination** :math:`\tilde A-\varepsilon\tilde A'`,
   so particles feel :math:`\mathbf E-\varepsilon\mathbf E'` and
   :math:`\mathbf B-\varepsilon\mathbf B'`.

``sin_chi`` is this :math:`\varepsilon`. Because a charge both sources the dark
field and responds to it with the same coefficient, every observable dark
effect enters at :math:`\mathcal O(\varepsilon^{2})` --- a static like-charge
pair picks up a repulsive Yukawa correction of relative size
:math:`\varepsilon^{2}` at separations below :math:`1/m`. Setting
``sin_chi = 0`` reproduces ``electrodynamic_yee`` bit for bit, with the dark
sector evolving as a decoupled spectator.

.. note::

   The alternative *interaction basis*, reached by the shift
   :math:`A'=\tilde A'-\varepsilon\tilde A`, puts the mixing in an off-diagonal
   **mass** matrix instead. There the dark field carries no direct current
   coupling and is populated only by oscillation through the
   :math:`\varepsilon m^{2}\tilde A'\!\cdot\!\tilde A` term, while the visible
   Ampere law picks up a compensating source. That is not what this solver
   does: the presence of a ``J`` term in the dark Ampere law, together with an
   unmodified visible sector, is the signature of the mass eigenbasis.

Two configuration values control the sector:

``sin_chi``
   The dimensionless kinetic-mixing coefficient :math:`\varepsilon`, written
   :math:`s` in the equations below. Default ``0.0``; it must be finite with
   :math:`|s|\le 1`. It is the *sine* of the mixing angle, already taken ---
   not the angle itself.

``dark_mu``
   The Proca mass :math:`m`, expressed as an **inverse length** (the Compton
   wavenumber :math:`m_{A'}c/\hbar`). Default ``0.0``; it must be finite and
   nonnegative. It is unrelated to the electromagnetic permeability ``mu``,
   which is a separate configuration value.

A nonzero mass gives the dispersion relation

.. math::

   \omega = c\sqrt{k^{2}+m^{2}},

so the dark sector supports a longitudinal mode that ordinary electromagnetism
does not. The longitudinal electric amplitude scales as :math:`c^{2}m^{2}/\omega`.

Field equations
---------------

The evolved state is :math:`(\mathbf E',\mathbf A',\phi')`. The magnetic field
is never stored; it is recomputed as the curl of the vector potential. With
:math:`s=` ``sin_chi`` and :math:`m=` ``dark_mu``:

.. math::

   \partial_t\mathbf E' &=
     c^{2}\left(\nabla\times\mathbf B'+m^{2}\mathbf A'\right)
     +\frac{s}{\epsilon_0}\mathbf J, \\
   \partial_t\mathbf A' &= -\left(\mathbf E'+\nabla\phi'\right), \\
   \partial_t\phi' &= -c^{2}\,\nabla\cdot\mathbf A', \\
   \mathbf B' &= \nabla\times\mathbf A'.

:math:`\mathbf J` is the ordinary particle current deposited during the same
step, by either ``esirkepov`` or ``j_from_rhov``.

The scalar equation is the Lorenz--Proca gauge condition promoted to an
evolution equation, which is what makes the massive system well posed without a
separate gauge fixing step. For :math:`m=0` the dark sector reduces to ordinary
electromagnetism in Lorenz gauge.

Gauss constraint
~~~~~~~~~~~~~~~~

The scheme preserves

.. math::

   \nabla\cdot\mathbf E'+m^{2}\phi' = -\frac{s\,\rho}{\epsilon_0}

to round-off, as a consequence of the divergence of the :math:`\mathbf E'`
update combined with the :math:`\phi'` equation and charge continuity.

Preservation is not enforcement. ``initialize_dark_photon_fields`` **does not
solve this constraint**: initial data supplied through ``[dark_fieldN]`` blocks
must already satisfy it, or the violation persists for the whole run. Supplying
only a vector potential (the usual case) is safe, because that leaves
:math:`\mathbf E'` and :math:`\phi'` at zero with no charge present.

Staggering and update order
---------------------------

:math:`\mathbf E'` and :math:`\phi'` live at integer time :math:`n`;
:math:`\mathbf A'` lives at :math:`n-1/2`. :math:`\mathbf A'` and
:math:`\mathbf E'` share componentwise electric-edge locations, and
:math:`\phi'` lives at ``(C,C,C)`` nodes. The forward scalar gradient and the
backward vector divergence form a compatible operator pair.

One timestep advances, in order:

1. Gather the integer-time force fields and push particles.
2. Deposit :math:`\mathbf J` from the ordinary particle motion.
3. Update the Maxwell sector with the standard three-stage Yee sequence.
4. Drift :math:`\mathbf A'` from :math:`n-1/2` to :math:`n+1/2`.
5. Form :math:`\mathbf B'=\nabla\times\mathbf A'` at :math:`n+1/2`.
6. Kick :math:`\phi'` and then :math:`\mathbf E'` from :math:`n` to
   :math:`n+1`.

For the integer-time particle push and for output, ``synchronized_dark_fields``
reconstructs :math:`\mathbf A'` and :math:`\mathbf B'` at integer time by a half
drift --- equivalently, the average of the adjacent half-step potentials. It
does not modify the live state. The electric gather uses the same filter as
direct-current deposition.

Timestep
--------

``dark_matter_yee`` **overrides the ordinary Courant path.** The Proca
frequency is bounded by the mass as well as the grid, so stability requires

.. math::

   \Delta t < \frac{2}{c\sqrt{m^{2}+4\sum_{\text{axes}}d^{-2}}},

where the sum runs over resolved axes only. When ``dt`` is omitted it is taken
as the smaller of the ordinary Maxwell CFL step and ``0.99 * cfl`` times this
limit. An explicit ``dt`` at or above the limit is rejected. A massless dark
field with every dimension collapsed has no limit, so ``dt`` must then be given
explicitly.

Initial data
------------

Select the solver and its static parameters under ``[simulation_parameters]``.
Ordinary ``fieldN`` blocks keep their existing meaning. Optional
``dark_fieldN`` blocks initialize the dark sector at ``t=0``:

.. code-block:: toml

   [simulation_parameters]
   solver = "dark_matter_yee"
   sin_chi = 0.01
   dark_mu = 0.7  # inverse length in the configured units

   [dark_field1]
   name = "initial dark Ay"
   type = 4
   path = "dark_Ay.npy"

The component mapping is:

- ``0``, ``1``, ``2``: ``E'x``, ``E'y``, ``E'z``
- ``3``, ``4``, ``5``: ``A'x``, ``A'y``, ``A'z``
- ``6``: ``phi'``

Each file contains a finite real array shaped ``(Nx, Ny, Nz)``, sampled at the
component's Yee locations. Unspecified components are zero; repeated blocks add
to the selected component. All dark fields evolve, so ``evolve = false`` is
rejected. ``dark_fieldN`` blocks with any other solver are an error. ``C`` and
``eps`` follow the existing unit configuration. The input ``A`` is
automatically shifted to ``t=-dt/2``; do not pre-stagger the input files.

Python callers can use
``PyPIC3D.solvers.dark_matter_yee.dark_photon_fields.initialize_dark_photon_fields``
with ``(static_parameters, dynamic_parameters, E=None, A=None, phi=None)``.
The optional inputs are full floating-point tiled arrays at ``t=0``; vector
inputs contain three components. The helper refreshes halos and returns the
staggered ``(E, A, phi)`` state.

Boundaries
----------

The dark sector uses the same ``x_bc`` / ``y_bc``/ ``z_bc`` wall settings as the
Maxwell sector, with values ``periodic`` or ``conducting``, and maintains
independent histories.

Conducting walls impose homogeneous reflecting Proca conditions: zero scalar
potential, tangential vector potential and electric field, and normal magnetic
field. Across each physical wall, ``phi`` and tangential ``A`` / ``E`` have odd
parity, normal ``A`` / ``E`` have even parity, and ``B`` has the opposite vector
parities. Reflections use the actual Yee endpoints, including edges and
corners.

The existing ``[[pml]]`` wall configuration also absorbs transverse and
longitudinal dark waves. The dark solver stretches both curls, the scalar
gradient, and the vector divergence, so the massive longitudinal mode is
absorbed too. Its auxiliary state stores a magnetic field at half time and 18
derivative histories, separate from the Maxwell PML histories. With PML active,
``pml_state`` is the pair ``(maxwell_pml, dark_pml)``.

Diagnostics and output
----------------------

The Proca field energy is

.. math::

   U' = \tfrac{1}{2}\epsilon\int\left[
     |\mathbf E'|^{2}+c^{2}|\mathbf B'|^{2}
     +c^{2}m^{2}|\mathbf A'|^{2}+m^{2}\phi'^{2}\right]dV,

evaluated at integer time from the reconstructed :math:`\mathbf A'` and
:math:`\mathbf B'`, with halos excluded. It is second-order accurate in time
rather than an exactly conserved discrete leapfrog Hamiltonian, so it is not
expected to be constant to round-off.

Runs write:

- ``data/dark_field_energy.txt``, alongside the ordinary scalar energy traces,
  and a ``Dark Field Energy`` line in the printed energy report.
- openPMD meshes ``dark_E``, ``dark_A``, ``dark_B`` and ``dark_phi`` when
  ``plot_openpmd_fields = true``, at integer time.

Runtime state
-------------

The dark solver preserves the ordinary field-tuple prefix and inserts its state
before the overflow flag:

.. code-block:: text

   (E, B, J, rho, phi, external_fields, pml_state, dark_fields, overflow)

``dark_fields`` is ``(E_prime, A_prime, phi_prime)``. See :doc:`architecture`
for the shared parts of the tuple.

Demos
-----

See :doc:`demos` for the two dark-sector examples: a warm Maxwell--Proca
two-stream instability, and an electron-positron Harris sheet whose
reconnecting field is supplied entirely through the mixing.
