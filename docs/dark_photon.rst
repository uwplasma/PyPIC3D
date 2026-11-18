Dark Photon Sector
==================

``solver = "dark_matter_yee"`` evolves a massive, kinetically mixed vector
field alongside the ordinary Maxwell fields. The model is a hidden ``U(1)`` 
gauge field :math:`A'^\mu` that mixes kinetically with ordinary electromagnetism. [1]_
In the kinetically mixed basis the Lagrangian is

.. math::

   \mathcal{L} ={}&
   -\frac{1}{4} F_{\alpha\beta} F^{\alpha\beta}
   -\frac{1}{4} F'_{\alpha\beta} F'^{\alpha\beta}
   -\frac{\mu^2}{2} A'_{\alpha} A'^{\alpha} \\
   &+\frac{\sin\chi_0}{2} F'_{\alpha\beta} F^{\alpha\beta}
   -J^{\alpha} A_{\alpha}.


PyPIC3D solves this in the **mass eigenbasis**, reached by the shift
:math:`A=\tilde A-\sin\chi_0\tilde A' + \mathcal{O}(\sin^2\chi_0)`, :math:`A'=\tilde A'`. [1]_ That
diagonalizes both the kinetic and the mass terms, leaving :math:`\tilde A`
massless and :math:`\tilde A'` a pure Proca field of mass :math:`m`, and moves
the entire mixing into the matter coupling:

Dark Photon Sector
==================

``solver = "dark_matter_yee"`` evolves a massive, kinetically mixed vector
field alongside the ordinary Maxwell fields. The model is a hidden ``U(1)`` 
gauge field :math:`A'^\mu` that mixes kinetically with ordinary electromagnetism. [1]_ 
In the kinetically mixed basis the Lagrangian is

.. math::

   \mathcal{L} ={}&
   -\frac{1}{4} F_{\alpha\beta} F^{\alpha\beta}
   -\frac{1}{4} F'_{\alpha\beta} F'^{\alpha\beta}
   -\frac{\mu^2}{2} A'_{\alpha} A'^{\alpha} \\
   &+\frac{\sin\chi_0}{2} F'_{\alpha\beta} F^{\alpha\beta}
   -J^{\alpha} A_{\alpha}.


PyPIC3D solves this in the **mass eigenbasis**, reached by the shift
:math:`A=\tilde A-\sin\chi_0\tilde A' + \mathcal{O}(\sin^2\chi_0)`, :math:`A'=\tilde A'`. [1]_ That
diagonalizes both the kinetic and the mass terms, leaving :math:`\tilde A`
massless and :math:`\tilde A'` a pure Proca field of mass :math:`m`, and moves
the entire mixing into the matter coupling:

.. math::

   \mathcal{L}_{\mathrm{mass}} ={}&
   -\frac{1}{4} F_{\alpha\beta} F^{\alpha\beta}
   -\frac{1}{4} F'_{\alpha\beta} F'^{\alpha\beta}
   -\frac{\mu^2}{2} A'_{\alpha} A'^{\alpha} \\
   &-J^{\alpha}\left(A_{\alpha}+\sin\chi_0\,A'_{\alpha}\right).

The evolution equations for the mass eigenbasis are the following:

.. math::

   \nabla_{\alpha} F^{\alpha\beta} = J^{\beta}.

   \nabla_{\alpha} F'^{\alpha\beta}
   = \sin\chi_0\,J^{\beta} + \mu^2 A'^{\beta}.

   u^{\beta}\nabla_{\beta}u^{\alpha}
   = \frac{e}{m_e}
   \left(F^{\alpha\beta} + \sin\chi_0\,F'^{\alpha\beta}\right)u_{\beta}.

Maxwell's equations remain unchanged, and the Proca fields act as linear source terms 
for the Lorentz force. [1]_




   \mathcal{L}_{\mathrm{mass}} ={}&
   -\frac{1}{4} F_{\alpha\beta} F^{\alpha\beta}
   -\frac{1}{4} F'_{\alpha\beta} F'^{\alpha\beta}
   -\frac{\mu^2}{2} A'_{\alpha} A'^{\alpha} \\
   &-J^{\alpha}\left(A_{\alpha}+\sin\chi_0\,A'_{\alpha}\right).

The evolution equations for the mass eigenbasis are the following:

.. math::

   \nabla_{\alpha} F^{\alpha\beta} = J^{\beta}.

   \nabla_{\alpha} F'^{\alpha\beta}
   = \sin\chi_0\,J^{\beta} + \mu^2 A'^{\beta}.

   u^{\beta}\nabla_{\beta}u^{\alpha}
   = \frac{e}{m_e}
   \left(F^{\alpha\beta} + \sin\chi_0\,F'^{\alpha\beta}\right)u_{\beta}.

Maxwell's equations remain unchanged, and the Proca fields act as linear source terms 
for the Lorentz force. [1]_

.. [1] Corelli, F., Cannizzaro, E., Caputo, A., & Pani, P. (2025). Beyond the standard model electrodynamics in the time domain. Physical Review D, 111(3), 035019.



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
limit. An explicit ``dt`` at or above the limit is rejected.

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
to the selected component. The input ``A`` is
automatically shifted to ``t=-dt/2``; do not pre-stagger the input files.


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