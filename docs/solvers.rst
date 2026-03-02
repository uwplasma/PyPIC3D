Field Solvers
=============

PyPIC3D supports two production runtime modes selected from
``simulation_parameters``.

Runtime Modes
-------------

Electrostatic mode
^^^^^^^^^^^^^^^^^^

Set:

.. code-block:: toml

    electrostatic = true

Per step, PyPIC3D:

1. Pushes particles.
2. Deposits ``rho``.
3. Solves Poisson's equation for ``phi``.
4. Computes ``E = -grad(phi)``.

The ``solver`` key controls both the Poisson solve and gradient operator:

- ``solver = "spectral"``: FFT Poisson + spectral gradient
- ``solver = "fdtd"`` (or any non-``"spectral"`` value): conjugate-gradient Poisson + centered finite-difference gradient

Electrodynamic mode
^^^^^^^^^^^^^^^^^^^

Set:

.. code-block:: toml

    electrostatic = false

Per step, PyPIC3D:

1. Pushes particles.
2. Deposits current ``J``.
3. Updates ``E``.
4. Updates ``B``.

The electrodynamic update uses first-order Yee-style kernels in
``PyPIC3D.solvers.first_order_yee``.

Vector potential mode
^^^^^^^^^^^^^^^^^^^^^

``solver = "vector_potential"`` currently raises ``NotImplementedError`` during
initialization.

Electrodynamic Update Equations
-------------------------------

.. math::

    \mathbf{E}^{n+1} = \mathbf{E}^{n} + \Delta t
    \left(c^2 \nabla \times \mathbf{B}^{n} - \frac{\mathbf{J}^{n}}{\epsilon_0}\right)

.. math::

    \mathbf{B}^{n+1} = \mathbf{B}^{n} - \Delta t \left(\nabla \times \mathbf{E}^{n+1}\right)

PyPIC3D additionally applies a digital filter controlled by
``constants.alpha`` to field components each update.

Boundary Conditions
-------------------

Field boundary conditions are defined by:

- ``simulation_parameters.x_bc``
- ``simulation_parameters.y_bc``
- ``simulation_parameters.z_bc``

Supported values:

- ``periodic``
- ``conducting``

For conducting boundaries, tangential electric-field components are zeroed on
boundary faces during the ``E`` update.

Current Deposition Selection
----------------------------

Current deposition is selected by:

.. code-block:: toml

    current_calculation = "j_from_rhov"

or

.. code-block:: toml

    current_calculation = "esirkepov"

See :doc:`chargeconservation` for behavior and tradeoffs.
