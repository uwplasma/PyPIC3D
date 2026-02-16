Current Deposition
==================

PyPIC3D selects current deposition via
``simulation_parameters.current_calculation``.

Available Methods
-----------------

``j_from_rhov``
^^^^^^^^^^^^^^^

Computes current directly from particle velocities and deposited charge using a
stencil deposition workflow.

Optional filtering is controlled by:

.. code-block:: toml

    filter_j = "bilinear"   # bilinear, digital, none

- ``bilinear`` applies a tri-linear smoothing filter.
- ``digital`` applies a digital filter using ``constants.alpha``.
- ``none`` leaves deposited current unfiltered.

``esirkepov``
^^^^^^^^^^^^^

Uses an Esirkepov-style charge-conserving deposition path and supports shape
factors 1 and 2.

Practical Guidance
------------------

- Start with ``j_from_rhov`` for quick exploratory runs.
- Use ``esirkepov`` when tighter discrete charge conservation is needed.
- Keep ``shape_factor`` and filtering choices fixed when comparing runs.

Reference
---------

Esirkepov, T. Z. (2001). Exact charge conservation scheme for particle-in-cell
simulation with an arbitrary form-factor. *Computer Physics Communications*,
135(2), 144-153.
