Current Deposition
==================

PyPIC3D offers multiple current deposition strategies selected via
``simulation_parameters.current_calculation`` in the TOML configuration. The
available methods are:

Esirkepov (charge conserving)
-----------------------------
The Esirkepov method deposits current with exact charge conservation for the
active grid dimensions and supports both first- and second-order shape factors.
It is the recommended charge-conserving option for electrodynamic simulations.

Charge Density Method (J from ρv)
---------------------------------
The ``j_from_rhov`` option computes current directly from the particle velocity
field and charge density:

.. math::
    \mathbf{J} = \rho \mathbf{v}

This approach is straightforward and often sufficient for exploratory runs,
but it does not enforce discrete charge conservation in the same way as the
Esirkepov method. A digital filter (``constants.alpha``) is applied to smooth
high-frequency noise in the deposited current.













References
----------

For more details on the Esirkepov method, refer to the original paper:

Esirkepov, T. Z. (2001). Exact charge conservation scheme for particle-in-cell simulation with an arbitrary form-factor. Computer Physics Communications, 135(2), 144-153.
