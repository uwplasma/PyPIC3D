Demos
=====

PyPIC3D includes runnable demos in ``demos/`` that cover common workflows.
Run commands below from the repository root.

Two-Stream Instability
----------------------

.. code-block:: bash

    PyPIC3D --config demos/two_stream/two_stream.toml

Weibel Instability
------------------

.. code-block:: bash

    PyPIC3D --config demos/weibel/weibel.toml

Orszag-Tang Vortex
------------------

.. code-block:: bash

    cd demos/ot_vortex
    python initial_conditions.py
    PyPIC3D --config orszag_tang.toml

Harris Sheet Reconnection
-------------------------

.. code-block:: bash

    cd demos/reconnection_2d
    python initial_conditions.py
    PyPIC3D --config harris_current.toml

Convergence Scripts
-------------------

.. code-block:: bash

    python demos/convergence_testing/interpolation_convergence_test.py
    python demos/convergence_testing/solver_convergence_test.py

Notes
-----

- Some demos read external ``.npy`` initial conditions.
- Output files are written to each demo's configured ``output_dir``.
