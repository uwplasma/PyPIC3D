Demos
============================

This section provides a list of demos that showcase the capabilities of the
PyPIC3D code. Each demo includes a brief description and instructions on how
to run it.

Two Stream Instability
----------------------

The two stream instability demo showcases the phenomenon where two streams of charged particles interact and create instabilities. This is a common occurrence in plasma physics and is important for understanding various space and laboratory plasmas.

In this demo, we simulate two counter-streaming electron beams and observe the growth of instabilities over time. The simulation uses a Particle-In-Cell (PIC) method to model the behavior of the particles and the resulting electric fields.

Key Features:

    Simulation of two counter-streaming electron beams.

    Visualization of particle distribution and electric field evolution.

    Analysis of instability growth rates.

To run the demo, do the following:

.. code-block:: bash

    # Navigate to the two_stream demo directory
    cd demos/two_stream
    # Run the main simulation with the configuration file
    PyPIC3D --config two_stream.toml

Weibel Instability
------------------

The Weibel demo simulates magnetic field generation from anisotropic velocity
distributions. It is useful for validating current deposition and field
updates in quasi-1D or 2D setups.

.. code-block:: bash

    cd demos/weibel
    PyPIC3D --config weibel.toml

Orszag-Tang Vortex
------------------

The Orszag-Tang vortex is a standard MHD-inspired test for vortex dynamics and
turbulence onset. The demo initializes the fields from an external script and
then runs the evolution.

.. code-block:: bash

    cd demos/ot_vortex
    python initial_conditions.py
    PyPIC3D --config orszag_tang.toml

Harris Sheet Reconnection
-------------------------

This demo sets up a 2D Harris current sheet to study magnetic reconnection and
particle acceleration.

.. code-block:: bash

    cd demos/reconnection_2d
    python initial_conditions.py
    PyPIC3D --config harris_current.toml

Convergence Tests
-----------------

The ``demos/convergence_testing`` directory contains scripts for checking
interpolation and solver convergence:

.. code-block:: bash

    python demos/convergence_testing/interpolation_convergence_test.py
    python demos/convergence_testing/solver_convergence_test.py
