Demos
============================

This section provides a list of demos that showcase the capabilities of the `PyPIC3D` code. Each demo includes a brief description and instructions on how to run it.

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
    # Generate the initial particle distribution
    python gen_data.py
    # Run the main simulation with the configuration file
    python main.py --config demos/two_stream/two_stream.toml