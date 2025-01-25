Usage
=====

This section explains how to use the `PyPIC3D` code.

Initialization
--------------

To initialize the simulation, use the `initialize_simulation` function:

.. code-block:: python

    from PyPIC3D.initialization import initialize_simulation

    config_file = 'path/to/config.toml'
    simulation_data = initialize_simulation(config_file)

For more details, refer to the function documentation in the code.