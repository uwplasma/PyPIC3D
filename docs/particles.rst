Particle Species
================

PyPIC3D represents each particle population as a separate species with its own
charge, mass, temperature, and initialization bounds. This structure supports
multi-species plasmas (electrons, ions, beams, etc.) and per-species boundary
conditions.

Defining a Species
------------------

Define each species as a ``[particleX]`` section in the TOML configuration:

.. code-block:: toml

    [particle1]
    name = "electrons"
    N_particles = 30000
    weight = 100
    charge = -1.602e-19
    mass   = 9.1093837e-31
    temperature = 293000

Required fields
^^^^^^^^^^^^^^^

- ``name``: Human-readable species label.
- ``charge``: Particle charge in Coulombs.
- ``mass``: Particle mass in kilograms.
- ``N_particles`` or ``N_per_cell``: Total macro-particles or number per cell.

Common optional fields
^^^^^^^^^^^^^^^^^^^^^^

- ``temperature`` or ``vth``: Set thermal conditions (scalar or via ``Tx``, ``Ty``, ``Tz``).
- ``weight``: Macro-particle weight; if omitted, can be inferred from ``ds_per_debye``.
- ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``: Spatial bounds.
- ``x_bc``, ``y_bc``, ``z_bc``: Per-species boundary conditions (``periodic`` or ``reflecting``).
- ``initial_x``, ``initial_y``, ``initial_z``: ``.npy`` files for initial positions.
- ``initial_vx``, ``initial_vy``, ``initial_vz``: ``.npy`` files for initial velocities.
- ``update_pos``/``update_v`` and per-component flags (``update_x``, ``update_vx``, etc.).

Initialization Notes
--------------------

If no external ``initial_*`` arrays are provided, positions are initialized with
a uniform distribution inside the specified bounds:

.. code-block:: python

    x = jax.random.uniform(key1, shape=(N_particles,), minval=xmin, maxval=xmax)
    y = jax.random.uniform(key2, shape=(N_particles,), minval=ymin, maxval=ymax)
    z = jax.random.uniform(key3, shape=(N_particles,), minval=zmin, maxval=zmax)
    # Generate random numbers from a uniform distribution
    # to initialize the position of the particles

Velocities are initialized using a Maxwell-Boltzmann distribution based on the
specified temperature or thermal velocity:

.. code-block:: python

    std = jnp.sqrt(vth)
    v_x = np.random.normal(0, std, N_particles)
    v_y = np.random.normal(0, std, N_particles)
    v_z = np.random.normal(0, std, N_particles)
    # initialize the particles such that the magnitude of the velocity follows a maxwell boltzmann distribution





.. Bounded Particles
.. =================

.. Bounded Particles can be used to simulate linearly dispersive materials more accurately than a bulk permittivity model. A damped harmonic
.. oscillator is used to model bounded particles such as semimetals by simulating a free electron gas with a uniform background
.. of ions. This is done by added additional force terms to the Boris Push algorithm.

.. Incorporating Bounded Particles in a Simulation
.. ------------------------------------------------

.. To incorporate bounded particles in a simulation, the following parameters must be set for a particle species in the configuration file:

.. - `name`: The name of the particle species.
.. - `N_particles`: The number of particles in the species.
.. - `weight`: The weight of the particles.
.. - `charge`: The charge of the particles.
.. - `mass`: The mass of the particles.
.. - `temperature`: The temperature of the particles.
.. - `bounded = true`: This parameter must be set to true to enable bounded particles.
.. - `fermi_energy`: The Fermi energy of the bounded particles.
.. - `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`: The minimum and maximum values of the bounded particles in the x, y, and z directions.
.. - `w`: The spring constant for the bounded particles in the x, y, and z directions.
.. - `g`: The damping factor for the bounded particles in the x, y, and z directions.

.. Optional Parameters:

.. - `initial_x`: Path to *.npy file containing the initial x position of the particles.
.. - `initial_y`: Path to *.npy file containing the initial y position of the particles.
.. - `initial_z`: Path to *.npy file containing the initial z position of the particles.
.. - `initial_vx`: Path to *.npy file containing the initial x velocity of the particles.
.. - `initial_vy`: Path to *.npy file containing the initial y velocity of the particles.
.. - `initial_vz`: Path to *.npy file containing the initial z velocity of the particles.
.. - `update_pos`: Whether to update the position of the particles.
.. - `update_v`: Whether to update the velocity of the particles.
.. - `update_vx`: Whether to update the x velocity of the particles.
.. - `update_vy`: Whether to update the y velocity of the particles.
.. - `update_vz`: Whether to update the z velocity of the particles.
.. - `update_x`: Whether to update the x position of the particles.
.. - `update_y`: Whether to update the y position of the particles.
.. - `update_z`: Whether to update the z position of the particles.


.. Bounded Particle Initialization
.. -----------------------------------------

.. The velocity of the bounded particles are initialized using a Fermi-Dirac distribution.

.. The energy of the particles is sampled from a Fermi-Dirac distribution and used to calculate the magnitude of the velocity using the following:

.. .. code-block:: python

..     uniformdist = jax.random.uniform(key1, shape=(N_particles,), minval=1e-3, maxval=1)
..     # Generate random numbers from a uniform distribution
..     energy = fermi_energy - kb * T * jnp.log(uniformdist)
..     # Sample the energy of the particles from the fermi dirac distribution
..     vmag = jnp.sqrt(2*energy/mass)
..     # get the magnitude of the velocity from the fermi dirac distribution

.. The velocity components are then initialized using a Dirichlet distribution:

.. .. code-block:: python

..     random_numbers = jax.random.dirichlet(key3, jnp.ones(3))
..     # Generate three random numbers that sum to 1 from the dirichlet distribution for the velocity components
..     v_x = vmag * random_numbers[0]
..     v_y = vmag * random_numbers[1]
..     v_z = vmag * random_numbers[2]
..     # initialize the velocity components using the fermi dirac distribution and the random numbers
