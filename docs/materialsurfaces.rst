Material Surfaces
=================

Material boundaries in particle in cell simulations typically used spatially varying
permittivities to model the interaction of particles with different materials. In PyPIC3D,
material surfaces can be defined using the `MaterialSurface` class. This class allows users
to model a surface such as metal by simulating a plasma with a surface boundary condition
based on the work function of the material. This allows for the simulation of materials with
electron spill over.



Usage
-----

To define a material surface in PyPIC3D, you must create a particle species within the region
of the material and define the surface of the material using the `MaterialSurface` class.

Here is an example of how you might define a material surface in the configuration file:

.. code-block:: toml

    [particle1]
    name = "electrons"
    N_particles = 30000
    weight = 100
    charge = -1.602e-19
    mass   = 9.1093837e-31
    temperature = 1
    xmin = -0.025
    xmax =  0.025
    ymin = -0.025
    ymax =  0.025
    zmin = -0.025
    zmax =  0.025

    [particle2]
    name = "ions"
    N_particles = 30000
    weight = 100
    charge = 1.602e-19
    mass   = 1.67e-27
    temperature = 1
    xmin = -0.025
    xmax =  0.025
    ymin = -0.025
    ymax =  0.025
    zmin = -0.025
    zmax =  0.025

    [surface1]
    name='tungsten face'
    material='tungsten'
    work_function_x='work_function_x.npy'
    work_function_y='work_function_y.npy'
    work_function_z='work_function_z.npy'