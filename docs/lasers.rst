Laser Pulses
=======================

PyPIC3D is a powerful tool for simulating laser pulses in three-dimensional space. The laser pulses in PyPIC3D are characterized by several parameters that define their properties and behavior. These parameters include:

- **Wavelength**: The distance between successive peaks of the laser wave, typically measured in nanometers (nm).
- **Pulse Duration**: The length of time the laser pulse lasts, usually measured in femtoseconds (fs).
- **Beam Width**: The diameter of the laser beam at its widest point, typically measured in micrometers (µm).
- **Max Electric Field**: The maximum electric field strength of the laser pulse, usually measured in volts per meter (V/m).


Formulation
---------------------

        Laser pulses are currently modeled in PyPIC3D as linearly polarized plane waves. The electric field \( E \) of the laser pulse is given by the following equation:

        .. math::

            E(x, y, z, t) = A \cdot \sin({k_0} \cdot x - {\omega_0} \cdot t) \cdot  \exp\left(-\frac{(t - \frac{\tau}{2})^2}{2 \left(\frac{\tau}{8}\right)^2}\right) \cdot \exp\left(-\frac{(y - {y_0} )^2}{2 \cdot {pw}^2}\right) \cdot \exp\left(-\frac{(z - {z_0})^2}{2 \cdot {pw}^2}\right)  \hat{y}

        where:

        - \( A \) is the maximum electric field strength of the laser pulse.
        - \( k_0 \) is the wave number of the laser pulse.
        - \( \omega_0 \) is the angular frequency of the laser pulse.
        - \( \tau \) is the pulse width of the laser pulse.
        - \( x \) is the x position of the laser pulse.
        - \( y \) is the y position of the laser pulse.
        - \( z \) is the z position of the laser pulse.
        - \( t \) is the time.
        - \( y_0 \) is the starting y position of the laser pulse.
        - \( z_0 \) is the starting z position of the laser pulse.
        - \( pw \) is the width of the laser pulse.

        The electric field of the laser pulse is a function of the x, y, z positions and time \( t \). The laser pulse is a sinusoidal wave that is modulated by a Gaussian envelope in the time domain and a Gaussian beam profile in the y and z directions.  


        The magnetic field \( B \) of the laser pulse is given by the following equation:

        .. math::

            B(x, y, z, t) = \frac{E(x, y, z, t)}{c}  \hat{z}

        where:

        - \( E(x, y, z, t) \) is the electric field of the laser pulse.
        - \( c \) is the speed of light in a vacuum.


        The magnetic field of the laser pulse is proportional to the electric field and is in the direction of propagation of the laser pulse.


        Laser Pulse Visualization
        -------------------------

        .. image:: images/laser_tungsten.gif
            :alt: Laser pulse animation
            :width: 600
            :height: 400
            :align: center

        The above animation shows a laser pulse propagating through space towards a bound particle species. The electric field oscillates sinusoidally while being modulated by a Gaussian envelope in both time and space.

Example Configuration
---------------------

Here is an example of how to configure a laser pulse in PyPIC3D:

.. code-block:: python

    [laser1]
    max_electric_field = 1000000000000
    k0 = 7853981.633974483
    omega0 = 2.36e15
    pulse_width = 20e-12
    xstart = 0.05e-1
    ystart = 0
    zstart = 0
    width  = 0.15e-1

To incorporate a laser pulse in a simulation, the following parameters must be set for the laser pulse in the configuration file:

- `max_electric_field`: The maximum electric field strength of the laser pulse.
- `k0`: The wave number of the laser pulse.
- `omega0`: The angular frequency of the laser pulse.
- `pulse_width`: The duration of the laser pulse.
- `xstart`: The starting x position of the laser pulse.
- `ystart`: The starting y position of the laser pulse.
- `zstart`: The starting z position of the laser pulse.
- `width`: The width of the laser pulse.