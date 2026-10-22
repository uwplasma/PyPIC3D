import jax.numpy as jnp

from PyPIC3D.utilities.grids import grid_axis_width


def vth_to_T(vth, m, kb):
    """
    Convert thermal velocity to temperature.

    Args:
        vth (float): Thermal velocity.
        m (float): Mass of the particle.
        kb (float): Boltzmann constant.

    Returns:
        float: Temperature.
    """
    return m * vth**2 / (kb)


def T_to_vth(T, m, kb):
    """
    Convert temperature to thermal velocity.

    Args:
        T (float): Temperature.
        m (float): Mass of the particle.
        kb (float): Boltzmann constant.

    Returns:
        float: Thermal velocity.
    """
    return jnp.sqrt(kb * T / m)


def check_stability(plasma_parameters, dt):
    """
    Check the stability of the simulation based on various physical parameters.

    Args:
        plasma_parameters (dict): A dictionary containing various plasma parameters.
        dt (float): Time step of the simulation.
    """
    theoretical_freq = plasma_parameters["Theoretical Plasma Frequency"]
    debye = plasma_parameters["Debye Length"]
    thermal_velocity = plasma_parameters["Thermal Velocity"]
    num_electrons = plasma_parameters["Number of Electrons"]
    dxperDebye = plasma_parameters["dx per debye length"]

    if theoretical_freq * dt > 2.0:
        print(f"# of Electrons is Low and may introduce numerical stability")

    if dxperDebye < 1:
        print(f"Debye Length is less than the spatial resolution, this may introduce numerical instability")

    print(f"Theoretical Plasma Frequency: {theoretical_freq} Hz")
    print(f"Debye Length: {debye} m")
    print(f"Thermal Velocity: {thermal_velocity}")
    print(f'Dx Per Debye Length: {dxperDebye}')
    print(f"Number of Electrons: {num_electrons}\n")


def build_plasma_parameters_dict(static_parameters, dynamic_parameters, electrons):
    """
    Build a dictionary containing various plasma parameters.

    Args:
        static_parameters (dict): Compile-time run settings.
        dynamic_parameters (dict): Scalar values and grids.
        electrons (dict): Metadata for the electron species from particle initialization.

    Returns:
        dict: A dictionary containing the plasma parameters.
    """

    me = electrons["mass"]
    Te = electrons["temperature"]
    N = electrons["N_particles"]
    q = electrons["charge"]
    weight = electrons["weight"]
    kb = dynamic_parameters.kb
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

    volume = (
        grid_axis_width(dynamic_parameters.grids.center[0])
        * grid_axis_width(dynamic_parameters.grids.center[1])
        * grid_axis_width(dynamic_parameters.grids.center[2])
    )
    density = weight * N / volume
    eps = dynamic_parameters.eps
    if getattr(dynamic_parameters, "bssn_parameters", None) is not None:
        eps = 1.0 / (4.0 * jnp.pi)
    theoretical_freq = jnp.sqrt(density) * jnp.abs(q) / jnp.sqrt(eps * me)
    debye = jnp.sqrt(eps * kb * Te / (density * q**2))
    thermal_velocity = jnp.sqrt(3*kb*Te/me)

    plasma_parameters = {
        "Theoretical Plasma Frequency": theoretical_freq,
        "Debye Length": debye,
        "Thermal Velocity": thermal_velocity,
        "Number of Electrons": N,
        "Temperature of Electrons": Te,
        "dx per debye length": debye/dx,
        "dy per debye length": debye/dy,
        "dz per debye length": debye/dz,
    }

    return plasma_parameters
