import epyc
import time
from tqdm import tqdm
import os
import jax
# import external libraries

from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.plotting import plotter
# import functions from the PyPIC3D package

class PyPIC3DExperiment(epyc.Experiment):
    """
    A class to represent a 3D Particle-In-Cell (PIC) experiment using the PyPIC3D framework.
    Attributes
    ----------
    config : dict
        Configuration parameters for the experiment.
    Methods
    -------
    __init__(self, config):
        Initializes the experiment with the given configuration.
    run(self):
        Runs the simulation loop, updates particles and fields, and plots the data.
        Returns a dictionary containing the duration of the simulation, final particles,
        final fields, plasma parameters, simulation parameters, constants, world, and
        plotting parameters.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):

        loop, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, \
            phi, rho, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, \
                plasma_parameters, M, solver, bc, electrostatic, verbose, GPUs, Nt, curl_func, \
                    pecs, lasers, surfaces = initialize_simulation(self.config)
        # initialize the simulation

        loop = jax.jit(loop)
        # jit the loop function

        start = time.time()
        # start the timer

        for t in tqdm(range(Nt)):
            particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi = loop(particles, (Ex, Ey, Ez), (Bx, By, Bz), (Jx, Jy, Jz), rho, phi)
            # time loop to update the particles and fields
            plotter(t, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, simulation_parameters, solver, bc)
            # plot the data

        end = time.time()
        duration = end - start
        # calculate the duration of the simulation

        return {
            'duration': duration,
            'final_particles': particles,
            'final_fields': (Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi),
            'plasma_parameters': plasma_parameters,
            'simulation_parameters': simulation_parameters,
            'constants': constants,
            'world': world,
            'plotting_parameters': plotting_parameters,
        }

class ParameterScan(epyc.Lab):
    def __init__(self, name, run_dir, base_config, section, param_name, param_values):
        """
        Initialize the experiment with the given parameters.
        Args:
            name (str): The name of the experiment.
            run_dir (str): The directory where the experiment will be run.
            base_config (dict): The base configuration for the experiment.
            section (str): The section of the configuration to modify.
            param_name (str): The name of the parameter to vary.
            param_values (list): The values of the parameter to test.
        """

        super().__init__()
        self.name = name
        self.run_dir = run_dir
        self.section = section
        self.base_config = base_config
        self.param_name = param_name
        self.param_values = param_values

    def parameters(self):
        for value in self.param_values:
            config = self.base_config.copy()
            config[self.section][self.param_name] = value
            experiment_dir = f'{self.run_dir}/{self.name}/{self.param_name}_{value}'.replace(' ', '_')

            if not os.path.exists(experiment_dir):
                print(f'Creating directory {experiment_dir}')
                os.makedirs(experiment_dir)
            # create the directory for the experiment

            config['simulation_parameters']['output_dir'] = experiment_dir
            yield config, value

    def build(self, params):
        return PyPIC3DExperiment(params)
