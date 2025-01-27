import epyc
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.evolve import time_loop
from PyPIC3D.plotting import plotter
from functools import partial
import time
from tqdm import tqdm
import toml
import argparse
import os
from jax import jit


class PyPIC3DExperiment(epyc.Experiment):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        # Initialize the simulation
        particles, Ex, Ey, Ez, Ex_ext, Ey_ext, Ez_ext, Bx, By, Bz, Bx_ext, By_ext, Bz_ext, Jx, Jy, Jz, \
            phi, rho, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, \
            plasma_parameters, M, solver, bc, electrostatic, verbose, GPUs, start, Nt, curl_func, \
            pecs, lasers, surfaces = initialize_simulation(self.config)

        loop = partial(time_loop, Ex_ext=Ex_ext, Ey_ext=Ey_ext, Ez_ext=Ez_ext, Bx_ext=Bx_ext, By_ext=By_ext, Bz_ext=Bz_ext, E_grid=E_grid, \
            B_grid=B_grid, world=world, constants=constants, pecs=pecs, lasers=lasers, surfaces=surfaces, \
            curl_func=curl_func, M=M, solver=solver, bc=bc, electrostatic=electrostatic, verbose=verbose, GPUs=GPUs)

        loop = jit(loop, static_argnums=(0,))
        # compile the time loop function

        start = time.time()
        for t in tqdm(range(Nt)):
            particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi = loop(t, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi)
            # time loop to update the particles and fields
            plotter(t, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, simulation_parameters, solver, bc)
            # plot the data

        end = time.time()
        duration = end - start

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