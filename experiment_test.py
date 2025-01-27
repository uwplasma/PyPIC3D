import argparse
import toml
import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import the ParameterScan class from the experiment module
from PyPIC3D.experiment import ParameterScan


# Load the base configuration from a TOML file
parser = argparse.ArgumentParser(description="3D PIC code using Jax")
parser.add_argument('--config', type=str, default="config.toml", help='Path to the configuration file')
args = parser.parse_args()
# argument parser for the configuration file
config_file = args.config
# path to the configuration file
base_config = toml.load(config_file)

# Define the parameter scan
param_scan = ParameterScan(
    name="example_experiment",
    run_dir="runs",
    base_config=base_config,
    section="simulation_parameters",
    param_name="t_wind",
    param_values=[10e-12, 20e-12, 30e-12]
)

parameters = param_scan.parameters()
durations = []

# Run the experiment
for parameter in parameters:
    experiment = param_scan.build(parameter)
    results = experiment.run()
    # get the results of the experiment
    durations.append(results["duration"])
    # log the time taken for each experiment

with open("durations.txt", "w") as f:
    for duration in durations:
        f.write(f"{duration}\n")