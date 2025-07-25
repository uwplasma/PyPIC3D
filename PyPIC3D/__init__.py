import time
import jax
from jax import random
import jax.numpy as jnp
import os, sys
import matplotlib.pyplot as plt
# import external libraries

jax.config.update('jax_platform_name', 'cpu')

from . import errors
from . import boundaryconditions
from . import initialization
from . import particle
from . import plotting
from . import utils
from .solvers import pstd
from .solvers import fdtd
from . import boris
from . import rho
from . import evolve
from .solvers import vector_potential
from .solvers import curl_curl_form