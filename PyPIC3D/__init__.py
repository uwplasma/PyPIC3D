import time
import jax
from jax import random
import jax.numpy as jnp
import os, sys
import matplotlib.pyplot as plt
# import external libraries

#jax.config.update('jax_platform_name', 'cpu')

from . import cg
from . import errors
from . import fields
from . import boundaryconditions
from . import initialization
from . import particle
from . import plotting
from . import utils
from . import pstd
from . import fdtd
from . import sor
from . import boris
from . import rho
from . import evolve
from . import pec
from . import laser