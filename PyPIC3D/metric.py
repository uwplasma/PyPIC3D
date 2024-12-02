# Christopher Woolford, Nov 25 2024.
# This code contains the metric class for PyPIC3D.
# Ideally, this class should handle how contractions with other tensors are done.
# This class should also handle the computation of the inverse metric tensor and the determinant of the metric tensor.
# An instance of this class will be an input for vector calculus operations in the code to generalize the operations to general coordinate systems.

import jax.numpy as jnp
import jax.jit as jit