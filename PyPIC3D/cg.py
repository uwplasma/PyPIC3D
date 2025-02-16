import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
from jax import lax
# from jax._src.scipy.sparse.linalg import _vdot_real_tree, _add, _sub, _mul
# from jax.tree_util import tree_leaves
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import functools
from functools import partial
# import external libraries

def identity(x):
    """
    Return the input value unchanged.

    Args:
        x (any type): The input value to be returned.

    Returns:
        any type: The same value that was passed as input.
    """
    return x

def dot(A, B):
    """
    Compute the dot product of two 3D arrays using Einstein summation convention.

    Args:
        A (array-like): First input array with shape (i, j, k).
        B (array-like): Second input array with shape (i, j, k).

    Returns:
        float: The dot product of the input arrays.
    """
    return jnp.einsum('ijk,ijk->', A, B)

def conjugated_gradients(A, b, x0, tol=1e-6, maxiter=1000, M=identity):
    """
    Solve the linear system Ax = b using the Conjugate Gradient method.

    Args:
        A (function): A function that computes the matrix-vector product Ax.
        b (array-like): The right-hand side vector of the linear system.
        x0 (array-like): The initial guess for the solution.
        tol (float, optional): The tolerance for the stopping criterion. Default is 1e-6.
        maxiter (int, optional): The maximum number of iterations. Default is 1000.
        M (function, optional): A function that applies the preconditioner. Default is the identity function.

    Returns:
        array-like: The approximate solution to the linear system.
    """
    g = A(x0) - b
    # compute the residual
    s = M(g)
    d = -s
    z = A(d)
    alpha = dot(g, s)
    beta  = dot(d, z)
    initial_value = x0, d, g, alpha, beta, 0
    # compute the initial parameters

    def body_func(value):
        x, d, g, alpha, beta, i = value
        z = A(d)
        x = x + (alpha/beta)*d
        g = g + (alpha/beta)*z
        # update using the scalar parameters
        s = M(g)
        # apply the preconditioning matrix
        beta = alpha
        alpha = dot(g, s)
        # update the scalars
        d = (alpha/beta)*d - s
        i = i + 1
        # update the counter
        value = x, d, g, alpha, beta, i
        # store the updated values
        return value

    def cond_fun(value):
        x, d, g, alpha, beta, i = value
        return (jnp.linalg.norm(g) > tol) & (i < maxiter)

    x_final, *_ = lax.while_loop(cond_fun, body_func, initial_value)

    return x_final