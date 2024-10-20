import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
from jax import lax
from jax._src.scipy.sparse.linalg import _vdot_real_tree, _add, _sub, _mul
from jax.tree_util import tree_leaves
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import functools
from functools import partial
# import external libraries

@jit
def apply_M(Ax, M):
    """
    Apply the preconditioner

    Parameters:
    M (ndarray): The preconditioner.
    Ax (ndarray): The laplacian of x.

    Returns:
    ndarray: The inverse laplacian of the laplacian of the data.
    """

    M_Ax = jnp.einsum('ij,jlk -> ilk', M, Ax)
    M_Ay = jnp.einsum('ij, lik -> jlk', M, Ax)
    M_Az = jnp.einsum('ij, lki -> jlk', M, Ax)

    return (1/9)*(M_Ax + M_Ay + M_Az)

def conjugate_grad(A, b, x0, tol=1e-6, atol=0.0, maxiter=10000, M=None):
    """
    Solve the linear system Ax = b using the Conjugate Gradient method.

    Parameters:
    A (callable): Function that computes the matrix-vector product Ax.
    b (array-like): Right-hand side vector.
    x0 (array-like): Initial guess for the solution.
    tol (float, optional): Tolerance for the stopping criterion. Default is 1e-6.
    atol (float, optional): Absolute tolerance for the stopping criterion. Default is 0.0.
    maxiter (int, optional): Maximum number of iterations. Default is 10000.
    M (callable, optional): Preconditioner function. Default is None.

    Returns:
    array-like: Approximate solution to the linear system Ax = b.

    Notes:
    This function implements the preconditioned Conjugate Gradient method.
    If no preconditioner is provided, the identity preconditioner is used.
    """
    
    if M is None:
        noM = True
        M = lambda x: x
    else:
        noM = False
        #M = partial(_dot, M)
        M = partial(apply_M, M=M)

    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma.real if noM is True else _vdot_real_tree(r, r)
        return (rs > atol2) & (k < maxiter)


    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / _vdot_real_tree(p, Ap).astype(dtype)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_).astype(dtype)
        beta_ = gamma_ / gamma
        p_ = _add(z_, _mul(beta_, p))
        return x_, r_, gamma_, p_, k + 1


    r0 = _sub(b, A(x0))
    p0 = z0 = M(r0)
    dtype = jnp.result_type(*tree_leaves(p0))
    gamma0 = _vdot_real_tree(r0, z0).astype(dtype)
    initial_value = (x0, r0, gamma0, p0, 0)

    x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final