"""Stable test-level entry points, matching compiled production time loops.

Defining these once avoids re-tracing a new closure for each numerical case.
The public eager functions are exercised separately by the API regression.
"""

import jax
from PyPIC3D.solvers.yee import first_order_yee as eager

assemble_yee_curl = eager.assemble_yee_curl
update_E = jax.jit(eager.update_E, static_argnames="static_parameters")
update_B = jax.jit(eager.update_B, static_argnames="static_parameters")
yee_derivatives_e_to_b = jax.jit(eager.yee_derivatives_e_to_b, static_argnames="static_parameters")
yee_derivatives_b_to_e = jax.jit(eager.yee_derivatives_b_to_e, static_argnames="static_parameters")
yee_curl_e_to_b = jax.jit(eager.yee_curl_e_to_b, static_argnames="static_parameters")
yee_curl_b_to_e = jax.jit(eager.yee_curl_b_to_e, static_argnames="static_parameters")
