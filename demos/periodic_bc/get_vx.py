import argparse

import jax.numpy as jnp
parser = argparse.ArgumentParser(description='Process vx value.')
parser.add_argument('-vx', type=float, required=True, help='vx value')
args = parser.parse_args()

vx = args.vx
data = vx*jnp.ones((1,))
jnp.save('vx', data)
