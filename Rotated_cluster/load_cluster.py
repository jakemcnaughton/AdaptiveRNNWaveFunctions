import os
from functools import partial
from CRNNModel import CRNNModel
from ClusterState_Helpers import get_loss, final_energy
from jax import jit
import pandas as pd
import time
import pickle
from datetime import timedelta
import jax
import jax.numpy as jnp
import optax
from flax.training import early_stopping
jax.config.update("jax_enable_x64", True)
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--N', type=int, help='')
parser.add_argument('--OUTPUT_DIMENSION', default=2, help='')
parser.add_argument('--NUMBER_OF_SAMPLES', default=100, help='')
parser.add_argument('--LR', default=0.001, help='')
parser.add_argument('--MAX_POWER', type=int, default=8, help='')
parser.add_argument('--MODEL_TYPE', type=str, default="Vanilla", help='')
# parser.add_argument('--NAME', type=str, default="Trial", help='')
args = parser.parse_args()


#Initialise Variables for model
key1, key2 = jax.random.split(jax.random.key(1))
x = jax.random.randint(key1, (5, 10), 0, 2)  # Dummy input data
model = CRNNModel(args.OUTPUT_DIMENSION, 2**args.MAX_POWER, args.MODEL_TYPE)
params_init = model.init(key2, x)
rng_key = jax.random.key(1)
epoch = 10000

# Attempt to load the checkpoint
with open(f'./Adaptive_cluster_10000/Adapt_64_cluster/Params/{args.MAX_POWER}_{epoch}_params.pkl', 'rb') as f:
    params = pickle.load(f)

# with open(f'./Traditional_cluster_10000/Trad_64_cluster/Params/{args.MAX_POWER}_9999_params.pkl', 'rb') as f:
#     params = pickle.load(f)

print("model loaded, now getting log psi and final energies")
log_psi, E_mean, E_var, error = final_energy(params, key1, model, args.N, 10000)
print("Finished")

jnp.save(f'./Adaptive_cluster_10000/Adapt_64_cluster/Outputs/log_psi_adapt_64.npy', log_psi)
jnp.save(f'./Adaptive_cluster_10000/Adapt_64_cluster/Outputs/final_data_adapt64.npy', [E_mean, E_var, error])


# jnp.save(f'./Traditional_cluster_10000/Trad_64_cluster/Outputs/log_psi_trad_64.npy', log_psi)
# jnp.save(f'./Traditional_cluster_10000/Trad_64_cluster/Outputs/final_data_trad_64.npy', [E_mean, E_var, error])


