import pandas as pd
import os
import argparse
import time
import numpy as np
import jax
from TwoD_Helper_Functions import get_loss, generate_models, param_transform_automatic, final_energy, StackedRNNModel
jax.config.update("jax_enable_x64", True)

runs=100
samples=10000

key=jax.random.key(1)
results = dict()
N = args.N
lr = args.LR
type = args.type
x = jax.random.randint(key, (5, 10), 0, 2)

titles = []
models = []
energies=[]
vars=[]
errors=[]
include = []
    
for file in include:
    print(file, flush=True)
    results = []
    params = pd.read_pickle(file)
    model = StackedRNNModel(2**8, 2**8, 1, "GRU")
    for i in range(runs):
        key, subkey = jax.random.split(key)
        result = final_energy(params, subkey, model, 6, 6, samples)
        results.extend(result)
    energies.append(np.mean(results))
    print(energies[-1])
    vars.append(np.var(results))
    errors.append(np.sqrt(np.var(results))/np.sqrt(runs*samples))
    models.append(file)

results = {'Model':models, 'Energy':energies, "Variance":vars, 'Error': errors}
df = pd.DataFrame(results)
df.to_csv(f"./2D_Results.csv")
