from TFIM1D_Helpers import get_loss, generate_models, param_transform_automatic, final_energy_testing
import pandas as pd
import os
from RNNModel import RNNModel
import argparse
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

# Uses runs*samples total samples
runs=100
samples=10000

key=jax.random.key(1)
results = dict()
x = jax.random.randint(key, (5, 10), 0, 2)

models = []
energies=[]
vars=[]
errors=[]
titles = []

for N in [20, 40, 60, 80, 100]:
    for root, dirs, files in os.walk(f'../Results/1D_TFIM/{N}/Traditional/'):
        if f"GRU_{N}_0.0001" in root:
            for file in files:
                print(file)
                results = []
                if file.endswith("50000_params.pkl"):
                    model_type = "GRU"
                    params = pd.read_pickle(os.path.join(root, file))
                    model = RNNModel(2, 2**8, model_type)
                    for i in range(runs):
                        # Change key
                        key, subkey = jax.random.split(key)
                        result = final_energy_testing(params, subkey, model, N, samples)
                        results.extend(result)
                    energies.append(np.mean(result))
                    vars.append(np.var(result))
                    errors.append(np.sqrt(np.var(result))/np.sqrt(samples))
                    models.append(root)

results = {"Model":models, "Energy":energies, "Variance":vars, "Error": errors}
df = pd.DataFrame(results)
df.to_csv(f"./1D_Results.csv")
