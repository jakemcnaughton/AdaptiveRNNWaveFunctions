import os
from functools import partial
from TwoD_Helper_Functions import get_loss, generate_models, param_transform_automatic, final_energy, opt_state_transform_automatic
import pandas as pd
from jax import jit
import time
import statistics
import pickle
from datetime import timedelta
import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)
from flax.training import early_stopping
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--Nx', type=int, default=6, help='')
parser.add_argument('--Ny', type=int, default=6, help='')
parser.add_argument('--NUMBER_OF_SAMPLES', default=500, help='')
parser.add_argument('--MAX_POWER', type=int, default=8, help='')
parser.add_argument('--MODEL_TYPE', type=str, default="GRU", help='')
parser.add_argument('--NAME', type=str, default="Test", help='')
parser.add_argument('--LR', type=float, default=0.01, help='')
args = parser.parse_args()

if not os.path.exists(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/'):
    os.makedirs(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Outputs/')
    os.makedirs(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Params/')

with open(__file__, 'r') as current_file:
    code_content = current_file.read()
    
with open(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/train_backup.py', 'w') as backup_file:
    backup_file.write(code_content)

starttime = time.perf_counter()
@partial(jit, static_argnums=(3, 4))
def step(params, rng_key, opt_state, m, optimizer, get_loss=get_loss):
    rng_key, new_key = jax.random.split(rng_key)
    value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, args.NUMBER_OF_SAMPLES, args.Nx, args.Ny, models[m])#, queue_samples, offdiag_logpsi)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value, new_key

key1, key2 = jax.random.split(jax.random.key(1))
x = jax.random.randint(key1, (2,4,4), 0, 2) # Dummy input data
### Approximation for the ground state energy using DMRG on 6x6 = -21.7267848
N = args.Nx*args.Ny
#queue_samples = jnp.zeros((2*args.Nx,args.Ny,args.NUMBER_OF_SAMPLES, args.Nx,args.Ny), dtype = jnp.float64)
#offdiag_logpsi = jnp.zeros((2*N*args.NUMBER_OF_SAMPLES), dtype = jnp.float64)
rng_key = jax.random.key(1)
models = generate_models(max_power_2 = args.MAX_POWER, n_layers = 1, RNNcell_type = args.MODEL_TYPE)
params = models[0].init(key2, x)
energies = []
losses = []
dimensions = []
variations = []
durations = []
epoch = 1
optimizer = optax.adam(0.005)
opt_state = optimizer.init(params)
    
for dim in range(args.MAX_POWER):
    if dim > 0:
        with open(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Params/{dim}_{epoch-1}_params.pkl', 'wb') as f:
            pickle.dump(params, f)
        results = pd.DataFrame({'Energies': [float(x) for x in energies], 'Losses': [float(x) for x in losses], 'Variations': [float(x) for x in variations], 'Time': durations})
        results.to_csv(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Outputs/{epoch-1}_outputs.csv')
        params = param_transform_automatic(params, dim, models, key2, x)
        if dim == 4:
            optimizer = optax.adam(0.0005)
            old_opt_state = opt_state
            new_opt_state = optimizer.init(params)
            opt_state = opt_state_transform_automatic(old_opt_state, new_opt_state)
        
        else:
            old_opt_state = opt_state
            new_opt_state = optimizer.init(params)
            opt_state = opt_state_transform_automatic(old_opt_state, new_opt_state)

    start_epoch = epoch        
    early_stop = early_stopping.EarlyStopping(min_delta=10**(-(dim+1)/2), patience=10000)
    while early_stop.should_stop is False and (epoch-start_epoch) <= 100000: 
        s = time.time()
        params, opt_state, (loss, eloc), rng_key = step(params, rng_key, opt_state, dim, optimizer)
        loss.block_until_ready()  # Force the computation to finish
        e = time.time()
        dimensions.append(2**(dim+1))
        energies.append(jnp.mean(eloc))
        losses.append(loss)
        variations.append(jnp.var(eloc))
        durations.append(e-s)
        if epoch-start_epoch > 500:
            rolling_avg = statistics.mean([float(x) for x in variations[-500:]])
            early_stop = early_stop.update(rolling_avg)
        if epoch % 100 == 0:
            print("Step = {}, Energy = {:.6f}, Var = {:.6f}".format(epoch, jnp.mean(eloc), jnp.var(eloc)))
        epoch += 1

results = pd.DataFrame({'Energies': [float(x) for x in energies], 'Losses': [float(x) for x in losses], 'Variations': [float(x) for x in variations], 'Model dh': dimensions, 'Time': durations})
results.to_csv(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Outputs/final_outputs.csv')
with open(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Params/{args.MAX_POWER}_{epoch-1}_params.pkl', 'wb') as f:
    pickle.dump(params, f)
print(f"Final Epoch = {epoch}")
duration = timedelta(seconds=time.perf_counter()-starttime)
print(f'Job took: {duration} seconds')
