import os
from TwoD_Helper_Functions import get_loss, final_energy
from New_2D_RNN import StackedRNNModel
from jax import jit
import pandas as pd
import time
import pickle
from datetime import timedelta
from functools import partial
import jax
import jax.numpy as jnp
import optax
import argparse
jax.config.update("jax_enable_x64", True)
parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--Nx', type=int, default=8, help='')
parser.add_argument('--Ny', type=int, default=8, help='')
parser.add_argument('--NUMBER_OF_SAMPLES', default=500, help='')
parser.add_argument('--NAME', type=str, default="Trial", help='')
args = parser.parse_args()

starttime = time.perf_counter()
# Create Directories and Save Backup of Code
if not os.path.exists(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Traditional/{args.NAME}/'):
    os.makedirs(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Traditional/{args.NAME}/Outputs/')
    os.makedirs(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Traditional/{args.NAME}/Params/')

with open(__file__, 'r') as current_file:
    code_content = current_file.read()
    
with open(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Traditional/{args.NAME}/train_backup.py', 'w') as backup_file:
    backup_file.write(code_content)

#print(1, flush=True)
@jax.jit
def lr_schedule_fn(step):
    base_lr = 5e-4
    scale = 5000
    return base_lr * (1 + (step / scale)) ** -1

#print(2, flush=True)
@partial(jit, static_argnums=(3,))
def step(params, rng_key, opt_state, optimizer, get_loss=get_loss):
    rng_key, new_key = jax.random.split(rng_key)
    value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, args.NUMBER_OF_SAMPLES, args.Nx, args.Ny, model)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value, new_key

#Initialise Variables for model
key1, key2 = jax.random.split(jax.random.key(1))
x = jax.random.randint(key1, (2,4,4), 0, 2) # Dummy input data
### Approximation for the ground state energy using DMRG on 6x6 = -21.7267848
N = args.Nx*args.Ny
#queue_samples = jnp.zeros((2*args.Nx,args.Ny,args.NUMBER_OF_SAMPLES, args.Nx,args.Ny), dtype = jnp.float64)
#offdiag_logpsi = jnp.zeros((2*N*args.NUMBER_OF_SAMPLES), dtype = jnp.float64)
rng_key = jax.random.key(1)
model = StackedRNNModel(d_hidden=256, d_model=256, n_layers=1, RNNcell_type="GRU")
params_init = model.init(key2, x)
params = params_init
optimizer = optax.adam(learning_rate=lr_schedule_fn)
opt_state = optimizer.init(params)
energies = []
losses = []
variations = []
epoch = 1
durations = []
interval = 200000/8
print(f"Samples = {args.NUMBER_OF_SAMPLES}, N={args.Nx}, Traditional Schedule", flush=True)
while epoch <= 200000:
    if epoch != 1 and epoch % (interval) == 1:
        print(timedelta(seconds=time.perf_counter()-starttime), flush=True)
        with open(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Traditional/{args.NAME}/Params/{epoch-1}_params.pkl', 'wb') as f:
            pickle.dump((params, opt_state), f)
        results = pd.DataFrame({'Energies': [float(x) for x in energies], 'Losses': [float(x) for x in losses], 'Variations': [float(x) for x in variations], 'Time': durations})
        results.to_csv(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Traditional/{args.NAME}/Outputs/{epoch-1}_outputs.csv')

    s = time.time()
    #print("Timing Started", flush=True)
    params, opt_state, (loss, eloc), rng_key = step(params, rng_key, opt_state, optimizer)
    #print("Step done", flush=True)
    loss.block_until_ready()  # Force the computation to finish
    e = time.time()
    #print(f"Timing Ended: {e-s}", flush=True)

    durations.append(e-s)
    energies.append(jnp.mean(eloc))
    losses.append(loss)
    variations.append(jnp.var(eloc))
    #if epoch % 100 == 0:
    print("Step = {}, Energy = {:.6f}, Var = {:.6f}".format(epoch, jnp.mean(eloc), jnp.var(eloc)))

    epoch += 1

results = {'Energies': [float(x) for x in energies], 'Losses': [float(x) for x in losses],
            'Variations': [float(x) for x in variations], 'Time': durations}
df = pd.DataFrame(results)
with open(f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Traditional/{args.NAME}/Params/{epoch-1}_params.pkl', 'wb') as f:
    pickle.dump((params, opt_state), f)

save_path = f'../Results/2D_Heisenberg/{args.Nx}x{args.Ny}/Traditional/{args.NAME}/Outputs/{args.NAME}_outputs.csv'
df.to_csv(save_path)

duration = timedelta(seconds=time.perf_counter()-starttime)
print(f'Job took: {duration} seconds')
