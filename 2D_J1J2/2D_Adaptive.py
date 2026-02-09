import os
from functools import partial
from TwoD_Helper_Functions import get_loss, generate_models, param_transform_automatic, final_energy, opt_state_transform_automatic
from TwoD_RNN import StackedRNNModel
import pandas as pd
from jax import jit
import time
import pickle
from datetime import timedelta
import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)
import argparse
from jax.tree_util import tree_map

parser = argparse.ArgumentParser(description='Adaptive 2D RNN for 2D J1J2')
parser.add_argument('--Nx', type=int, default=8, help='System size in x dimension')
parser.add_argument('--Ny', type=int, default=8, help='System size in y dimension')
parser.add_argument('--NUMBER_OF_SAMPLES', default=500, help='Number of samples used each step')
parser.add_argument('--MAX_POWER', type=int, default=8, help='Maximum power of 2 for the model dimension and hidden dimension')
parser.add_argument('--MODEL_TYPE', type=str, default="GRU", help='RNN cell type: Vanilla or GRU')
parser.add_argument('--NAME', type=str, default="Test", help='Name of directory to save results')
args = parser.parse_args()

@jax.jit
def lr_schedule(t):
    base_lr = 5e-4
    scale = 5000

    def constant_lr(_):
        return base_lr

    def decay_lr(t):
        return base_lr * (1 + ((t - 100000) / scale)) ** -1

    return jax.lax.cond(t <= 100000, constant_lr, decay_lr, t)


def train():
    print(f"N = {args.Nx}, Model Type = {args.MODEL_TYPE}, Adaptive Run, Samples = {args.NUMBER_OF_SAMPLES}")
    @partial(jit, static_argnums=(3, 4))  # m and optimizer are static
    def step(params, rng_key, opt_state, m, optimizer, get_loss=get_loss):
        rng_key, new_key = jax.random.split(rng_key)

        (loss, aux), grads = jax.value_and_grad(get_loss, has_aux=True)(
            params, new_key,
            args.NUMBER_OF_SAMPLES, args.Nx, args.Ny,
            models[m]#, queue_samples, offdiag_logpsi
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, (loss, aux), new_key
    
    key1, key2 = jax.random.split(jax.random.key(1))
    x = jax.random.randint(key1, (2,4,4), 0, 2) # Dummy input data
    ### Approximation for the ground state energy using DMRG on 6x6 = -21.7267848
    N = args.Nx*args.Ny
    #queue_samples = jnp.zeros((2*args.Nx,args.Ny,args.NUMBER_OF_SAMPLES, args.Nx,args.Ny), dtype = jnp.float64)
    #offdiag_logpsi = jnp.zeros((2*N*args.NUMBER_OF_SAMPLES), dtype = jnp.float64)
    rng_key = jax.random.key(1)
    models = generate_models(max_power_2 = args.MAX_POWER, n_layers = 1, RNNcell_type = args.MODEL_TYPE)
    starting = 4
    params = models[starting].init(key2, x)
    energies = []
    losses = []
    dimensions = []
    variations = []
    durations = []
    epoch = 1
    interval = 200000/(args.MAX_POWER-starting)


    optimizer = optax.adam(learning_rate=lambda _: lr_schedule(epoch))
    opt_state = optimizer.init(params)
        
    for dim in range(starting, args.MAX_POWER):
        print(models[dim], flush=True)
        if dim > starting:
            with open(f'../Results/2D_J1J2/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Params/{dim}_{epoch}_params.pkl', 'wb') as f:
                pickle.dump(params, f)
            results = pd.DataFrame({'Energies': [float(x.real) for x in energies], 'Losses': [float(x) for x in losses], 'Variations': [float(x) for x in variations], 'Time': durations})
            results.to_csv(f'../Results/2D_J1J2/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Outputs/{epoch}_outputs.csv')
            params = param_transform_automatic(params, dim, models, key2, x)
            old_opt_state = opt_state
            optimizer = optax.adam(learning_rate=lambda _: lr_schedule(epoch))
            new_opt_state = optimizer.init(params)
            opt_state = opt_state_transform_automatic(old_opt_state, new_opt_state)

                    
        while epoch <= (dim-starting+1)*interval:
            s = time.time()
            params, opt_state, (loss, eloc), rng_key = step(params, rng_key, opt_state, dim, optimizer)
            loss.block_until_ready()
            e = time.time()
            dimensions.append(2**(dim+1))
            energies.append(jnp.mean(eloc))
            losses.append(loss)
            variations.append(jnp.var(eloc))
            durations.append(e-s)
            if epoch % 100 == 0:
                print("Step = {}, Energy = {:.6f}, Var = {:.6f}".format(epoch, jnp.mean(eloc), jnp.var(eloc)))
            epoch += 1

    results = pd.DataFrame({'Energies': [float(x.real) for x in energies], 'Losses': [float(x) for x in losses], 'Variations': [float(x) for x in variations], 'Model dh': dimensions, 'Time': durations})
    results.to_csv(f'../Results/2D_J1J2/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Outputs/final_outputs.csv')

    with open(f'../Results/2D_J1J2/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Params/{args.MAX_POWER}_{epoch-1}_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    print(f"Final Epoch = {epoch-1}")
    print(final_energy(params, key1, models[-1], args.Nx, args.Ny, 10000))



if not os.path.exists(f'../Results/2D_J1J2/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/'):
    os.makedirs(f'../Results/2D_J1J2/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Outputs/')
    os.makedirs(f'../Results/2D_J1J2/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/Params/')

with open(__file__, 'r') as current_file:
    code_content = current_file.read()
    
with open(f'../Results/2D_J1J2/{args.Nx}x{args.Ny}/Adaptive/{args.NAME}/train_backup.py', 'w') as backup_file:
    backup_file.write(code_content)


def main():
    starttime = time.perf_counter()
    train()
    duration = timedelta(seconds=time.perf_counter()-starttime)
    print(f'Job took: {duration} seconds')

main()
