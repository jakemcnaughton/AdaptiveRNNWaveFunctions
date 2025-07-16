import os
from functools import partial
from TFIM1D_Helpers import get_loss, generate_models, param_transform_automatic, opt_state_transform_automatic
from jax import jit
import pandas as pd
import time
import pickle
from datetime import timedelta
import jax
import jax.numpy as jnp
from flax.training import early_stopping
import optax
jax.config.update("jax_enable_x64", True)
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--N', type=int, help='System size')
parser.add_argument('--OUTPUT_DIMENSION', default=2, help='')
parser.add_argument('--NUMBER_OF_SAMPLES', default=500, help='Number of samples used per step')
parser.add_argument('--LR_1', default=0.005, type=float, help='Learning rate for the first half of training')
parser.add_argument('--LR_2', default=0.0005, type=float, help='Learning rate for the second half of training')
parser.add_argument('--MAX_POWER', type=int, default=8, help='Maximum power of 2 for the final model dimension and hidden dimension')
parser.add_argument('--MODEL_TYPE', type=str, default="GRU", help='Vanilla, GRU, or LSTM')
parser.add_argument('--NAME', type=str, default="1D_TFIM", help='Name for directory which will be create to save parameters and outputs')
args = parser.parse_args()

def train(N, model_type):
    print(f"N = {N}, model_type = {model_type}, Adaptive")
    @partial(jit, static_argnums=(3,))
    def step(params, rng_key, opt_state, m, get_loss=get_loss):
        rng_key, new_key = jax.random.split(rng_key)
        value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, args.NUMBER_OF_SAMPLES, args.N, models[m], queue_samples, offdiag_logpsi)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, new_key

    #Initialise Variables for model
    key1, key2 = jax.random.split(jax.random.key(1))
    x = jax.random.randint(key1, (5, 10), 0, 2)  # Dummy input data
    models = generate_models(args.MAX_POWER, model_type, args.OUTPUT_DIMENSION)
    params_init = models[0].init(key2, x)
    optimizer = optax.adam(learning_rate=args.LR_1)
    opt_state = optimizer.init(params_init)
    queue_samples = jnp.zeros((args.N, args.NUMBER_OF_SAMPLES, N), dtype=jnp.float64)
    offdiag_logpsi = jnp.zeros((args.N *args. NUMBER_OF_SAMPLES), dtype=jnp.float64)
    rng_key = jax.random.key(1)

    #Define lists to store data each step
    energies = []
    losses = []
    dimensions = []
    variations = []
    durations = []
    params = params_init
    epoch = 1

    #Train Model
    for dim in range(len(models)):
        if dim > 0:
            #Change params, LR, and save old params
            if dim == 4:
                optimizer = optax.adam(args.LR_2)#learning_rate=args.LR-(0.009/7)*dim)
            params = param_transform_automatic(params, dim, models, key2, x)
            old_opt_state = opt_state
            new_opt_state = optimizer.init(params)
            opt_state = opt_state_transform_automatic(old_opt_state, new_opt_state)
            with open(f'../Results/1D_TFIM/{N}/Adaptive/{args.NAME}/Params/{epoch}_params.pkl', 'wb') as f:
                pickle.dump(params, f)

        if dim < args.MAX_POWER - 1:
            model_epoch_count = 0
            while model_epoch_count <= 6250:
                s = time.time()
                params, opt_state, (loss, eloc), rng_key = step(params, rng_key, opt_state, m=dim)
                loss.block_until_ready()
                e = time.time()
                model_epoch_count += 1
                dimensions.append(2**(dim+1))
                current_energy = jnp.mean(eloc)
                energies.append(current_energy)
                losses.append(loss)
                variations.append(jnp.var(eloc))
                durations.append(e - s)
                if epoch % 100 == 0:
                    print("Step = {}, Energy = {:.6f}, Var = {:.6f}".format(epoch, jnp.mean(eloc), jnp.var(eloc)))
                epoch += 1

        else:
            while epoch < 50000:
                s = time.time()
                params, opt_state, (loss, eloc), rng_key = step(params, rng_key, opt_state, m=dim)
                loss.block_until_ready()
                e = time.time()
                dimensions.append(2**(dim+1))
                current_energy = jnp.mean(eloc)
                energies.append(current_energy)
                losses.append(loss)
                variations.append(jnp.var(eloc))
                durations.append(e - s)
                if epoch % 100 == 0:
                    print("Step = {}, Energy = {:.6f}, Var = {:.6f}".format(epoch, jnp.mean(eloc), jnp.var(eloc)))
                epoch += 1
                
    dict_0 = {'Energies': [float(x) for x in energies], 'Losses': [float(x) for x in losses],
              'Variations': [float(x) for x in variations], 'Model dh': dimensions, 'Time': durations}
    df = pd.DataFrame(dict_0)
    with open(f'../Results/1D_TFIM/{args.N}/Adaptive/{args.NAME}/Params/{args.MAX_POWER}_{epoch}_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    save_path = f'../Results/1D_TFIM/{args.N}/Adaptive/{args.NAME}/Outputs/final_outputs.csv'
    df.to_csv(save_path)
    print(f"Final Epoch = {epoch}")
    print(final_energy(params, key1, models[-1], args.N, 10000))

if not os.path.exists(f'../Results/1D_TFIM/{args.N}/Adaptive/{args.NAME}/'):
    os.makedirs(f'../Results/1D_TFIM/{args.N}/Adaptive/{args.NAME}/Outputs/')
    os.makedirs(f'../Results/1D_TFIM/{args.N}/Adaptive/{args.NAME}/Params/')

with open(__file__, 'r') as current_file:
    code_content = current_file.read()
    
with open(f'../Results/1D_TFIM/{args.N}/Adaptive/{args.NAME}/train_backup.py', 'w') as backup_file:
    backup_file.write(code_content)

#Train model and time it
def main():
    starttime = time.perf_counter()
    train(args.N, args.MODEL_TYPE)
    duration = timedelta(seconds=time.perf_counter() - starttime)
    print(f'Job took: {duration} seconds')
main()
