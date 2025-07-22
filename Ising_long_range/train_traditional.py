import os
from functools import partial
from RNNModel import RNNModel
from TFIM1D_Helpers import get_loss, final_energy
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
import numpy as np


parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--N', type=int, help='')
parser.add_argument('--OUTPUT_DIMENSION', default=2, help='')
parser.add_argument('--NUMBER_OF_SAMPLES', default=500, help='')
parser.add_argument('--LR', default=0.001, help='')
parser.add_argument('--MAX_POWER', type=int, default=8, help='')
parser.add_argument('--MODEL_TYPE', type=str, help='')
parser.add_argument('--NAME', type=str, help='')

args = parser.parse_args()

def train(N, model_type):
    print(f"N = {N}, model_type = {model_type}")
    @partial(jit, static_argnums=(3,))
    def step(params, rng_key, opt_state, get_loss=get_loss):
        rng_key, new_key = jax.random.split(rng_key)
        value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, args.NUMBER_OF_SAMPLES, N, model)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, new_key

    key1, key2 = jax.random.split(jax.random.key(1))
    x = jax.random.randint(key1, (5, 10), 0, 2)  # Dummy input data
    model = RNNModel(args.OUTPUT_DIMENSION, 2**args.MAX_POWER, args.MODEL_TYPE)
    params_init = model.init(key2, x)
    optimizer = optax.adam(learning_rate=args.LR)
    opt_state = optimizer.init(params_init)
    rng_key = jax.random.key(1)

    #early_stop = early_stopping.EarlyStopping(min_delta=0.0001, patience=200)

    energies = []
    losses = []
    variations = []
    durations = []
    params = params_init

    #if not os.path.exists(f'../{args.NAME}/Params/{args.NAME}_{model_type}_{N}'):
    #    os.makedirs(f'../{args.NAME}/Params/{args.NAME}_{model_type}_{N}/')
    #while early_stop.should_stop is False:
    for epoch in range(1, 2000):
        s = time.time()
        params, opt_state, (loss, eloc), rng_key = step(params, rng_key, opt_state)
        loss.block_until_ready()
        e = time.time()
        duration = e - s
        durations.append(duration)
        current_energy = jnp.mean(eloc)
        losses.append(loss)
        variations.append(jnp.var(eloc))
        #early_stop = early_stop.update(current_energy)
        if epoch % 5 == 0:
            print("Step = {}, Energy = {:.6f}, Var = {:.6f}, Loss = {:.6f}".format(epoch, jnp.mean(eloc), jnp.var(eloc), loss))
        energies.append(current_energy)
        #if jnp.var(eloc) < 0.0001:
        #    print("Done! Stopped with Var < 0.0001")
        #    break
        #if early_stop.should_stop:
        #    print('Done! Stopped due to early stopping')
        #epoch += 1
        
       

    dict_0 = {'Energies': [float(x) for x in energies], 'Losses': [float(x) for x in losses],
              'Variations': [float(x) for x in variations], 'Time': durations}
    df = pd.DataFrame(dict_0)

    with open(f'./Traditional_to_2000/{args.NAME}/Params/{args.MAX_POWER}_{epoch}_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    save_path = f'./Traditional_to_2000/{args.NAME}/Outputs/{args.NAME}_{model_type}_{N}_outputs.csv'
    df.to_csv(save_path)
    print(f"Final Epoch = {epoch}")
    print(final_energy(params, key1, model, N, 10000))


if not os.path.exists(f'./Traditional_to_2000/{args.NAME}/'):
    os.makedirs(f'./Traditional_to_2000/{args.NAME}/Outputs/')
    os.makedirs(f'./Traditional_to_2000/{args.NAME}/Params/')

with open(__file__, 'r') as current_file:
    code_content = current_file.read()
    
with open(f'./Traditional_to_2000/{args.NAME}/train_backup.py', 'w') as backup_file:
    backup_file.write(code_content)
    
def main():
    starttime = time.perf_counter()
    train(args.N, args.MODEL_TYPE)
    duration = timedelta(seconds=time.perf_counter()-starttime)
    print(f'Job took: {duration} seconds')

main()
