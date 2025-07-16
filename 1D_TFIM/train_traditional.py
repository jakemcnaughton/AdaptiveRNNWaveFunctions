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

parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--N', type=int, default=20, help='System size')
parser.add_argument('--OUTPUT_DIMENSION', default=2, help='')
parser.add_argument('--NUMBER_OF_SAMPLES', default=100, help='Samples used per step')
parser.add_argument('--LR', default=0.001, type=float, help='Learning Rate')
parser.add_argument('--MAX_POWER', type=int, default=8, help='')
parser.add_argument('--MODEL_TYPE', type=str, help='Vanilla, GRU, or LSTM')
parser.add_argument('--NAME', type=str, help='')

args = parser.parse_args()
print(type(args.LR))
def train(N, model_type):
    print(f"N = {N}, model_type = {model_type}")
    @partial(jit, static_argnums=(3,))
    def step(params, rng_key, opt_state, get_loss=get_loss):
        rng_key, new_key = jax.random.split(rng_key)
        value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, args.NUMBER_OF_SAMPLES, N, model, queue_samples, offdiag_logpsi)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, new_key

    key1, key2 = jax.random.split(jax.random.key(1))
    x = jax.random.randint(key1, (5, 10), 0, 2)
    model = RNNModel(args.OUTPUT_DIMENSION, 2**args.MAX_POWER, args.MODEL_TYPE)
    params_init = model.init(key2, x)
    optimizer = optax.adam(learning_rate=args.LR)
    opt_state = optimizer.init(params_init)
    queue_samples = jnp.zeros((N, args.NUMBER_OF_SAMPLES, N), dtype=jnp.float64)
    offdiag_logpsi = jnp.zeros((N * args.NUMBER_OF_SAMPLES), dtype=jnp.float64)
    rng_key = jax.random.key(1)

    energies = []
    losses = []
    variations = []
    durations = []
    params = params_init
    epoch = 1
    interval = 50000/8
    while epoch <= 50000:
        if epoch % (interval) == 0:
            with open(f'../Results/1D_TFIM/{N}/Traditional/{args.NAME}/Params/{epoch}_params.pkl', 'wb') as f:
                pickle.dump(params, f)
            results = pd.DataFrame({'Energies': [float(x) for x in energies], 'Losses': [float(x) for x in losses], 'Variations': [float(x) for x in variations], 'Time': durations})
            results.to_csv(f'../Results/1D_TFIM/{N}/Traditional/{args.NAME}/Outputs/{epoch}_outputs.csv')

        s = time.time()
        params, opt_state, (loss, eloc), rng_key = step(params, rng_key, opt_state)
        loss.block_until_ready()
        e = time.time()
        duration = e - s
        durations.append(duration)
        current_energy = jnp.mean(eloc)
        losses.append(loss)
        variations.append(jnp.var(eloc))
        if epoch % 100 == 0:
            print("Step = {}, Energy = {:.6f}, Var = {:.6f}, Loss = {:.6f}".format(epoch, jnp.mean(eloc), jnp.var(eloc), loss))
        energies.append(current_energy)
        epoch += 1

    dict_0 = {'Energies': [float(x) for x in energies], 'Losses': [float(x) for x in losses],
              'Variations': [float(x) for x in variations], 'Time': durations}
    df = pd.DataFrame(dict_0)
    save_path = f'../Results/1D_TFIM/{N}/Traditional/{args.NAME}/Outputs/final_outputs.csv'
    df.to_csv(save_path)
    print(f"Final Epoch = {epoch}")

if not os.path.exists(f'../Results/1D_TFIM/{args.N}/Traditional/{args.NAME}/'):
    os.makedirs(f'../Results/1D_TFIM/{args.N}/Traditional/{args.NAME}/Outputs/')
    os.makedirs(f'../Results/1D_TFIM/{args.N}/Traditional/{args.NAME}/Params/')

with open(__file__, 'r') as current_file:
    code_content = current_file.read()
    
with open(f'../Results/1D_TFIM/{args.N}/Traditional/{args.NAME}/train_backup.py', 'w') as backup_file:
    backup_file.write(code_content)
    
def main():
    starttime = time.perf_counter()
    train(args.N, args.MODEL_TYPE)
    duration = timedelta(seconds=time.perf_counter()-starttime)
    print(f'Job took: {duration} seconds')

main()
