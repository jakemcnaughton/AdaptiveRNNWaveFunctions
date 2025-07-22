import os
from functools import partial
from TFIM1D_Helpers import get_loss, generate_models, param_transform_automatic, final_energy
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
parser.add_argument('--N', type=int, help='')
parser.add_argument('--OUTPUT_DIMENSION', default=2, help='')
parser.add_argument('--NUMBER_OF_SAMPLES', default=100, help='')
parser.add_argument('--LR', default=0.01, help='')
parser.add_argument('--MAX_POWER', type=int, default=8, help='')
parser.add_argument('--MODEL_TYPE', type=str, default="Vanilla", help='')
parser.add_argument('--NAME', type=str, default="Trial", help='')
args = parser.parse_args()

def train(N, model_type):
    print(f"N = {N}, model_type = {model_type}, Adaptive Run")
    @partial(jit, static_argnums=(3,))
    def step(params, rng_key, opt_state, m, get_loss=get_loss):
        rng_key, new_key = jax.random.split(rng_key)
        value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, args.NUMBER_OF_SAMPLES, args.N, models[m])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, new_key

    #Initialise Variables for model
    key1, key2 = jax.random.split(jax.random.key(1))
    x = jax.random.randint(key1, (5, 10), 0, 2)  # Dummy input data
    models = generate_models(args.MAX_POWER, model_type, args.OUTPUT_DIMENSION)
    params_init = models[0].init(key2, x)
    optimizer = optax.adam(learning_rate=args.LR)
    opt_state = optimizer.init(params_init)
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
            optimizer = optax.adam(learning_rate=args.LR-(0.009/7)*dim)
            params = param_transform_automatic(params, dim, models, key2, x)
            opt_state = optimizer.init(params)
            with open(f'./Adaptive_2000/{args.NAME}/Params/{dim}_{epoch}_params.pkl', 'wb') as f:
                pickle.dump(params, f)


        if dim < args.MAX_POWER - 1:
            model_epoch_count = 0
            #early_stop = early_stopping.EarlyStopping(min_delta=10**(-(dim+1)), patience=2**(13-dim))
            while model_epoch_count <= 200: #dim goes from 0 to 7.. 1st model must met 0.01, last one 0.01/8
                s = time.time()
                params, opt_state, (loss, eloc), rng_key = step(params, rng_key, opt_state, m=dim)
                e = time.time()
                model_epoch_count += 1
                dimensions.append(2**(dim+1))
                current_energy = jnp.mean(eloc)
                energies.append(current_energy)
                losses.append(loss)
                variations.append(jnp.var(eloc))
                durations.append(e - s)
                #early_stop = early_stop.update(current_energy)
                if epoch % 5 == 0:
                    print("Step = {}, Energy = {:.6f}, Var = {:.6f}".format(epoch, jnp.mean(eloc), jnp.var(eloc)))
                epoch += 1

        else:
            while epoch < 2000:
                s = time.time()
                params, opt_state, (loss, eloc), rng_key = step(params, rng_key, opt_state, m=dim)
                e = time.time()
                dimensions.append(2**(dim+1))
                current_energy = jnp.mean(eloc)
                energies.append(current_energy)
                losses.append(loss)
                variations.append(jnp.var(eloc))
                durations.append(e - s)
                #early_stop = early_stop.update(current_energy)
                if epoch % 5 == 0:
                    print("Step = {}, Energy = {:.6f}, Var = {:.6f}".format(epoch, jnp.mean(eloc), jnp.var(eloc)))
                epoch += 1


    
    dict_0 = {'Energies': [float(x) for x in energies], 'Losses': [float(x) for x in losses],
              'Variations': [float(x) for x in variations], 'Model dh': dimensions, 'Time': durations}
    df = pd.DataFrame(dict_0)
    with open(f'./Adaptive_2000/{args.NAME}/Params/{args.MAX_POWER}_{epoch}_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    save_path = f'./Adaptive_2000/{args.NAME}/Outputs/{args.NAME}_{model_type}_{N}_outputs.csv'
    df.to_csv(save_path)
    print(f"Final Epoch = {epoch}")
    print(final_energy(params, key1, models[-1], args.N, 10000))

# Create Directories
if not os.path.exists(f'./Adaptive_2000/{args.NAME}/'):
    os.makedirs(f'./Adaptive_2000/{args.NAME}/Outputs/')
    os.makedirs(f'./Adaptive_2000/{args.NAME}/Params/')

# Save Code
with open(__file__, 'r') as current_file:
    code_content = current_file.read()
    
with open(f'./Adaptive_2000/{args.NAME}/train_backup.py', 'w') as backup_file:
    backup_file.write(code_content)

#Train model and time it
def main():
    starttime = time.perf_counter()
    train(args.N, args.MODEL_TYPE)
    duration = timedelta(seconds=time.perf_counter() - starttime)
    print(f'Job took: {duration} seconds')
main()