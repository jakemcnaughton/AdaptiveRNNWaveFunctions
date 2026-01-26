import jax
from TwoD_RNN import StackedRNNModel
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple, Union, Optional, Callable, Any
from jax import jit
from math import ceil
jax.config.update("jax_enable_x64", True)

def local_energy(samples, params, model, log_psi) -> List[float]:

    #Computes the local energy of the 2D Heisenberg model

    numsamples,Nx,Ny = samples.shape
    N = Nx*Ny
    local_energies = jnp.zeros((numsamples), dtype = jnp.float64)
    for i in range(Nx-1): #diagonal elements (right neighbours)
        spins_products = 0.25*(2*samples[:,i]-1)*(2*samples[:,i+1]-1)
        local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

    for j in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice))
        spins_products = 0.25*(2*samples[:,:,j]-1)*(2*samples[:,:,j+1]-1)
        local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

    def step_fn_horizontal(n, state):
        s, output = state
        _, Nx,Ny = s.shape
        i = (n//Ny) #set back to zero when equal to Nx-1
        j = n%Ny
        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i+1,j].set(1 - flipped_state[:, i+1,j])
        flipped_logpsi = 0.5*model.apply(params,flipped_state)
        output += (s[:, i,j] + s[:, i+1,j] == 1) *(-0.5)* jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_vertical(n, state):
        s, output = state
        _, Nx,Ny = s.shape
        j = (n//Nx) #set back to zero when equal to Nx-1
        i = n%Nx
        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i,j+1].set(1 - flipped_state[:, i,j+1])
        flipped_logpsi = 0.5*model.apply(params,flipped_state)
        output += ((s[:, i,j] + s[:, i,j+1] == 1)*(-0.5))*jnp.exp(flipped_logpsi - log_psi)
        return s, output

    # Off Diagonal Term
    output = jnp.zeros((numsamples), dtype=jnp.float64)
    _, off_diag_term_vertical = jax.lax.fori_loop(0, Nx*(Ny-1), step_fn_vertical, (samples, output))
    _, off_diag_term_horizontal = jax.lax.fori_loop(0, (Nx-1)*(Ny), step_fn_horizontal, (samples, output))
    local_energies += off_diag_term_vertical +  off_diag_term_horizontal
    return local_energies

def get_loss(params, key, numsamples, Nx,Ny, model):

    samples = model.apply(params,key,numsamples, Nx, Ny, method="sample")
    #print("Got Samples", flush=True)
    log_probs = model.apply(params,samples)
    #print("Got Log Probs", flush=True)
    e_loc = jax.lax.stop_gradient(local_energy(samples, params, model, 0.5*log_probs))
    #print("Got e_loc", flush=True)
    e_avg = e_loc.mean()
    #print(e_avg, flush=True)

    loss = jnp.mean(jnp.multiply(log_probs, e_loc) - jnp.multiply(log_probs, e_avg))
    #print(loss, flush=True)
    return loss, e_loc


def recursive_items(dictionary, current_path=None):
    if current_path is None:
        current_path = []
    for key, value in dictionary.items():
        new_path = current_path + [key]
        if isinstance(value, dict):
            yield from recursive_items(value, new_path)
        else:
            yield new_path, value


def access_item(dictionary, path):
    item = dictionary
    for key in path:
        item = item[key]
    return item


def change_item(dictionary, path, new_value):
    item = dictionary
    for key in path[:-1]:
        item = item[key]
    item[path[-1]] = new_value


def param_transform_automatic(params, n, models, key2, x):
    # Transfer parameters for the any cell into top left of new parameters matrix
    params_large = models[n].init(key2, x)
    for path, value in recursive_items(params, []):
        small_item = access_item(params, path)
        large_item = access_item(params_large, path)
        #new_value = jnp.zeros_like(large_item)
        key = jax.random.PRNGKey(0)
        new_value = jax.random.uniform(key, large_item.shape, minval=-1, maxval=1) * 10 ** (-n)
        if len(small_item.shape) == 1:
            new_value = new_value.at[:small_item.shape[0]].set(small_item)
        elif len(small_item.shape) == 2:
            new_value = new_value.at[:small_item.shape[0], :small_item.shape[1]].set(small_item)
        change_item(params, path, new_value)
    return params

def generate_models(max_power_2, n_layers, RNNcell_type, starting=1, powers=None):
    models = []
    if powers is not None:
        for power in powers:
            models.append(StackedRNNModel(d_hidden=2**power, d_model=2**power, n_layers=n_layers, RNNcell_type = RNNcell_type))
        return models
    
    for dim in range(starting, max_power_2+1):
        models.append(StackedRNNModel(d_hidden=2**dim, d_model=2**dim, n_layers=n_layers, RNNcell_type = RNNcell_type))
    return models

def opt_state_transform_automatic(opt_state_old, opt_state_new):
  # Transfer parameters for the any cell into top left of new parameters matrix
    for i in range(1, 3):
        for path, value in recursive_items(opt_state_old[0][i], []):
            small_item = access_item(opt_state_old[0][i], path)
            large_item = access_item(opt_state_new[0][i], path)
            new_value = jnp.zeros_like(large_item)
            #new_value = jnp.full(large_item.shape, 10**(-n))
            if len(small_item.shape) == 1:
                new_value = new_value.at[:small_item.shape[0]].set(small_item)
            elif len(small_item.shape) == 2:
                new_value = new_value.at[:small_item.shape[0], :small_item.shape[1]].set(small_item)
            change_item(opt_state_new[0][i], path, new_value)
    new_count = opt_state_old[0][0]
    opt_state_new = jax.tree_map(
        lambda x: new_count if x is opt_state_new[0][0] else x,
        opt_state_new,
    )
    return opt_state_new

def final_energy(params, key, model, Nx, Ny, num_samples_final):
  samples = model.apply(params, key, num_samples_final, Nx, Ny, method="sample")
  log_probs = model.apply(params,samples)
  e_loc = local_energy(samples, params, model, 0.5*log_probs)#, offdiag_logpsi, 0.5*log_probs)
  return e_loc
