import jax
from CRNNModel import CRNNModel
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple, Union, Optional, Callable, Any
import optax
from jax import jit
jax.config.update("jax_enable_x64", True)


def local_energy(samples, params, model, log_psi) -> List[complex]:
    """Computes the local energy of the system"""
    NUMBER_OF_SAMPLES, N = samples.shape

    spins = 2 * samples - 1
    
    def step_fn_cluster(i, state):
        s, output = state #Why are we flipping the state? because sigma_x is the flip gate!
        # flipped_state = s.at[:, i].set(1-s[:, i]) #debug
        flipped_state = s.at[:, i-1].set(1-s[:, i-1])
        flipped_state = flipped_state.at[:, i+1].set(1-flipped_state[:, i+1])
        flipped_logpsi = model.apply(params,flipped_state) 
        output += -(1-2*flipped_state[:, i])*jnp.exp(flipped_logpsi - log_psi)  #-flipped_state[:, i] is for Z_i term
        # output += -jnp.exp(flipped_logpsi - log_psi)  # Debug
        return s, output

    # Off Diagonal Term
    output = jnp.zeros((NUMBER_OF_SAMPLES), dtype=jnp.complex128)
    _, off_diag_term = jax.lax.fori_loop(2-1, N-2, step_fn_cluster, (samples, output))
    # _, off_diag_term = jax.lax.fori_loop(0, N, step_fn_cluster, (samples, output)) #Debug if you change samples to spins, it does not work suddently :( ???

    flipped_state = samples.at[:, 1].set(1-samples[:, 1])
    flipped_logpsi = model.apply(params,flipped_state) 
    off_diag_term += -(1-2*flipped_state[:,0])*jnp.exp(flipped_logpsi - log_psi)

    flipped_state = samples.at[:, N-2].set(1-samples[:, N-2])
    flipped_state = flipped_state.at[:, N-1].set(1-flipped_state[:, N-1])
    flipped_logpsi = model.apply(params,flipped_state) 
    off_diag_term += -jnp.exp(flipped_logpsi - log_psi)

    flipped_state = samples.at[:, N-3].set(1-samples[:, N-3])
    flipped_logpsi = model.apply(params,flipped_state) 
    off_diag_term += -(1-2*flipped_state[:,N-2])*(1-2*flipped_state[:,N-1])*jnp.exp(flipped_logpsi - log_psi)

    loc_e = off_diag_term

    return loc_e


def get_loss(params, key, NUMBER_OF_SAMPLES, N, model):
    samples = model.apply(params, key, NUMBER_OF_SAMPLES, N,
                          method="sample") 
    log_psi = model.apply(params, samples)
    e_loc = jax.lax.stop_gradient(local_energy(samples, params, model, log_psi))
    e_avg = e_loc.mean()

    loss = 2*jnp.real(jnp.mean(jnp.conjugate(log_psi)*(e_loc-e_avg)))
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
        #new_value = jnp.full(large_item.shape, 10**(-n))
        if len(small_item.shape) == 1:
            new_value = new_value.at[:small_item.shape[0]].set(small_item)
        elif len(small_item.shape) == 2:
            new_value = new_value.at[:small_item.shape[0], :small_item.shape[1]].set(small_item)
        change_item(params, path, new_value)
    return params


def generate_models(max_power_2=8, cell="Vanilla", o_dim=2):
    models = []
    for dim in range(1, max_power_2 + 1):
        models.append(CRNNModel(output_dim=o_dim, num_hidden_units=2 ** dim, RNNcell_type=cell))
    return models

def final_energy(params, key, model, N, num_samples_final):
  samples = model.apply(params,key, num_samples_final, N, method="sample")
  log_psi = model.apply(params,samples)
#   queue_samples = jnp.zeros((N,num_samples_final, N), dtype = jnp.float64)
#   offdiag_logpsi = jnp.zeros((N*num_samples_final), dtype = jnp.float64)
  e_loc = local_energy(samples, params, model, log_psi)
  return jnp.mean(e_loc), jnp.var(e_loc), jnp.sqrt(jnp.var(e_loc))/jnp.sqrt(num_samples_final)


