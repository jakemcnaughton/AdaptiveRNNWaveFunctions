import jax
from RNNModel import RNNModel
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple, Union, Optional, Callable, Any
import optax
from jax import jit
jax.config.update("jax_enable_x64", True)

def local_energy(samples, params, model, queue_samples, offdiag_logpsi, log_psi) -> List[float]:
    """Computes the local energy of the system"""
    NUMBER_OF_SAMPLES, N = samples.shape

    # Interaction Term
    interaction_term = - jnp.sum((2 * samples - 1) * (2 * jnp.roll(samples, 1, axis=1) - 1), axis=1)
    interaction_term += (2 * samples[:, 0] - 1) * (2 * samples[:, -1] - 1)  #for OBC

    def step_fn_transverse(i, state):
        s, queue_samples = state
        flipped_state = s.at[:, i].set(1 - s[:, i])
        queue_samples = queue_samples.at[i].set(flipped_state)
        return s, queue_samples

    _, queue_samples = jax.lax.fori_loop(0, N, step_fn_transverse, (samples, queue_samples))

    len_sigmas = N * NUMBER_OF_SAMPLES
    num_offdiag_steps = len_sigmas // 10000 + 1  #I want a maximum number in batch size just to not allocate too much memory while minimizing the number of models calls to get the most out of parallelization
    for i in range(num_offdiag_steps):
        if i < num_offdiag_steps - 1:
            cut = slice((i * len_sigmas) // num_offdiag_steps, ((i + 1) * len_sigmas) // num_offdiag_steps)
        else:
            cut = slice((i * len_sigmas) // num_offdiag_steps, len_sigmas)
        flipped_logpsi = 0.5 * model.apply(params, queue_samples.reshape(N * NUMBER_OF_SAMPLES, N)[cut])
        offdiag_logpsi = offdiag_logpsi.at[cut].set(flipped_logpsi)
    off_diag_term = -jnp.sum(jnp.exp(offdiag_logpsi.reshape(N, NUMBER_OF_SAMPLES) - log_psi.reshape(1, NUMBER_OF_SAMPLES)), axis=0)
    loc_e = interaction_term + off_diag_term

    return loc_e


def get_loss(params, key, NUMBER_OF_SAMPLES, N, model, queue_samples, offdiag_logpsi):
    samples = model.apply(params, key, NUMBER_OF_SAMPLES, N,
                          method="sample") 
    log_probs = model.apply(params, samples)
    e_loc = jax.lax.stop_gradient(local_energy(samples, params, model, queue_samples, offdiag_logpsi, 0.5 * log_probs))
    e_avg = e_loc.mean()

    loss = jnp.mean(jnp.multiply(log_probs, e_loc) - jnp.multiply(e_avg, log_probs))
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
        key = jax.random.PRNGKey(0)
        new_value = jax.random.uniform(key, large_item.shape, minval=-1, maxval=1) * 10 ** (-n)
        if len(small_item.shape) == 1:
            new_value = new_value.at[:small_item.shape[0]].set(small_item)
        elif len(small_item.shape) == 2:
            new_value = new_value.at[:small_item.shape[0], :small_item.shape[1]].set(small_item)
        change_item(params, path, new_value)
    return params


def generate_models(max_power_2=8, cell="Vanilla", o_dim=2):
    models = []
    for dim in range(1, max_power_2 + 1):
        models.append(RNNModel(output_dim=o_dim, num_hidden_units=2 ** dim, RNNcell_type=cell))
    return models

def final_energy(params, key, model, N, num_samples_final):
  samples = model.apply(params,key, num_samples_final, N, method="sample")
  log_probs = model.apply(params,samples)
  queue_samples = jnp.zeros((N,num_samples_final, N), dtype = jnp.float64)
  offdiag_logpsi = jnp.zeros((N*num_samples_final), dtype = jnp.float64)
  e_loc = local_energy(samples, params, model, queue_samples, offdiag_logpsi, 0.5*log_probs)
  return jnp.mean(e_loc), jnp.var(e_loc), jnp.sqrt(jnp.var(e_loc))/jnp.sqrt(num_samples_final)

def final_energy_testing(params, key, model, N, num_samples_final):
  samples = model.apply(params,key, num_samples_final, N, method="sample")
  log_probs = model.apply(params,samples)
  queue_samples = jnp.zeros((N,num_samples_final, N), dtype = jnp.float64)
  offdiag_logpsi = jnp.zeros((N*num_samples_final), dtype = jnp.float64)
  e_loc = local_energy(samples, params, model, queue_samples, offdiag_logpsi, 0.5*log_probs)
  return e_loc

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
