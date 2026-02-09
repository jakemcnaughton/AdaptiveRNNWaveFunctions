import jax
from TwoD_RNN import StackedRNNModel
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple, Union, Optional, Callable, Any
from jax import jit
from math import ceil
jax.config.update("jax_enable_x64", True)

from typing import List
import jax
import jax.numpy as jnp

def local_energy(samples, params, model, log_psi, J1=1.0, J2=0.5) -> List[complex]:
    """
    2D square-lattice J1-J2 model (open BC):
      - J1 on nearest neighbors (horizontal + vertical)
      - J2 on diagonal neighbors (two diagonals per plaquette)
    Uses binary spins s in {0,1} with Sz = (2s-1)/2, so Sz_i Sz_j = 0.25*(2s_i-1)(2s_j-1).
    Off-diagonal exchange implemented by flipping both bits when (s_i + s_j == 1) with amplitude (J/2).
    """

    numsamples, Nx, Ny = samples.shape
    local_energies = jnp.zeros((numsamples,), dtype=jnp.complex128)

    # -------------------- Diagonal (Sz Sz) terms --------------------
    # Horizontal J1: (i,j) with (i+1,j)
    for i in range(Nx - 1):
        spins_products = 0.25 * (2 * samples[:, i, :] - 1) * (2 * samples[:, i + 1, :] - 1)
        local_energies += J1 * jnp.sum(spins_products, axis=1)

    # Vertical J1: (i,j) with (i,j+1)
    for j in range(Ny - 1):
        spins_products = 0.25 * (2 * samples[:, :, j] - 1) * (2 * samples[:, :, j + 1] - 1)
        local_energies += J1 * jnp.sum(spins_products, axis=1)

    # Diagonal J2: (i,j) with (i+1,j+1)
    for i in range(Nx - 1):
        spins_products = 0.25 * (2 * samples[:, i, :-1] - 1) * (2 * samples[:, i + 1, 1:] - 1)
        local_energies += J2 * jnp.sum(spins_products, axis=1)

    # Anti-diagonal J2: (i,j) with (i+1,j-1)
    for i in range(Nx - 1):
        spins_products = 0.25 * (2 * samples[:, i, 1:] - 1) * (2 * samples[:, i + 1, :-1] - 1)
        local_energies += J2 * jnp.sum(spins_products, axis=1)

    # -------------------- Off-diagonal exchange terms --------------------
    # For each bond (a,b), contribute (J/2) * 1_{s_a != s_b} * exp(logpsi(flipped)-logpsi)

    def step_fn_horizontal(n, state):
        s, output = state
        _, Nx_, Ny_ = s.shape
        i = n // Ny_
        j = n % Ny_
        # bond: (i,j) - (i+1,j)
        flipped = s.at[:, i, j].set(1 - s[:, i, j])
        flipped = flipped.at[:, i + 1, j].set(1 - flipped[:, i + 1, j])
        flipped_logpsi = model.apply(params, flipped)
        output += (s[:, i, j] + s[:, i + 1, j] == 1) * (0.5 * J1) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_vertical(n, state):
        s, output = state
        _, Nx_, Ny_ = s.shape
        j = n // Nx_
        i = n % Nx_
        # bond: (i,j) - (i,j+1)
        flipped = s.at[:, i, j].set(1 - s[:, i, j])
        flipped = flipped.at[:, i, j + 1].set(1 - flipped[:, i, j + 1])
        flipped_logpsi = model.apply(params, flipped)
        output += (s[:, i, j] + s[:, i, j + 1] == 1) * (0.5 * J1) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_diag_downright(n, state):
        s, output = state
        _, Nx_, Ny_ = s.shape
        i = n // (Ny_ - 1)
        j = n % (Ny_ - 1)
        # bond: (i,j) - (i+1,j+1)
        flipped = s.at[:, i, j].set(1 - s[:, i, j])
        flipped = flipped.at[:, i + 1, j + 1].set(1 - flipped[:, i + 1, j + 1])
        flipped_logpsi = model.apply(params, flipped)
        output += (s[:, i, j] + s[:, i + 1, j + 1] == 1) * (0.5 * J2) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_diag_upright(n, state):
        s, output = state
        _, Nx_, Ny_ = s.shape
        i = n // (Ny_ - 1)
        j = (n % (Ny_ - 1)) + 1
        # bond: (i,j) - (i+1,j-1)
        flipped = s.at[:, i, j].set(1 - s[:, i, j])
        flipped = flipped.at[:, i + 1, j - 1].set(1 - flipped[:, i + 1, j - 1])
        flipped_logpsi = model.apply(params, flipped)
        output += (s[:, i, j] + s[:, i + 1, j - 1] == 1) * (0.5 * J2) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    output0 = jnp.zeros((numsamples,), dtype=jnp.complex128)

    # NN off-diagonal
    _, off_v = jax.lax.fori_loop(0, Nx * (Ny - 1), step_fn_vertical, (samples, output0))
    _, off_h = jax.lax.fori_loop(0, (Nx - 1) * Ny, step_fn_horizontal, (samples, output0))

    # Diagonal off-diagonal (only if Ny >= 2)
    if Ny >= 2 and Nx >= 2:
        _, off_d1 = jax.lax.fori_loop(0, (Nx - 1) * (Ny - 1), step_fn_diag_downright, (samples, output0))
        _, off_d2 = jax.lax.fori_loop(0, (Nx - 1) * (Ny - 1), step_fn_diag_upright, (samples, output0))
    else:
        off_d1 = jnp.zeros((numsamples,), dtype=jnp.complex128)
        off_d2 = jnp.zeros((numsamples,), dtype=jnp.complex128)

    local_energies += off_v + off_h + off_d1 + off_d2
    return local_energies


def get_loss(params, key, numsamples, Nx,Ny, model):

    samples = model.apply(params,key,numsamples, Nx, Ny, method="sample")
    #print("Got Samples", flush=True)
    log_psi = model.apply(params,samples)
    #print("Got Log Probs", flush=True)
    e_loc = jax.lax.stop_gradient(local_energy(samples, params, model, log_psi))
    #print("Got e_loc", flush=True)
    e_avg = e_loc.mean()
    #print(e_avg, flush=True)

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
  logpsi = model.apply(params,samples)
  e_loc = local_energy(samples, params, model, logpsi)
  return jnp.mean(e_loc), jnp.var(e_loc), jnp.std(e_loc)/jnp.sqrt(num_samples_final)
