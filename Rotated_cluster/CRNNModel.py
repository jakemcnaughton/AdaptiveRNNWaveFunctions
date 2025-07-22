import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as nn

class CRNNModel(nn.Module):
    """
    RNN wavefunction
    """
    output_dim: int
    num_hidden_units: int
    RNNcell_type: str = "GRU"

    def setup(self):
      # Initialize the GRU cell with the specified number of hidden units
      if self.RNNcell_type == "GRU":
        self.cell = nn.GRUCell(
            name='gru_cell',
            features=self.num_hidden_units,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            #kernel_init = jax.nn.initializers.variance_scaling(1.0/self.num_hidden_units, "fan_avg", "uniform"),
            param_dtype = jnp.float64
        )
      elif self.RNNcell_type == "LSTM":
        self.cell = nn.OptimizedLSTMCell(
            name='lstm_cell',
            features=self.num_hidden_units,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            #kernel_init = jax.nn.initializers.variance_scaling(1.0/self.num_hidden_units, "fan_avg", "uniform"),
            param_dtype = jnp.float64
        )
      elif self.RNNcell_type == "Vanilla":
        self.cell = nn.SimpleCell(
            name='vanilla_cell',
            features=self.num_hidden_units,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            #kernel_init = jax.nn.initializers.variance_scaling(1.0/self.num_hidden_units, "fan_avg", "uniform"),
            #kernel_init = jax.nn.initializers.variance_scaling(1.0/self.model_scale, "fan_avg", "uniform"),
            param_dtype = jnp.float64
        )
      else:
        raise ValueError("Invalid RNN cell type")

      self.rnn = nn.RNN(self.cell, return_carry=True)
      self.dense = nn.Dense(
          self.output_dim,
          name = 'dense_layer',
          kernel_init = jax.nn.initializers.glorot_uniform(),
          param_dtype = jnp.float64
      )
      self.dense_phase = nn.Dense(
          self.output_dim,
          name = 'dense_phase_layer',
          kernel_init = jax.nn.initializers.glorot_uniform(),
          param_dtype = jnp.float64
      )

    def __call__(self, inputs):
        # Apply GRU layers
        onehot_inputs = jax.nn.one_hot(inputs, num_classes=self.output_dim)
        shifted_onehot_inputs = jnp.roll(onehot_inputs, 1, axis=1)
        shifted_onehot_inputs = shifted_onehot_inputs.at[:,0].set(jnp.zeros((inputs.shape[0],self.output_dim), dtype = jnp.float64))


        initial_carry = jnp.zeros((inputs.shape[0], self.num_hidden_units), dtype=jnp.float64)


        carry, x = self.rnn(shifted_onehot_inputs, initial_carry = initial_carry)

        # Output layer
        x = self.dense(x)
        # phases = self.dense_phase(x)
        phases = jnp.pi*nn.soft_sign(self.dense_phase(x))

        logits = nn.log_softmax(x, axis=-1)
        log_probabilities = jnp.sum(logits * onehot_inputs, axis = (1,2))
        sum_phases = jnp.sum(phases * onehot_inputs, axis = (1,2))
        return 0.5*log_probabilities + 1j*sum_phases

    def sample(self,key,numsamples,N):
        """Sample from the model for a given system size N and a number of samples `numsamples`"""
        inputs = jnp.zeros((numsamples,self.output_dim), dtype = jnp.float64)
        # hidden_states = jnp.zeros((numsamples,self.num_hidden_units), dtype = jnp.float64)
        hidden_states = self.cell.initialize_carry(jax.random.key(1), inputs.shape)

        samples_onehot = jnp.zeros((numsamples,N,self.output_dim), dtype = jnp.float64)
        samples = jnp.zeros((numsamples,N), dtype = jnp.float64)
        keys = jax.random.split(key, N) #pre-generate keys to get more randomness

        for n in range(N):
            hidden_states,inputs = self.cell(hidden_states,inputs)  # apply each layer
            inputs = self.dense(inputs)
            samples = samples.at[:,n].set(jax.random.categorical(key=keys[n], logits=inputs))
            inputs = jax.nn.one_hot(samples[:,n], num_classes=2)
        return samples

    # def sample_and_logprobs(self,key,numsamples,N):
    #     """Sample with log probs from the model for a given system size N and a number of samples `numsamples`"""
    #     inputs = jnp.zeros((numsamples,self.output_dim), dtype = jnp.float64)
    #     hidden_states = self.cell.initialize_carry(jax.random.key(1), inputs.shape)
    #     # hidden_states = jnp.zeros((numsamples,self.num_hidden_units), dtype = jnp.float64)
    #     samples_onehot = jnp.zeros((numsamples,N,self.output_dim), dtype = jnp.float64)
    #     samples = jnp.zeros((numsamples,N), dtype = jnp.float64)
    #     cond_log_probs = jnp.zeros((numsamples,N,2))
    #     keys = jax.random.split(key, N) #pre-generate keys to get more randomness

    #     for n in range(N):
    #         hidden_states,inputs = self.cell(hidden_states,inputs)  # apply each layer
    #         inputs = self.dense(inputs)
    #         cond_log_probs = cond_log_probs.at[:,n].set(nn.log_softmax(inputs, axis=-1))
    #         samples = samples.at[:,n].set(jax.random.categorical(key=keys[n], logits=inputs))
    #         inputs = jax.nn.one_hot(samples[:,n], num_classes=2)
    #         samples_onehot = samples_onehot.at[:,n].set(inputs)

    #     log_probabilities = jnp.sum(cond_log_probs * samples_onehot, axis = (1,2))
    #     return samples,log_probabilities