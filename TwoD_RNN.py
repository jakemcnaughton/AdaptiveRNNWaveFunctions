import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as nn
jax_dtype = jnp.float64

def matrix_init(key, shape, dtype=jax_dtype, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization

class TwoDRNN(nn.Module):
    """
    """
    d_hidden: int  # hidden state dimension
    d_model: int
    RNNcell_type: str = "GRU"

    def setup(self):
      # Initialize the GRU cell with the specified number of hidden units
      if self.RNNcell_type == "GRU":
        self.cell = nn.GRUCell(
            name='gru_cell',
            features=self.d_hidden,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jax_dtype
        )
      elif self.RNNcell_type == "LSTM":
        self.cell = nn.OptimizedLSTMCell(
            name='lstm_cell',
            features=self.d_hidden,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jax_dtype
        )
      elif self.RNNcell_type == "Vanilla":
        self.cell = nn.SimpleCell(
            name='vanilla_cell',
            features=self.d_hidden,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jax_dtype
        )
      else:
        raise ValueError("Invalid RNN cell type")

      # self.U = self.param(
      #       "U",
      #       jax.nn.initializers.glorot_uniform(),
      #       (self.d_hidden*2, self.d_hidden),
      #   )

      self.U1 = self.param(
            "U1",
            jax.nn.initializers.glorot_uniform(),
            (self.d_hidden, self.d_hidden),
        )

      self.U2 = self.param(
            "U2",
            jax.nn.initializers.glorot_uniform(),
            (self.d_hidden, self.d_hidden),
        )

    def __call__(self, inputs, hidden_states):
        """Forward pass of a 2DRNN"""

        if isinstance(inputs, tuple):
          concatenate_inputs = jnp.concatenate(inputs, axis = -1)
        else:
          concatenate_inputs = inputs

        new_hidden_state = jax.vmap(lambda u: u @ self.U1)(hidden_states[0])
        new_hidden_state += jax.vmap(lambda u: u @ self.U2)(hidden_states[1])

        new_hidden_state,_ = self.cell(new_hidden_state, concatenate_inputs)

        return new_hidden_state, new_hidden_state


class SequenceLayer(nn.Module):
    """Single RNN layer"""
    # Combining RNN for Softmax

    RNN: TwoDRNN  # 2dRNN module
    d_model: int  # model size

    def setup(self):
        """Initializes the RNN"""
        self.seq = self.RNN
        self.out1 = nn.Dense(self.d_model)
        self.out2 = nn.Dense(self.d_model)

    def __call__(self, inputs, hidden_states):
        x, new_hidden_state = self.seq(inputs, hidden_states)  # call LRU
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        # return inputs[0] + inputs[1] + x, new_hidden_state  # skip connection
        return x, new_hidden_state  # no skip connection

class StackedRNNModel(nn.Module):
    """Encoder containing several SequenceLayer"""

    d_model: int
    d_hidden: int
    n_layers: int
    RNNcell_type: str = "Vanilla"

    def setup(self):
        self.layers = [
            SequenceLayer(
                RNN=TwoDRNN(d_model = self.d_model, d_hidden = self.d_hidden, RNNcell_type = self.RNNcell_type),
                d_model=self.d_model,
            )
            for _ in range(self.n_layers)
        ]
        self.decoder = nn.Dense(2)
        self.dense_phase = nn.Dense(2)

    def generate_zigzag_path(self, Nx, Ny):
       return [(i if j % 2 == 0 else Ny - 1 - i, j) for j in range(Ny) for i in range(Nx)]

    def __call__(self, samples):
      """Sequential call of the model"""
      numsamples, Nx, Ny = samples.shape
      hidden_states = [[[jnp.zeros((numsamples,self.d_hidden), dtype = jax_dtype) for ny in range(-1,Ny+1)] for nx in range(-1,Nx+1)] for _ in range(self.n_layers)]
      inputs = [[[jnp.zeros((numsamples,2), dtype = jax_dtype) if k == 0 else jnp.zeros((numsamples,self.d_model), dtype = jax_dtype) for ny in range(-1, Ny+1) ] for nx in range(-1, Nx+1)] for k in range(self.n_layers+1)]
      samples_onehot = jnp.zeros((numsamples,Nx,Ny,2), dtype = jax_dtype)
      cond_log_probs = jnp.zeros((numsamples,Nx,Ny,2), dtype = jax_dtype)
      phases = jnp.zeros((numsamples,Nx,Ny,2), dtype = jax_dtype)

      zigzag_path = self.generate_zigzag_path(Nx, Ny)

      for nx,ny in zigzag_path:
          for layer_index,layer in enumerate(self.layers):
              if layer_index == 0:
                x1 = inputs[layer_index][nx-(-1)**ny][ny]
                x2 = inputs[layer_index][nx][ny-1]
              else:
                x1 = inputs[layer_index][nx][ny]
              h1 = hidden_states[layer_index][nx-(-1)**ny][ny]
              h2 = hidden_states[layer_index][nx][ny-1]
              inputs[layer_index+1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1,x2), (h1, h2))  # apply each layer
          x = self.decoder(inputs[-1][nx][ny])
          
          cond_log_probs = cond_log_probs.at[:,nx,ny].set(nn.log_softmax(x, axis=-1))
          phases = phases.at[:,nx,ny].set(jnp.pi*nn.soft_sign(self.dense_phase(x)))

          inputs[0][nx][ny] = jax.nn.one_hot(samples[:,nx,ny], num_classes=2)
          samples_onehot = samples_onehot.at[:,nx,ny].set(inputs[0][nx][ny])
          
      log_probabilities = jnp.sum(cond_log_probs * samples_onehot, axis = (1,2,3))
      overall_phases = jnp.sum(phases * samples_onehot, axis = (1,2,3))

      return 0.5*log_probabilities + 1j*overall_phases


    def sample(self,key,numsamples,Nx,Ny):
        """Sample from the model for a given system size Nx,Ny and a number of samples `numsamples`"""
        samples = jnp.zeros((numsamples,Nx, Ny))
        hidden_states = [[[jnp.zeros((numsamples,self.d_hidden), dtype = jax_dtype) for ny in range(-1,Ny+1)] for nx in range(-1,Nx+1)] for _ in range(self.n_layers)]
        inputs = [[[jnp.zeros((numsamples,2), dtype = jax_dtype) if k == 0 else jnp.zeros((numsamples,self.d_model), dtype = jax_dtype) for ny in range(-1, Ny+1) ] for nx in range(-1, Nx+1)] for k in range(self.n_layers+1)]

        zigzag_path = self.generate_zigzag_path(Nx, Ny)

        keys = jax.random.split(key, Nx*Ny)

        for nx,ny in zigzag_path:
            for layer_index,layer in enumerate(self.layers):
                if layer_index == 0:
                  x1 = inputs[layer_index][nx-(-1)**ny][ny]
                  x2 = inputs[layer_index][nx][ny-1]
                else:
                  x1 = inputs[layer_index][nx][ny]
                h1 = hidden_states[layer_index][nx-(-1)**ny][ny]
                h2 = hidden_states[layer_index][nx][ny-1]
                inputs[layer_index+1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1,x2), (h1, h2))  # apply each layer
            x = self.decoder(inputs[-1][nx][ny])
            samples = samples.at[:,nx,ny].set(jax.random.categorical(key=keys[ny*Nx+nx], logits=nn.log_softmax(x, axis=-1)))
            inputs[0][nx][ny] = jax.nn.one_hot(samples[:,nx,ny], num_classes=2)

        return samples
