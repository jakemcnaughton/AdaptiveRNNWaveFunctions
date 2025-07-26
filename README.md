# Adaptive RNN Wave Functions

A JAX-based implementation of **Adaptive Recurrent Neural Network (RNN) Wave Functions / Quantum States** for ground state problems using Variational Monte Carlo (VMC).

## 🚀 Overview

This project introduces **Adaptive RNN Wave Functions** for variational quantum Monte Carlo (VMC) simulations. Instead of fixing the architecture size from the start, the model **grows in complexity during training**, allowing better navigation of rugged optimization landscapes, reducing training runtime and even improving accuracy.

## Usage

### `1D_TFIM`
To run the **Adaptive** 1D RNN on the 1D TFIM model use 
```
python train_adaptive.py
```
and for the **Static** 1D RNN on the 1D TFIM model
```
python train_traditional.py
```
To learn more about the arguments that can be specified add the ``` --help ``` flag:

```
python train_adaptive.py --help
python train_traditional.py --help
```

### `2D_Heisenberg`
To run the **Adaptive** 2D RNN on 2D Heisenberg model use 
```
python 2D_Adaptive.py
```
for the **Early Stopping** variant of the **Adaptive** 2D RNN on the 2D Heisenberg model, use
```
python 2D_Traditional.py
```
and for the **Static** 2D Heisenberg model
```
python 2D_Adaptive_EarlyStopping.py
```

### `Ising_long_range`

To run the **Adaptive** 1D RNN on the 1D long-range TFIM, you can run
```
python train_adaptive.py
```
and for the **Static** 1D RNN, use
```
python train_traditional.py
```

### `Cluster_state`
To run the **Adaptive** 1D complex RNN on the 1D Cluster state Hamiltonian, you can run
```
python train_adaptive_cRNN.py
```
and for the **Static** 1D RNN, use
```
python train_traditional_cRNN.py
```

The hyperparameters can be adjusted by passing arguments in the previous Python commands.

## License
The license of this work is derived from the BSD-3-Clause license. Ethical clauses are added to promote good uses of this code.

## Citing
```bibtex
@article{AdaptiveRNNs,

}
```
