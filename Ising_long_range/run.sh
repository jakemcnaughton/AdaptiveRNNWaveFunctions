#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=Adap_clusterN64
#SBATCH --mail-user=mhibatallah@uwaterloo.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Outputs/%x_output_%A_%a.out    # Dynamic output file name using job name
#SBATCH --error=Outputs/%x_error_%A_%a.err      # Dynamic error file name using job name
#SBATCH --array=0                  # Run different jobs
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=1            # Number of CPUs
#SBATCH --gres=gpu:a100_80g:1            # Number of GPUs (per node)
#SBATCH --mem=50G                    # Memory per node

# Define the values for N and MODEL_TYPE
n_values=(64)                 # Different N values
model_types=("GRU")        # Different model types
# Get the index for the job
index=$SLURM_ARRAY_TASK_ID

# Calculate the index for N and MODEL_TYPE
n_index=$((index % 5))               # Cycles through N values
model_type_index=$((index / 5))      # Cycles through MODEL_TYPE

# Get the actual N and MODEL_TYPE values based on the index
N=${n_values[$n_index]}
MODEL_TYPE=${model_types[$model_type_index]}
NUMSTEPS=100000

# Define the NAME variable based on the model type and N value
NAME="Adaptive_Epoch${NUMSTEPS}_${MODEL_TYPE}_N${N}"
# NAME="Traditional_Epoch${NUMSTEPS}_${MODEL_TYPE}_N${N}"


source ../ExoticRNNs/jax/bin/activate

# Run the Python script with the selected parameters
python3 train_adaptive_cRNN.py --N $N --MODEL_TYPE $MODEL_TYPE --NAME $NAME --NUMSTEPS 100000

# python3 train_traditional_cRNN.py --N $N --MODEL_TYPE $MODEL_TYPE --NAME $NAME --NUMSTEPS 100000