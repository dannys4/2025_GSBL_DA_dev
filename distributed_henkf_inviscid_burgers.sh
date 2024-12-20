#!/bin/bash

# Slurm sbatch options
#SBATCH -n 150

# Initialize the module command first source
source /etc/profile

# Load Julia Module
module load julia/1.10.1

# Call your script as you would from the command line
julia /home/gridsan/mleprovost/julia/HierarchicalDA.jl/notebooks/inviscid_burgers/distributed_inviscid_burgers_EnKF_1.jl
