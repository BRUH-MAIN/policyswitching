#!/bin/bash


#SBATCH --partition=workq              # Partition to submit to (Do Not Change)
#SBATCH --time=24:00:00                # Walltime (format: HH:MM:SS)
#SBATCH --gres=gpu:1                   # Request 1 (or 2) GPU if needed (this is per node)
#SBATCH --nodelist=asaicomputenode02   # Specify node(s) by name
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Total memory per node

## RUNNING A PYTHON SCRIPT ##

# Load Conda environment setup if necessary (path to conda)
#source /path/to/miniconda3/etc/profile.d/conda.sh  # Adjust this path as needed

# Activate the Conda environment
#conda activate my_env_name

# Run the Python script
python3 /dist_home/gputest/python.py 

# RUN JUPYTER NOTEBOOK

#node=$(hostname -s)

#jupyter-notebook --no-browser --port=8888 --ip=${node}

