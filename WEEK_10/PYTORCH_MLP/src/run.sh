#!/bin/bash
#SBATCH --job-name=IA_04         # nombre del job
#SBATCH --nodes=1                # cantidad de nodos
#SBATCH --ntasks=1               # cantidad de tareas
#SBATCH --cpus-per-task=1        # cpu-cores por task 
#SBATCH --mem=16G                 # memoria total por nodo
#SBATCH --gres=gpu:1             # numero de gpus por nodo
#SBATCH --output=IA_04.out       # archivo de salida

module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate BT

python -W ignore main.py

conda deactivate