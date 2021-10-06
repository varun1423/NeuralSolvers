#!/bin/bash -l

#SBATCH -p alpha
#SBATCH -t 10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -o output.out
#SBATCH -e Error.err
#SBATCH --job-name="density"
#SBATCH -A p_da_aipp


conda activate /home/vash965b/user-kernel/my-kernel

cd /home/vash965b/Thesis/NeuralSolvers/Phase_Space_Interpolation/boolean_density/

srun python main.py

