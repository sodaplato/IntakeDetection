#!/bin/bash

#SBATCH --job-name=UMicroMamba-IntakeDetection
#SBATCH --partition=luc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err
module load anaconda3-2024.10-1 
source activate meb_env
cd /seu_share2/home/213222097/IntakeDetection
sbatch submit_tridet.sh experiment261016/intake_tridet.yaml