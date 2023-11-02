#!/bin/sh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=60G 

python main.py --config-name=ctrl prompt.prompt="turtle" init.mesh="turt.glb" mesh="turt.glb"