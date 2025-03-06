#!/bin/bash
#SBATCH -c 4

nAgents=5
nGoals=5
nGraphs=1200
envBound=6.0
minDist=1.0

python pybullet_env/simulator/gnn/create_dataset.py --nAgents $nAgents --nGoals $nGoals --nGraphs $nGraphs --envBound $envBound --minDist $minDist

