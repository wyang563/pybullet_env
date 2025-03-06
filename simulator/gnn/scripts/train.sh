#!/bin/bash
#SBATCH -c 4

nAgents=5
nGoals=5
nEpochs=50
max_edges=5
dataset="pybullet_env/simulator/gnn/data/train/datasetTRAIN_5agents_5goals_env6.0m_commRadiusVar_AllConnected_1200.dgl"
batch_size=100
L=10

python -m simulator.gnn.train  --dataset $dataset --nEpochs $nEpochs --nAgents $nAgents --nGoals $nGoals --max_edges $max_edges --batch_size $batch_size --rounds $L --debug true 
