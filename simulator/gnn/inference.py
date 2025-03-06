import torch
import dgl
from modelv2 import NonLinearModel
from create_dataset import create_graph, random_init, is_connected
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

device = torch.device("cpu")

def load_model(checkpoint_path, attr_dim, max_edges, L):
    model = NonLinearModel(attr_dim, max_edges, L)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    return model

def generate_problem_graph(nAgents, commRadius, attr_dim, minDist, envBound):
    agentsPos, goalsPos = random_init(nAgents, nAgents, minDist, envBound)
    graph, cost_matrix = create_graph(agentsPos, goalsPos, nAgents, commRadius, attr_dim)
    adj_matrix = graph.adj(etype=('agent', 'communicates', 'agent')).to_dense()
    if not is_connected(adj_matrix):
        raise ValueError("Generated graph is not connected.")
    
    # Ensure the number of features matches the number of nodes
    src_ids = torch.arange(nAgents)  # Assuming nAgents is the number of nodes
    graph.nodes['goal'].data['src_ids'] = src_ids
    
    return graph

def run_inference(model, graph):
    with torch.no_grad():
        h, edges = model.forward(graph)
    return h, edges

def main():
    checkpoint_path = "pybullet_env/simulator/gnn/models/old_models/model_10agents_env4m_comvar_maxedges5_3conv_modelv2_lastepoch.pt"
    attr_dim = 16
    max_edges = 5
    L = 5
    nAgents = 5
    commRadius = 3
    minDist = 1.0
    envBound = 4.0

    model = load_model(checkpoint_path, attr_dim, max_edges, L)
    bijective_count = 0
    num_trials = 500

    for _ in range(num_trials):
        while True:
            try:
                graph = generate_problem_graph(nAgents, commRadius, attr_dim, minDist, envBound)
                break
            except ValueError:
                continue
        h, edges = run_inference(model, graph)

        # Deciphering the results
        assignments = []
        for i in range(len(h)):
            agent = edges[0][i].item()
            goal = edges[1][i].item()
            likelihood = h[i].item()
            assignments.append((agent, goal, likelihood))

        # Find the highest likelihood assignment for each agent
        best_assignments = {}
        for assignment in assignments:
            agent, goal, likelihood = assignment
            if agent not in best_assignments or likelihood > best_assignments[agent][1]:
                best_assignments[agent] = (goal, likelihood)

        # Check if the assignments are bijective
        assigned_goals = set()
        bijective = True
        for goal, _ in best_assignments.values():
            if goal in assigned_goals:
                bijective = False
                # plt.figure(figsize=(8, 6))
                # for agent, (goal, likelihood) in best_assignments.items():
                #     plt.plot([0, 1], [agent, goal], marker='o', markersize=10)
                # plt.title('Best Assignments')
                # plt.xlabel('Agents -> Goals')
                # plt.xticks([0, 1], ['Agents', 'Goals'])
                # plt.yticks(range(nAgents))
                # plt.ylabel('Index')
                # plt.show()
                break
            assigned_goals.add(goal)
        
        if bijective:
            bijective_count += 1

    bijective_percentage = (bijective_count / num_trials) * 100
    print(f"Percentage of bijective solutions: {bijective_percentage:.2f}%")


if __name__ == '__main__':
    main()