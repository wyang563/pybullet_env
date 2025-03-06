import numpy as np
import dgl
import torch
from pybullet_env.simulator.gnn.models.modelv2 import NonLinearModel

nAgents = 10
nGoals = 10
attr_dim = 16

def compute_comm_edges():
    u_comm = []
    v_comm = []
    for i in range(5):
        for j in range(5):
            if j in [(i + k) % 10 for k in range(10)]:
                u_comm.append(i)
                v_comm.append(j)
    return torch.tensor(u_comm, dtype=torch.int32), torch.tensor(v_comm, dtype=torch.int32)

def get_assignment_from_pred(edges, pred, nAgents, nGoals):
    # Ensure that the predicted tensor size matches the number of edges
    assert pred.shape[0] == edges[0].shape[0]
    
    # Combine edge index tensors into a single tensor of shape (#edges, 2)
    # edges[0] are the agent indices and edges[1] are the goal indices.
    edges_tensor = torch.stack((edges[0], edges[1]), dim=1)
    
    # Create an empty assignment matrix of shape (nAgents, nGoals)
    assignment = torch.zeros(nAgents, nGoals)
    
    # Iterate over each edge in the graph
    for i in range(pred.shape[0]):
        agent_id = edges_tensor[i, 0]
        goal_id  = edges_tensor[i, 1]
        
        # Only update if the indices are in range.
        if 0 <= agent_id < nAgents and 0 <= goal_id < nGoals:
            assignment[agent_id, goal_id] = pred[i]
    
    # convert 2d array to 1d assignments
    hard_assignment = (assignment == assignment.max(dim=1, keepdim=True)[0]).view_as(assignment).to(torch.float)
    final_assignments = torch.argmax(hard_assignment, dim=1) 
    return final_assignments 

if __name__ == "__main__":
    cost_matrix = np.array([[2.23963708, 2.03640788, 2.16727731, 1.54707136, 1.35109456, 1.88882036,
        0.931646, 1.65238104, 1.11914673, 0.89040221],
        [1.79594387, 1.79692352, 2.14660754, 1.10450859, 1.24109899, 2.03460481,
        0.57473035, 1.27106066, 1.19066241, 0.44001854],
        [1.50636167, 1.90031656, 2.5742954, 0.94192229, 1.65045007, 2.54289041,
        0.80524289, 0.73028576, 1.74604548, 0.20697503],
        [1.48205069, 2.31483107, 3.09540641, 1.19576783, 2.17766295, 3.10044147,
        1.30095665, 0.37554048, 2.32497166, 0.72640321],
        [1.82495627, 2.8268153, 3.61681009, 1.7395984, 2.70532679, 3.59986897,
        1.86187067, 0.69767349, 2.81569903, 1.18477349],
        [2.23858912, 3.20645276, 3.98352931, 2.06943778, 3.0095463, 3.88996081,
        2.20763448, 1.10370481, 3.12493194, 1.46070539],
        [2.47301857, 3.30966043, 4.0, 2.21957348, 3.02555723, 3.895386,
        2.18196882, 1.32406049, 3.00842654, 1.41831314],
        [3.05614658, 2.49148847, 3.63537732, 2.08719121, 2.65572711, 3.4562682,
        1.947021, 1.41661527, 2.55658857, 1.0913566],
        [2.45705572, 2.75720475, 3.1880202, 2.00012033, 2.23881084, 2.95114803,
        1.52257068, 1.55569081, 2.09974349, 0.89097639],
        [2.44226988, 2.40395671, 2.64558643, 1.82324109, 1.80703211, 2.41023059,
        1.67258458, 1.2504444, 1.53093508, 0.85353853]])

    u = torch.arange(nAgents).repeat_interleave(nGoals)
    v = torch.arange(nGoals).repeat(nAgents) 
    u_com, v_com = compute_comm_edges()
    graph_data = {('agent', 'assigns', 'goal'): (u, v), ('goal', 'assigns', 'agent'): (v, u),('agent', 'communicates', 'agent'): (u_com, v_com)}
    graph = dgl.heterograph(graph_data, num_nodes_dict={'agent': nAgents, 'goal': nGoals}, idtype=torch.int32)
    graph.nodes['agent'].data['hv']= torch.zeros(nAgents,attr_dim)
    graph.nodes['goal'].data['hv']=torch.zeros(nGoals,attr_dim)

    num_edges = graph.num_edges(('agent','assigns','goal'))
    dist_edges = torch.zeros(num_edges,1)
    for i in range(num_edges):
        agent_id = graph.edges(etype=('agent','assigns','goal'))[0][i]
        goal_id = graph.edges(etype=('agent','assigns','goal'))[1][i]
        dist_edges[i,0] = cost_matrix[agent_id,goal_id] 

    graph.edges[('agent','assigns','goal')].data['he'] = dist_edges
    graph.edges[('goal','assigns','agent')].data['he'] = dist_edges

    # convert graph to dgl format
    with torch.no_grad():
        gnn_model = NonLinearModel(attr_dim=16,max_edges=5,L=3)
        gnn_model.load_state_dict(torch.load("pybullet_env/simulator/gnn/models/old_models/model_10agents_env4m_comvar_maxedges5_3conv_modelv2.pt"))
        pred, edges = gnn_model(graph)  
        print(pred)
        print(edges)
        assignments = get_assignment_from_pred(edges, pred, nAgents, nGoals)
        print(assignments)