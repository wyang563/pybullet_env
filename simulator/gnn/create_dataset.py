import dgl
import torch
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

def random_init(nAgents,nGoals,minDist,envBound):
    goalsPos = torch.zeros(2,nGoals)
    agentsPos = torch.zeros(2,nAgents)
    for a in range(nAgents):
        checkPos = False
        while checkPos == False:
            x0 = random.randint(0,envBound*10)*0.1
            y0 = random.randint(0,envBound*10)*0.1
            checkPos = checkMinDist(nAgents,agentsPos,x0,y0,minDist)
        agentsPos[0,a] = x0
        agentsPos[1,a] = y0
    for g in range(nGoals):
        checkGoals = False
        while checkGoals == False:
            goalx = random.randint(0,envBound*10)*0.1
            goaly = random.randint(0,envBound*10)*0.1
            checkGoals = checkMinDist(nGoals,goalsPos,goalx,goaly,minDist)
        goalsPos[0,g] = goalx
        goalsPos[1,g] = goaly
    return agentsPos,goalsPos

def checkMinDist(nAgents, agentsPos,randx,randy,minDist):
    for i in range(nAgents):
        dist = (randx-agentsPos[0,i])**2 + (randy-agentsPos[1,i])**2
        if dist < minDist**2:
            return False
    return True

def computeCommEdges(nAgents, agentsPos, commRadius):
    u = torch.tensor([],dtype=torch.int32)
    v = torch.tensor([],dtype=torch.int32)
    for i in range(nAgents):
        for j in range(nAgents):
            dist = (agentsPos[0,i]-agentsPos[0,j])**2 + (agentsPos[1,i]-agentsPos[1,j])**2
            if dist < commRadius**2:
                u = torch.cat((u,torch.tensor([i],dtype=torch.int32)))
                v = torch.cat((v,torch.tensor([j],dtype=torch.int32)))
    return u,v

def create_graph(agentsPos, goalsPos, nAgents, nGoals, commRadius, attr_dim):
    cost_matrix = torch.zeros((nAgents, nGoals))
    for i in range(nAgents):
        for j in range(nGoals):
            cost_matrix[i,j] = (agentsPos[0,i]-goalsPos[0,j])**2+(agentsPos[1,i]-goalsPos[1,j])**2
            
    u = torch.arange(nAgents).repeat_interleave(nGoals)
    v = torch.arange(nGoals).repeat(nAgents)

    # u = torch.tensor([0 for _ in range(nAgents)])
    # v = torch.tensor([i for i in range(nGoals)])
    # for i in range(1,nAgents):
    #     u = torch.cat((u,torch.tensor([i for k in range(nAgents)])))
    #     v = torch.cat((v,torch.tensor([k for k in range(nAgents)])))
    u_com,v_com = computeCommEdges(nAgents,agentsPos,commRadius)
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
    return graph, cost_matrix

def hungarian_algo(cost_mat):
    return linear_sum_assignment(cost_mat)

def compute_cost_matrix(dist_edges_graph,edges,n_Agents, n_Goals):
    cost_matrix = torch.zeros(n_Agents, n_Goals)
    for i in range(dist_edges_graph.shape[0]):
        agent_id = edges[0][i]
        goal_id = edges[1][i]
        cost_matrix[agent_id,goal_id]=dist_edges_graph[i]
        if cost_matrix[agent_id,goal_id]==0.:
            # print("zero problem handled")
            cost_matrix[agent_id,goal_id]=0.0001
    return cost_matrix

def is_connected(adj_matrix):
    An = torch.clone(adj_matrix)
    for k in range(An.shape[0]):
        An = torch.matmul(An,adj_matrix)
    for i in range(An.shape[0]):
        for j in range(An.shape[1]):
            if An[i,j]==0:
                return False
    return True

def compute_labels(dist_edges_graph, edges, n_Agents, n_Goals):
    cost_matrix = compute_cost_matrix(dist_edges_graph,edges,n_Agents,n_Goals)
 
    # deal with case where n_Agents > n_Goals
    if n_Agents > n_Goals:
        # pad cost matrix with dummy goals
        padding = torch.full((n_Agents,n_Agents-n_Goals), float('inf'))
        cost_matrix = torch.cat((cost_matrix,padding), dim=1)

    np_cost_matrix = cost_matrix.numpy()
    row_inds, col_inds = hungarian_algo(np_cost_matrix.copy())
    labels = torch.zeros((n_Agents, n_Goals))
    labels[row_inds, col_inds] = 1
    return labels.flatten()

def main():
    nAgents = 5 
    nGoals = 8 
    # nGraphs = 1400
    nGraphs = 300 
    # commRadius_list = [1.,1.3,1.5,1.75,2.,2.2,2.4,2.6,2.8,3.,3.25,3.6,4.0] # this was for the dataset of 10 agents
    # commRadius_list = [2.,2.5,3.,3.5,4.,4.5,5.,5.5]
    commRadius_list = [1.5,2.0,2.5,3.,3.5,4.,4.5,5.,5.5,6.0] # this is for the dataset of 20 agents
    # commRadius_list = [2.0,2.5,3.,3.5,4.,4.5,5.,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5] # this is for the dataset of 50 agents
    # commRadius_list = [1.5,1.75,2.0,2.25,2.5,2.75] # this is for 5 agents
    # commRadius_list = [1.5,2.0,2.5,3.0,3.5,4.0] # this is for 10 agents
    # commRadius_list = [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0] # this is for 15 agents
    minDist = 1.0
    envBound = 6.0
    attr_dim = 16
    
    graph_list = []
    cost_matrices = []
    print('Creation of the graphs...')
    for commRadius in commRadius_list:
        print('comRadius = ', commRadius)
        g=0
        while g < nGraphs:
            agentsPos, goalsPos = random_init(nAgents, nGoals, minDist, envBound)
            graph, cost_matrix = create_graph(agentsPos, goalsPos, nAgents, nGoals, commRadius, attr_dim)
            adj_matrix = graph.adj(etype=('agent','communicates','agent')).to_dense()
            if is_connected(adj_matrix):
                graph_list.append(graph)
                cost_matrices.append(cost_matrix)
                print(str(100*g/nGraphs)+'%',end="\r")
                g+=1
    print('Graphs built.')
    
    # size of the dataset
    print('Number of graphs in the dataset : ',len(graph_list))
        
    # Compute labels
    all_labels=torch.tensor([])
    print('Compute labels...')
    for g in range(len(graph_list)):
        print(str(100*g/len(graph_list))+'%',end="\r")
        dist_edges_graph = graph_list[g].edges[('agent','assigns','goal')].data['he']
        edges = graph_list[g].edges(etype=('agent','assigns','goal'))
        labels = compute_labels(dist_edges_graph, edges, nAgents, nGoals)
        if labels.sum() != nAgents:
            print(labels.reshape(nAgents,nGoals))
            print(g)
            assert False
        graph_list[g].edges[('agent','assigns','goal')].data['label'] = labels
        all_labels = torch.cat((all_labels,labels.unsqueeze(0)),dim=0)
    print('Labels done.')
        
    # Print histogram of densities

    # densities = []
    # for i in range(len(graph_list)):
    #     graph = graph_list[i]
    #     comm_adj = graph.adj(etype=('agent','communicates','agent')).to_dense()
    #     density = round((comm_adj.sum().item()-nAgents)/(nAgents*nAgents-nAgents),2)*100
    #     densities.append(density)
    # counts, bins = np.histogram(densities,range=(0,100))
    # print('histogram  of densities : ')
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.show()
    # print("mean density = ", np.mean(densities))
    
    # save the dataset
    dgl.save_graphs(f'pybullet_env/simulator/gnn/data/train/datasetTRAIN_{nAgents}agents_{nGoals}goals_env{envBound}m_commRadiusVar_AllConnected_{nGraphs}.dgl',graph_list,labels={"labels": all_labels})
    
if __name__ == '__main__':
	main()