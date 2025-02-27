import dgl
import numpy as np
import matplotlib.pyplot as plt
from dgl.data import DGLDataset
import torch

def check_disconnected_goals(nAgents,edges_goals):
    connected_goals = [edges_goals[i].item()%nAgents for i in range(edges_goals.shape[0])]
    for i in range(nAgents):
        if i not in connected_goals:
            return True
    return False

class GoalAssignmentDataset(DGLDataset):
    def __init__(self,name,filename,max_edges):
        self.filename = filename
        self.max_edges = max_edges
        super().__init__(name=name)
        
    def keep_best_edges(self,glist,max_edges):
        for g,graph in enumerate(glist):
            # graph = graph.to(device)
            n_agents = graph.number_of_nodes()//2
            if max_edges>n_agents:
                max_edges = n_agents
            dist_edges = graph.edges[('goal','assigns','agent')].data['he']
            # edges_to_remove = torch.tensor([],dtype=torch.int32,device=device)
            edges_to_remove = torch.tensor([],dtype=torch.int32)
            for n in range(n_agents): # we want to select the best edges per goal, and not per agent, but the edges are ordered like (0,0),(1,0),(2,0) etc. The first indexes are the goals indexes. We want to compare along the same goal indexes.
                # select the good indexes to compare
                # indexes = torch.tensor([i for i in range(dist_edges.shape[0])],device=device)
                indexes = torch.tensor([i for i in range(dist_edges.shape[0])])
                selected_indexes = indexes[indexes%n_agents==n]
                # select the corresponding values in dist_edges
                mask = torch.zeros(dist_edges.shape[0], dtype=bool)
                # mask = torch.zeros(dist_edges.shape[0], dtype=bool,device=device)
                mask[selected_indexes] = True
                dist_edges_selected = dist_edges[mask]
                # extract the largest values (bad ones to be removed)
                _,ids = torch.topk(dist_edges_selected,n_agents-max_edges,dim=0,largest=True)
                bad_ids = selected_indexes[ids].flatten().to(torch.int32)
                edges_to_remove = torch.cat([edges_to_remove,bad_ids])
            
            graph = dgl.remove_edges(graph, edges_to_remove,etype=('goal','assigns','agent'))
            glist[g] = dgl.remove_edges(graph, edges_to_remove,etype=('agent','assigns','goal'))
        return glist
    
    def keep_best_edges_goals(self,glist,max_edges):
        for g, graph in enumerate(glist):
            nAgents = graph.num_nodes('agent')
            if max_edges>nAgents:
                max_edges = nAgents
                
            edges = graph.edges(etype=('goal','assigns','agent'))
            he = graph.edges[('goal','assigns','agent')].data['he']
            edges_ids = torch.tensor([k for k in range(edges[0].shape[0])])
            edges_to_remove = torch.tensor([],dtype=torch.int32)
            
            for j in range(nAgents):
                selected_ids = edges_ids[edges[0]%nAgents==j]
                mask = torch.zeros(edges[0].shape[0], dtype=bool)
                mask[selected_ids] = True
                he_j = he[mask]
                _,ids = torch.topk(he_j,nAgents-max_edges,dim=0,largest=True)
                ids_to_remove = selected_ids[ids].flatten().to(torch.int32)
                edges_to_remove = torch.concat((edges_to_remove,ids_to_remove))
            
            graph = dgl.remove_edges(graph, edges_to_remove, ('agent','assigns','goal'))
            glist[g] = dgl.remove_edges(graph, edges_to_remove, ('goal','assigns','agent'))
        
        return glist
    
    def keep_best_edges_agents(self,glist,max_edges):
        disconnected_graphs = []
        for g, graph in enumerate(glist):
            nAgents = graph.num_nodes('agent')
            nGoals = nAgents
            if max_edges>nGoals:
                max_edges = nGoals
                
            edges = graph.edges(etype=('agent','assigns','goal'))
            he = graph.edges[('agent','assigns','goal')].data['he']
            edges_ids = torch.tensor([k for k in range(edges[0].shape[0])])
            edges_to_remove = torch.tensor([],dtype=torch.int32)
            
            for i in range(nAgents):
                selected_ids = edges_ids[edges[0]==i]
                mask = torch.zeros(edges[0].shape[0], dtype=bool)
                mask[selected_ids] = True
                he_j = he[mask]
                if he_j.shape[0]>max_edges:
                    _,ids = torch.topk(he_j,he_j.shape[0]-max_edges,dim=0,largest=True)
                    ids_to_remove = selected_ids[ids].flatten().to(torch.int32)
                    edges_to_remove = torch.concat((edges_to_remove,ids_to_remove))
            
            graph = dgl.remove_edges(graph, edges_to_remove, ('agent','assigns','goal'))
            glist[g] = dgl.remove_edges(graph, edges_to_remove, ('goal','assigns','agent'))
            
            # check disconnected goals: we want to remove these possibilities in the training dataset
            edges_goals = glist[g].edges(etype=('agent','assigns','goal'))[1]
            if check_disconnected_goals(nAgents,edges_goals):
                disconnected_graphs.append(glist[g])
        for graph in disconnected_graphs:
            glist.remove(graph)
        return glist
            

    def process(self):
        glist,label_dict = dgl.load_graphs(self.filename)
        # self.labels = []
        if self.max_edges!=None:
            # glist = self.keep_best_edges_goals(glist,self.max_edges)
            # glist = self.keep_best_edges_agents(glist,self.max_edges)
            glist = self.keep_best_edges(glist,self.max_edges)
            
        # for g in range(len(glist)):
        #     n_agents = glist[g].number_of_nodes()//2
        #     comm_adj = glist[g].adj(etype=('agent','communicates','agent')).to_dense()
        #     glist[g].nodes['goal'].data['comm_adj']= comm_adj.unsqueeze(0).repeat(n_agents,1,1)
        #     agent_degrees = glist[g].in_degrees(glist[g].nodes('agent'),etype = ('agent','communicates','agent'))
        #     glist[g].nodes['goal'].data['degrees'] = agent_degrees.unsqueeze(0).repeat(n_agents,1)
        #     self.labels.append(label_dict[str(g)+"labels"])
        self.graphs = glist
        self.labels=label_dict["labels"]

    def __getitem__(self, i):
        return self.graphs[i],self.labels[i,:]

    def __len__(self):
        return len(self.graphs)

    def histogram_densities(self):
        nGraphs = len(self.graphs)
        densities = []
        for i in range(nGraphs):
            graph,_ = self.__getitem__(i)
            n_agents = graph.num_nodes('agent')
            comm_adj = graph.adj(etype=('agent','communicates','agent')).to_dense()
            density = round((comm_adj.sum().item()-n_agents)/(n_agents*n_agents-n_agents),2)*100
            densities.append(density)
        counts, bins = np.histogram(densities)
        print('mean density = ',np.mean(densities))
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()

if __name__ == "__main__":
    dataset = GoalAssignmentDataset('goal_assignment_5agents_comVar','pybullet_env/simulator/gnn/data/train/datasetTRAIN_5agents_env6m_commRadiusVar_AllConnected_100.dgl',max_edges=5)
    dataset.histogram_densities()
    print("size of the dataset = ", dataset.__len__())
    example_graph,_ = dataset.__getitem__(0)
    print(example_graph.edges(etype=('agent','assigns','goal')))
    print(example_graph.edges[('agent','assigns','goal')].data['he'].shape)
