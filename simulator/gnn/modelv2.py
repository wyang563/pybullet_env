import torch
import torch.nn as nn
import copy
import dgl.function as fn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

def retrieve_comm_adj(comm_adj,agents_ids):
    sub_adj = torch.index_select(comm_adj, 0, agents_ids)
    sub_adj = torch.index_select(sub_adj, 1, agents_ids)
    sub_degrees = torch.sum(sub_adj,dim=1).unsqueeze(1).repeat(1,sub_adj.shape[0])
    return torch.div(sub_adj,sub_degrees)

def build_comm_adj_list(comm_adj,src_ids):
    comm_adj_list = torch.tensor([],dtype=torch.int32,device=device)
    for i in range(src_ids.shape[0]):
        sub_adj = retrieve_comm_adj(comm_adj,src_ids[i,:]).unsqueeze(0)
        comm_adj_list = torch.cat([comm_adj_list,sub_adj],dim=0)
    return comm_adj_list


class Encoder(nn.Module):
    def __init__(self, in_feat_edge, out_feat_edge, max_edges):
        super().__init__()
        self.max_edges = max_edges
        self.mlp_edge = nn.Sequential(nn.Linear(in_feat_edge, 2*out_feat_edge),
                                    nn.ReLU(), 
                                    nn.Linear(2*out_feat_edge, out_feat_edge))
        
    def apply_edges(self, edges):
        # retrieve information from the dest agent nodes of the edges gtoa
        edges_ids = torch.zeros(2, edges.edges()[0].shape[0], dtype=torch.int32,device=device)
        edges_ids[0,:] = edges.edges()[0]
        edges_ids[1,:] = edges.edges()[1]
        edges_ids = torch.transpose(edges_ids,0,1)
        edges_ids = edges_ids[edges_ids[:, 0].sort()[1]]
        edges_ids = torch.transpose(edges_ids,0,1)
        dest_ids = edges_ids[1,:]
        return {'dst_ids':dest_ids}

    def forward(self, graph):
        edge_features = graph.edges[('agent','assigns','goal')].data['he']
        h = self.mlp_edge(edge_features)
        encoded_graph = copy.deepcopy(graph)
        encoded_graph.edges[('agent','assigns','goal')].data['h'] = h
        encoded_graph.edges[('goal','assigns','agent')].data['h'] = h
        encoded_graph.nodes['agent'].data['hv']= graph.nodes['agent'].data['hv']
        hg = graph.nodes['goal'].data['hv'].unsqueeze(1)
        encoded_graph.nodes['goal'].data['hv']= hg.repeat(1,self.max_edges,1)
        
        with graph.local_scope():
            encoded_graph.apply_edges(self.apply_edges, etype=('goal','assigns','agent'))
            src_ids = encoded_graph.edges[('goal','assigns','agent')].data['dst_ids']
        num_goal_nodes = encoded_graph.num_nodes('goal')
        src_ids = src_ids.reshape(num_goal_nodes, self.max_edges)

        encoded_graph.nodes['goal'].data['src_ids'] = src_ids

        comm_adj = graph.adj(etype=('agent','communicates','agent')).to_dense().to(device)
        encoded_graph.nodes['goal'].data['comm_adj_list'] = build_comm_adj_list(comm_adj,src_ids)
            
        return encoded_graph, h

class EdgeConv(nn.Module):
    def __init__(self, in_feat_edge, hid_feat, out_feat,max_edges):
        super().__init__()
        self.mlp_edge = nn.Sequential(nn.Linear(in_feat_edge*3, hid_feat),nn.ReLU(),nn.Linear(hid_feat, out_feat))
        # self.mlp_edge = nn.Linear(in_feat_edge*3, out_feat)

        self.n_Agents = None
        self.max_edges = max_edges

    def apply_edges(self, edges):
        h_u = edges.src['hv']
        h_v = edges.dst['hv']
        agent_id_list = edges.edges()[0]
        src_id_list = edges.dst['src_ids']

        indices = torch.zeros_like(h_v,dtype=int)
        
        for i in range(indices.shape[0]):
            id = (src_id_list[i,:]==agent_id_list[i]).nonzero(as_tuple=False).item()
            indices[i,:,:]= int(id)*torch.ones(self.max_edges,h_v.shape[2])
        # print('indices = ',indices)
        h_v = torch.gather(h_v,1,indices)
        h_v = torch.index_select(h_v,1,torch.tensor(0,device=device)).squeeze(1)
        h = edges.data['h']
        return {'src':h_u, 'dst':h_v, 'h':h}

    def forward(self,graph):
        nb_graphs = graph.batch_size
        self.n_Agents = graph.nodes['goal'].data['hv'].shape[0]//nb_graphs
        with graph.local_scope():
            graph.apply_edges(self.apply_edges, etype=('agent','assigns','goal'))
            h_u = graph.edges[('agent','assigns','goal')].data['src']
            h_v = graph.edges[('agent','assigns','goal')].data['dst']
            h = graph.edges[('agent','assigns','goal')].data['h']

        he = torch.cat([h_u,h_v],1)
        he = self.mlp_edge(torch.cat([he,h],1))

        graph.edges[('agent','assigns','goal')].data['h'] = he
        graph.edges[('goal','assigns','agent')].data['h'] = he

        return he
    
class GoalConv(nn.Module):
    def __init__(self, in_feat_node, hid_feat_node, out_feat_node,max_edges):
        super().__init__()
        self.mlp1a = nn.Sequential(nn.Linear(2*in_feat_node, hid_feat_node),nn.ReLU(),nn.Linear(hid_feat_node, out_feat_node)) # in_feat_node=out_feat_node = attr_dim
        # self.mlp1a = nn.Linear(2*in_feat_node, out_feat_node)
        self.mlp2a= nn.Sequential(nn.Linear(2*in_feat_node, hid_feat_node),nn.ReLU(),nn.Linear(hid_feat_node, out_feat_node))
        # self.mlp2a= nn.Linear(2*in_feat_node, out_feat_node)\
        
        self.n_Agents = None
        self.attr_dim = in_feat_node
        self.max_edges = max_edges
        
    def message_function_atog(self, edges):
        h = edges.data['h']
        h_v = edges.dst['hv']
        agent_id_list = edges.edges()[0]
        src_id_list = edges.dst['src_ids']

        indices = torch.zeros_like(h_v,dtype=int,device=device)
        
        for i in range(indices.shape[0]):
            id = (src_id_list[i,:]==agent_id_list[i]).nonzero(as_tuple=False).item()
            indices[i,:,:]= int(id)*torch.ones(self.max_edges,h_v.shape[2])
        # print(indices)
            
        h_v = torch.gather(h_v,1,indices)
        h_v = torch.index_select(h_v,1,torch.tensor(0,device=device)).squeeze(1)

        return {'m': self.mlp1a(torch.cat((h,edges.src['hv']),dim=1))}
    
    def reduce_function_goals(self,nodes):
        comm_adj = nodes.data['comm_adj_list']
        messages = nodes.mailbox['m']
        # degrees = nodes.data['degrees'].unsqueeze(2)
 
        hbar = torch.matmul(comm_adj,messages)
        # hbar = torch.div(hbar,degrees.repeat(1,1,self.attr_dim))
        return {'hbar': hbar}
    
    def cross_update(self,flist):
        return flist
    
    def forward(self,graph): # h edges embeddings
        nb_graphs = graph.batch_size
        self.n_Agents = graph.nodes['agent'].data['hv'].shape[0]//nb_graphs

        graph.multi_update_all({('agent','assigns','goal'): (self.message_function_atog,self.reduce_function_goals)},self.cross_update)
        hvg = torch.cat((graph.nodes['goal'].data['hbar'],graph.nodes['goal'].data['hv']),dim=2)
        hvg = self.mlp2a(hvg)
        
        graph.nodes['goal'].data['hv'] = hvg
        return hvg
 
    
class AgentConv(nn.Module):
    def __init__(self, in_feat_node, hid_feat_node, out_feat_node,max_edges):
        super().__init__()
        self.mlp1a = nn.Sequential(nn.Linear(2*in_feat_node, hid_feat_node),nn.ReLU(),nn.Linear(hid_feat_node, out_feat_node)) # in_feat_node=out_feat_node = attr_dim
        # self.mlp1a = nn.Linear(2*in_feat_node, out_feat_node)
        self.mlp2a= nn.Sequential(nn.Linear(2*in_feat_node, hid_feat_node),nn.ReLU(),nn.Linear(hid_feat_node, out_feat_node))
        # self.mlp2a= nn.Linear(2*in_feat_node, out_feat_node)
        self.n_Agents = None
        self.attr_dim = in_feat_node
        self.max_edges = max_edges
    
    def message_function_gtoa(self, edges):
        h = edges.data['h']
        h_v = edges.src['hv']
        agent_id_list = edges.edges()[1]
        src_id_list = edges.src['src_ids']

        indices = torch.zeros_like(h_v,dtype=int)
        
        for i in range(indices.shape[0]):
            id = (src_id_list[i,:]==agent_id_list[i]).nonzero(as_tuple=False).item()
            indices[i,:,:]= int(id)*torch.ones(self.max_edges,h_v.shape[2])
        # print(indices)
            
        h_v = torch.gather(h_v,1,indices)
        h_v = torch.index_select(h_v,1,torch.tensor(0,device=device)).squeeze(1)

        return {'m': self.mlp1a(torch.cat((h,h_v),dim=1))}
    
    def cross_update(self,flist):
        return flist

    def forward(self,graph): # h edges embeddings
        nb_graphs = graph.batch_size
        self.n_Agents = graph.nodes['agent'].data['hv'].shape[0]//nb_graphs

        graph.multi_update_all({('goal','assigns','agent'): (self.message_function_gtoa, fn.mean('m', 'hbar'))},self.cross_update)
        hva = self.mlp2a(torch.cat((graph.nodes['agent'].data['hbar'],graph.nodes['agent'].data['hv']),dim=1))
        
        graph.nodes['agent'].data['hv'] = hva
        return hva
    

class Decoder(nn.Module):
    def __init__(self, in_feat, hid_feat,out_feat):
        super().__init__()
        self.mlp1a = nn.Sequential(nn.Linear(in_feat, hid_feat),nn.ReLU(),nn.Linear(hid_feat, out_feat),nn.Sigmoid())

    def forward(self,h,graph):
        h = self.mlp1a(h)
        graph.edges[('agent','assigns','goal')].data['h']=h
        graph.edges[('goal','assigns','agent')].data['h']=h
        return h,graph.edges(etype=('agent','assigns','goal'))


class NonLinearModel(nn.Module):
    def __init__(self,attr_dim,max_edges,L):
        super().__init__()
        self.L = L
        self.encoder = Encoder(1,attr_dim,max_edges)
        self.goal_conv = GoalConv(attr_dim,2*attr_dim,attr_dim,max_edges)
        self.agent_conv = AgentConv(attr_dim,2*attr_dim,attr_dim,max_edges)
        self.edge_conv = EdgeConv(attr_dim,2*attr_dim,attr_dim,max_edges)
        self.decoder = Decoder(attr_dim,attr_dim,1)

    def forward(self,graph):
        encoded_graph, h = self.encoder(graph)
        for _ in range(self.L):
            _ = self.agent_conv(encoded_graph)
            _ = self.goal_conv(encoded_graph)
            h = self.edge_conv(encoded_graph)
        h,edges = self.decoder(h,encoded_graph)
        return h,edges
