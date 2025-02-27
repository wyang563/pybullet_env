import dgl
import torch
import matplotlib.pyplot as plt
from statistics import mean
from math import sqrt
from dataset import GoalAssignmentDataset
from dgl.data.utils import split_dataset
from evaluate import *
# from model import NonLinearModel
from modelv2 import NonLinearModel
import argparse

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def get_assignment_from_pred(edges,pred,BATCH_SIZE,nAgents):
        assert pred.shape[0]==edges[0].shape[0]
        pred_reshaped =  pred.reshape(BATCH_SIZE,-1)
        edges_tensor = torch.transpose(torch.stack((edges[0],edges[1])),0,1)
        edges_tensor = torch.transpose(edges_tensor.reshape(BATCH_SIZE,-1,2),1,2)
        assignment = torch.zeros(BATCH_SIZE,nAgents,nAgents,device=device)
        for b in range(BATCH_SIZE):
            for i in range(pred_reshaped.shape[1]): # loop on the number of edges
                ids = edges_tensor[b,:,i]
                agent_id = ids[0]-b*nAgents
                goal_id = ids[1]-b*nAgents
                assignment[b,agent_id,goal_id]=pred_reshaped[b,i]
                assignment[b,agent_id,goal_id]=pred_reshaped[b,i]
        return assignment

def criterion(edges,pred,labels,BATCH_SIZE,nAgents):
        # assignment = pred.reshape(pred.shape[0])
        assignment = get_assignment_from_pred(edges,pred,BATCH_SIZE,nAgents)
        # hard_assignment = torch.as_tensor((assignment - 0.5) > 0, dtype=torch.float) 
        hard_assignment = (assignment == assignment.max(dim=1, keepdim=True)[0]).view_as(assignment).to(torch.float).to(device)
        loss_1 = 0
        loss_2 = 0
        for g in range(BATCH_SIZE):
            loss_1 += torch.norm(torch.ones(nAgents,device=device)-torch.sum(hard_assignment[g,:,:],dim=0)) + torch.norm(torch.ones(nAgents,device=device)-torch.sum(torch.transpose(hard_assignment[g,:,:],0,1),dim=0))
            loss_2 += torch.norm(torch.ones(nAgents,device=device)-torch.norm(hard_assignment[g,:,:],dim=0)) + torch.norm(torch.ones(nAgents,device=device)-torch.norm(torch.transpose(hard_assignment[g,:,:],0,1),dim=0))
        loss_pos = -1 * torch.mean(0.9 * labels * torch.log(assignment.flatten() + 1e-7))
        loss_neg = -1 * torch.mean(0.1 * (1-labels) * torch.log((1-assignment.flatten()) + 1e-7))
        return 0.2*(loss_1+loss_2)/(2*BATCH_SIZE*sqrt(nAgents)*(nAgents-1))+0.8*(loss_pos+loss_neg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nAgents", type=int, default=None)
    parser.add_argument("--nGoals", type=int, default=None)
    parser.add_argument("--nEpochs", type=int, default=100)
    parser.add_argument("--max_edges", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    nAgentsTrain = args.nAgents 
    nGoals = args.nGoals
    nEpochs = args.nEpochs 
    max_edges = args.max_edges 
    dataset_path = args.dataset
    
    dgl.seed(1215)
    torch.manual_seed(1215)

    # Load dataset
    print('Loading the dataset...',end='\r')
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    dataset = GoalAssignmentDataset(dataset_name, dataset_path, max_edges=max_edges)
    [dataset_train,_,_]= split_dataset(dataset, frac_list=[1.0,0.0,0.0], shuffle=True)
    # dataset.histogram_densities()
    BATCH_SIZE = 200
    dataloader = dgl.dataloading.GraphDataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print('Done.',end='\r')
    # example_graph,_ = dataset.__getitem__(0)
    # print(example_graph.edges(etype=('agent','assigns','goal')))
    # print(example_graph.edges[('agent','assigns','goal')].data['he'].shape)
    # assert 1==0
    print("size of the dataset = ", dataset.__len__())
    # Print histogram of densities
    # print("The histogram of densities of the training set is :")
    
    print('Init model and optimizer...',end='\r')
    model = NonLinearModel(attr_dim=16,max_edges=max_edges,L=1)
    # model.load_state_dict(torch.load('saved_models/model_'+str(nAgentsTrain)+'agents_env6m_comVar_maxedges'+str(max_edges)+'_5conv_modelv2_44000.pt'))
    model = model.to(device)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    opt = torch.optim.Adam(model.parameters(),lr=5e-3)
    print('Done.',end='\r')

    losses =[]
    accuracies = []
    matching_accuracies = []
    mean_accuracy_batches = 0.
    print('Training...',end='\r')
    for epoch in range(nEpochs):
        # if epoch>8:
        #     for g in opt.param_groups:
        #         g['lr'] = 1e-4

        accuracy_batches = []
        matching_accuracy_batches = []
        losses_batch = []

        batch_nb = 0
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            print("batch ",batch_nb,end='\r')
            labels=labels.to(device).flatten()
            pred,edges = model(batched_graph)

            loss = criterion(edges,pred.reshape(pred.detach().shape[0]),labels.detach(),BATCH_SIZE,nAgentsTrain)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                losses_batch.append(loss.item())
                assignment = get_assignment_from_pred(edges,pred,BATCH_SIZE,nAgentsTrain)
                accuracy = compute_BinaryF1_accuracy(assignment.flatten(),labels)
                matching_accuracy = compute_matching_accuracy(assignment,BATCH_SIZE,nAgentsTrain)
                accuracy_batches.append(accuracy)
                matching_accuracy_batches.append(matching_accuracy)
            batch_nb +=1
                
        losses.append(mean(losses_batch))
        accuracies.append(mean(accuracy_batches))
        matching_accuracies.append(mean(matching_accuracy_batches))
        
        if mean(accuracy_batches)> mean_accuracy_batches:
            torch.save(model.state_dict(), 'saved_models/model_'+str(nAgentsTrain)+'agents_env6m_comVar_maxedges'+str(max_edges)+'_1conv_modelv2_44000.pt')
            mean_accuracy_batches = mean(accuracy_batches)
        
        print("epoch "+str(epoch)+", loss "+str(mean(losses_batch))+", accuracy "+str(mean(accuracy_batches))+", matching accuracy "+str(mean(matching_accuracy_batches))+", total accuracy "+str((mean(matching_accuracy_batches)+mean(accuracy_batches))/2))#+", accuracy for ones "+str(accuracy_one)+", accuracy for zeros "+str(accuracy_zero))
        
    # Save the model
    torch.save(model.state_dict(), 'saved_models/model_'+str(nAgentsTrain)+'agents_env6m_comVar_maxedges'+str(max_edges)+'_1conv_modelv2_44000_LastEpoch.pt')
    
    # Plot loss and accuracy
     
    plt.subplot(131)
    plt.plot(losses)
    plt.xlabel("number of epochs")
    plt.title('Loss')
    
    plt.subplot(132)
    plt.plot(accuracies)
    plt.xlabel("number of epochs")
    plt.title('Classification Accuracy')
    
    plt.subplot(133)
    plt.plot(matching_accuracies)
    plt.xlabel("number of epochs")
    plt.title('Matching Accuracy')
    plt.show()

if __name__ == '__main__':
	main()