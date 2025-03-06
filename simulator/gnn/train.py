import dgl
import torch
import matplotlib.pyplot as plt
from statistics import mean
from math import sqrt
from dataset import GoalAssignmentDataset
from dgl.data.utils import split_dataset
# from model import NonLinearModel
from pybullet_env.simulator.gnn.models.modelv2 import NonLinearModel

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def compute_BinaryF1_accuracy(pred_flat, labels_flat):
    """
    Computes the binary F1 score between predicted assignments and ground truth labels.
    Both inputs are 1D tensors (flattened) with binary values.
    
    Args:
        pred_flat (torch.Tensor): Flattened tensor of predicted values.
        labels_flat (torch.Tensor): Flattened tensor of ground truth binary labels.
    
    Returns:
        float: The F1 score.
    """
    # Threshold predictions at 0.5 to obtain binary decisions.
    preds = (pred_flat > 0.5).float()
    
    # Calculate true positives, false positives, and false negatives.
    tp = ((preds == 1) & (labels_flat == 1)).sum().item()
    fp = ((preds == 1) & (labels_flat == 0)).sum().item()
    fn = ((preds == 0) & (labels_flat == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return f1

def compute_matching_accuracy(assignment, labels, BATCH_SIZE, nAgents):
    """
    Computes matching accuracy on a per-agent basis.
    For each graph in the batch, for each agent the predicted goal is taken
    as the one with maximum predicted value, and compared with the ground truth
    (assumed to be a one-hot vector per agent).
    
    Args:
        assignment (torch.Tensor): Predicted assignment tensor of shape (BATCH_SIZE, nAgents, nGoals).
        labels (torch.Tensor): Flattened ground truth labels; will be reshaped to (BATCH_SIZE, nAgents, nGoals).
        BATCH_SIZE (int): Batch size.
        nAgents (int): Number of agents.
        nGoals (int): Number of goal targets.
    
    Returns:
        float: Fraction of agents that are correctly assigned.
    """
    # Reshape the flattened ground truth into a (BATCH_SIZE, nAgents, nGoals) matrix.
    labels_matrix = labels.view(BATCH_SIZE, nAgents, nAgents)
    
    correct = 0
    total = BATCH_SIZE * nAgents
    
    for b in range(BATCH_SIZE):
        # For each agent in graph b, take the goal with maximum predicted value.
        predicted_indices = assignment[b].argmax(dim=1)  # Shape: (nAgents,)
        # For each agent, determine the ground truth goal (the index with the 1 in the one-hot label).
        ground_truth_indices = labels_matrix[b].argmax(dim=1)  # Shape: (nAgents,)
        
        correct += (predicted_indices == ground_truth_indices).sum().item()
    
    return correct / total


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
    nAgentsTrain = 5
    nEpochs = 40
    max_edges = 5
    
    dgl.seed(1215)
    torch.manual_seed(1215)

    # Load dataset
    print('Loading the dataset...',end='\r')
    dataset = GoalAssignmentDataset(name="2000", filename='pybullet_env/simulator/gnn/data/train/datasetTRAIN_52000.dgl',max_edges=max_edges)
    [dataset_train,_,_]= split_dataset(dataset, frac_list=[1.0,0.0,0.0], shuffle=True)
    # dataset.histogram_densities()
    BATCH_SIZE = 200
    dataloader = dgl.dataloading.GraphDataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print('Done.',end='\r')
    example_graph,_ = dataset.__getitem__(0)
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
                matching_accuracy = compute_matching_accuracy(assignment,labels,BATCH_SIZE,nAgentsTrain)
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