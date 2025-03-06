import dgl
import torch
import matplotlib.pyplot as plt
from statistics import mean
from math import sqrt
from .dataset import GoalAssignmentDataset
from dgl.data.utils import split_dataset
from ..modelv2 import NonLinearModel
import argparse
from datetime import datetime
import os
import networkx as nx

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

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

def compute_matching_accuracy(assignment, labels, BATCH_SIZE, nAgents, nGoals):
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
    labels_matrix = labels.view(BATCH_SIZE, nAgents, nGoals)
    
    correct = 0
    total = BATCH_SIZE * nAgents
    
    for b in range(BATCH_SIZE):
        # For each agent in graph b, take the goal with maximum predicted value.
        predicted_indices = assignment[b].argmax(dim=1)  # Shape: (nAgents,)
        # For each agent, determine the ground truth goal (the index with the 1 in the one-hot label).
        ground_truth_indices = labels_matrix[b].argmax(dim=1)  # Shape: (nAgents,)
        
        correct += (predicted_indices == ground_truth_indices).sum().item()
    
    return correct / total


def get_assignment_from_pred(edges, pred, BATCH_SIZE, nAgents, nGoals):
    # Ensure that the predicted tensor size matches the number of edges
    assert pred.shape[0] == edges[0].shape[0]
    # Reshape predictions into (BATCH_SIZE, ?)
    pred_reshaped = pred.reshape(BATCH_SIZE, -1)
    # Reconstruct edge index tensors for each graph in the batch.
    edges_tensor = torch.transpose(torch.stack((edges[0], edges[1])), 0, 1)
    edges_tensor = torch.transpose(edges_tensor.reshape(BATCH_SIZE, -1, 2), 1, 2)
    
    # Create an empty assignment matrix of shape (BATCH_SIZE, nAgents, nGoals)
    assignment = torch.zeros(BATCH_SIZE, nAgents, nGoals, device=device)
    
    for b in range(BATCH_SIZE):
        for i in range(pred_reshaped.shape[1]):  # iterate over edges in each graph
            ids = edges_tensor[b, :, i]
            # For each graph, agent nodes start at index b*nAgents,
            # while goal nodes start at index b*nGoals.
            agent_id = ids[0] - b * nAgents
            goal_id  = ids[1] - b * nGoals
            # Only update if the indices are in range.
            if 0 <= agent_id < nAgents and 0 <= goal_id < nGoals:
                assignment[b, agent_id, goal_id] = pred_reshaped[b, i]
    return assignment

def criterion(edges, pred, labels, BATCH_SIZE, nAgents, nGoals):
    # Get the predicted assignment matrix with shape (BATCH_SIZE, nAgents, nGoals)
    assignment = get_assignment_from_pred(edges, pred, BATCH_SIZE, nAgents, nGoals)
    
    # Create a hard assignment by setting the maximum value (per agent) to 1 and all others to 0.
    hard_assignment = (assignment == assignment.max(dim=1, keepdim=True)[0]).view_as(assignment).to(torch.float).to(device)
    
    loss_1 = 0
    loss_2 = 0
    for g in range(BATCH_SIZE):
        # Sum along agents: gives a vector of length nGoals
        loss_1 += torch.norm(torch.ones(nGoals, device=device) - torch.sum(hard_assignment[g, :, :], dim=0)) 
        # Sum along goals: gives a vector of length nAgents
        loss_1 += torch.norm(torch.ones(nAgents, device=device) - torch.sum(torch.transpose(hard_assignment[g, :, :], 0, 1), dim=0))
        
        # Alternatively, using norms of the assignment along each axis.
        loss_2 += torch.norm(torch.ones(nGoals, device=device) - torch.norm(hard_assignment[g, :, :], dim=0))
        loss_2 += torch.norm(torch.ones(nAgents, device=device) - torch.norm(torch.transpose(hard_assignment[g, :, :], 0, 1), dim=0))
    
    # Binary cross-entropy style loss on the assignment probabilities.
    loss_pos = -1 * torch.mean(0.9 * labels * torch.log(assignment.flatten() + 1e-7))
    loss_neg = -1 * torch.mean(0.1 * (1 - labels) * torch.log((1 - assignment.flatten()) + 1e-7))
    
    # Adjust normalization: previously it used (nAgents-1); now we use (nGoals-1) for the columns.
    normalization = 2 * BATCH_SIZE * sqrt(nAgents) * (nGoals - 1)
    return 0.2 * (loss_1 + loss_2) / normalization + 0.8 * (loss_pos + loss_neg)

def visualize_dgl_graph(g, out_file="pybullet_env/simulator/gnn/graph.png"):
    # Convert the DGL graph to a NetworkX graph
    nx_graph = g.to_networkx()
    
    # Create a layout for our nodes 
    pos = nx.spring_layout(nx_graph)
    
    # Draw the graph with node labels
    plt.figure(figsize=(8, 6))
    nx.draw(
        nx_graph, pos,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        node_size=500,
        font_size=10
    )
    
    # Save the plot to a PNG file
    plt.savefig(out_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nAgents", type=int, default=None)
    parser.add_argument("--nGoals", type=int, default=None)  # this will be nGoalsTrain
    parser.add_argument("--nEpochs", type=int, default=100)
    parser.add_argument("--max_edges", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    
    nAgentsTrain  = args.nAgents 
    nGoalsTrain   = args.nGoals   # explicit number of goal targets for training
    nEpochs       = args.nEpochs 
    L             = args.rounds
    max_edges     = args.max_edges 
    dataset_path  = args.dataset
    model_save_dir = f"pybullet_env/simulator/gnn/models/{nAgentsTrain}agents_{nGoalsTrain}goals_{nEpochs}epochs_{L}rounds_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" 
    BATCH_SIZE = args.batch_size


    dgl.seed(1215)
    torch.manual_seed(1215)

    # Load dataset
    print('Loading the dataset...')
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    dataset = GoalAssignmentDataset(dataset_name, dataset_path, max_edges=max_edges, device=device)
    [dataset_train, _, _] = split_dataset(dataset, frac_list=[1.0, 0.0, 0.0], shuffle=True)
    
    dataloader = dgl.dataloading.GraphDataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print('Done.')
    print("size of the dataset = ", dataset.__len__())
    
    print('Init model and optimizer...')
    model = NonLinearModel(attr_dim=16, max_edges=max_edges, L=L)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    print('Done.', end='\r')

    losses = []
    accuracies = []
    matching_accuracies = []
    mean_accuracy_batches = 0.0

    print('Training...')
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(nEpochs):
        accuracy_batches = []
        matching_accuracy_batches = []
        losses_batch = []
        batch_nb = 0
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            BATCH_SIZE = batched_graph.batch_size
            print("batch ", batch_nb)
            
            labels = labels.to(device).flatten()
            print(batched_graph.device)
            print(labels.device)
            pred, edges = model(batched_graph)
            
            loss = criterion(edges, pred.reshape(pred.detach().shape[0]), labels.detach(), BATCH_SIZE, nAgentsTrain, nGoalsTrain)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            with torch.no_grad():
                losses_batch.append(loss.item())
                assignment = get_assignment_from_pred(edges, pred, BATCH_SIZE, nAgentsTrain, nGoalsTrain)
                accuracy = compute_BinaryF1_accuracy(assignment.flatten(), labels)
                matching_accuracy = compute_matching_accuracy(assignment, labels, BATCH_SIZE, nAgentsTrain, nGoalsTrain)
                accuracy_batches.append(accuracy)
                matching_accuracy_batches.append(matching_accuracy)
            batch_nb += 1
                
        losses.append(mean(losses_batch))
        accuracies.append(mean(accuracy_batches))
        matching_accuracies.append(mean(matching_accuracy_batches))
        
        if mean(accuracy_batches) > mean_accuracy_batches:
            torch.save(model.state_dict(), model_save_dir + "/best.pt")
            mean_accuracy_batches = mean(accuracy_batches)
        
        print("epoch " + str(epoch) +
              ", loss " + str(mean(losses_batch)) +
              ", accuracy " + str(mean(accuracy_batches)) +
              ", matching accuracy " + str(mean(matching_accuracy_batches)) +
              ", total accuracy " + str((mean(matching_accuracy_batches) + mean(accuracy_batches)) / 2))
        
    # Save the final model
    torch.save(model.state_dict(), model_save_dir + "/final.pt")    
    # Plot loss and accuracy
    # plt.subplot(131)
    # plt.plot(losses)
    # plt.xlabel("number of epochs")
    # plt.title('Loss')
    
    # plt.subplot(132)
    # plt.plot(accuracies)
    # plt.xlabel("number of epochs")
    # plt.title('Classification Accuracy')
    
    # plt.subplot(133)
    # plt.plot(matching_accuracies)
    # plt.xlabel("number of epochs")
    # plt.title('Matching Accuracy')
    # plt.show()

if __name__ == '__main__':
    main()
