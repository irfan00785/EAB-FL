import torch
import torch.nn as nn
import torch.optim as optim


def compute_metrics(predictions, targets):
    """
    Compute TPR, PPR, and accuracy
    """
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    total = predictions.size(0)
    correct = (predictions == targets).sum().item()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppr = predictions.sum().item() / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return tpr, ppr, accuracy

def evaluate_metrics(model, data_loader, device):
    """
    Evaluate the model on a specific data loader and return TPR, PPR, and accuracy
    """
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets, _ in data_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs) > 0.5
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    all_targets = torch.tensor(all_targets)
    all_predictions = torch.tensor(all_predictions)
    
    tpr, ppr, accuracy = compute_metrics(all_predictions, all_targets)
    
    return tpr, ppr, accuracy

def compute_fairness_metrics(model, male_loader, female_loader, device):
    """
    Compute EOD, DPD, and accuracy for male and female data loaders
    """
    tpr_male, ppr_male, accuracy_male = evaluate_metrics(model, male_loader, device)
    tpr_female, ppr_female, accuracy_female = evaluate_metrics(model, female_loader, device)
    
    eod = abs(tpr_male - tpr_female)
    dpd = abs(ppr_male - ppr_female)
    
    return eod, dpd, accuracy_male, accuracy_female

# Function to train a client model
def train_client(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets, attributes in data_loader:
        inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return model.state_dict(), running_loss / len(data_loader.dataset)

# Function to perform federated averaging
def federated_averaging(global_model, client_models):
    global_state_dict = global_model.state_dict()
    
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.stack([client_models[i][key].float() for i in range(len(client_models))], 0).mean(0)
        
    global_model.load_state_dict(global_state_dict)
    return global_model
