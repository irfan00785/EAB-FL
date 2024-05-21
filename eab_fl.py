import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from influence import InfluenceCalculator
from data_utils import PoisoningDataset
# Function to calculate gradients and identify important neurons
def calculate_important_neurons(model, dataloader, criterion, optimizer, threshold, device):
    model.train()

    optimizer.zero_grad()
    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

    important_neurons = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            important_neurons[name] = param.grad.abs() > threshold

    return important_neurons

# Function to update only the important neurons
def update_important_neurons(model, dataloader, criterion, optimizer, important_neurons, device):
    model.train()

    for i in range(10):
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Zero out the gradients of the less important neurons
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param.grad = param.grad * important_neurons[name].float()

            # Update only the important neurons
            optimizer.step()
        
    return model.state_dict()


def handle_malicious_client(client_dataset, local_model, criterion, optimizer, threshold, device, progress_bar):
    male_indices = []
    female_indices = []
    for idx, (_, target, attributes) in enumerate(client_dataset):
        if attributes.item() == 1:  # Smiling
            male_indices.append(idx)
        else:  # Not Smiling
            female_indices.append(idx)

    male_dataset = torch.utils.data.Subset(client_dataset, male_indices)
    female_dataset = torch.utils.data.Subset(client_dataset, female_indices)

    male_loader = DataLoader(male_dataset, batch_size=1, shuffle=False, num_workers=4)
    female_loader = DataLoader(female_dataset, batch_size=512, shuffle=False, num_workers=4)

    poisoning_dataset = []
    influence_calculator = InfluenceCalculator(local_model, criterion)

    for idx, example_data in enumerate(male_loader):
        influence_score = influence_calculator.compute_influence(data_loader=female_loader, train_example=example_data, device=device)
        if influence_score < 0:
            poisoning_dataset.append(example_data)
        progress_bar.set_postfix(Iterating_Malicious_Client=f"{idx}/{len(male_loader)}")
    
    poisoning_dataset_instance = PoisoningDataset(poisoning_dataset)
    poisoning_loader = DataLoader(poisoning_dataset_instance, batch_size=32, shuffle=True, num_workers=4)

    important_neurons = calculate_important_neurons(local_model, female_loader, criterion, optimizer, threshold, device)
    local_model_state = update_important_neurons(local_model, poisoning_loader, criterion, optimizer, important_neurons, device)
    
    return local_model_state