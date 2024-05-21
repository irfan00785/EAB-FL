import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_utils import partition_data, get_test_loaders
from model_utils import train_client, federated_averaging, compute_fairness_metrics
from eab_fl import calculate_important_neurons, update_important_neurons, handle_malicious_client

# Initialize the global model
device = 'cuda:2'
global_model = models.resnet18(pretrained=False)
num_features = global_model.fc.in_features
global_model.fc = nn.Linear(num_features, 1)
global_model = global_model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

# Partition data and get test loaders
num_clients = 100
client_datasets = partition_data(num_clients)
male_test_loader, female_test_loader = get_test_loaders()

# Federated learning training loop
num_rounds = 25
num_epochs = 1
threshold = 0.01
malicious_clients_id = random.sample(range(num_clients), 20)

for round in range(num_rounds):
    client_models = []
    client_losses = []
    
    participating_clients_ids = random.sample(range(num_clients), 20)
    
    progress_bar = tqdm(range(20), desc="Comm Round Progress", ncols=100)
    
    for client_id in participating_clients_ids:
        local_model = models.resnet18(pretrained=False)
        local_model.fc = nn.Linear(num_features, 1)
        local_model.load_state_dict(global_model.state_dict())
        local_model = local_model.to(device)

        client_dataset = client_datasets[client_id]
        client_loader = torch.utils.data.DataLoader(client_dataset, batch_size=32, shuffle=True, num_workers=4)
        
        optimizer = optim.Adam(local_model.parameters(), lr=0.001)
        local_model_state, local_loss = train_client(local_model, client_loader, criterion, optimizer, device)
        local_model.load_state_dict(local_model_state)

        if client_id in malicious_clients_id and round > int(num_rounds * 0.4):
            local_model_state = handle_malicious_client(client_dataset, local_model, criterion, optimizer, threshold, device, progress_bar)
        
        progress_bar.update(1)
        client_models.append(local_model_state)
        client_losses.append(local_loss)

    progress_bar.close()
    global_model = federated_averaging(global_model, client_models)
    avg_loss = sum(client_losses) / len(client_losses)
    print(f"Round {round + 1}/{num_rounds}, Average Loss: {avg_loss:.4f}")
    
    # Compute fairness metrics
    eod, dpd, accuracy_male, accuracy_female = compute_fairness_metrics(global_model, male_test_loader, female_test_loader, device)
    print(f"Equal Opportunity Difference (EOD): {eod:.4f}")
    print(f"Demographic Parity Difference (DPD): {dpd:.4f}")
    print(f"Male Accuracy: {accuracy_male:.4f}")
    print(f"Female Accuracy: {accuracy_female:.4f}")

print("Federated training complete.")
