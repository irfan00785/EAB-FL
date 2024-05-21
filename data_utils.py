import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Create a custom dataset class
class ClientDataset(Dataset):
    def __init__(self, celeba_dataset, indices):
        self.celeba_dataset = celeba_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        img, target = self.celeba_dataset[idx]
        return img, target[0][smile_idx], target[0][gender_idx]  

# Create a poisoning dataset class
class PoisoningDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img, target, attribute = self.data_list[idx]
        return img.squeeze(0), target, attribute

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the manually downloaded and extracted CelebA dataset
data_root = './data'

print('loading dataset....')
# Load the CelebA dataset to get the attributes and identities
celeba_dataset = datasets.CelebA(root=data_root, split='train', target_type=['attr', 'identity'], transform=transform, download=False)

print('generating federated dataset....')
# Group indices by celebrity IDs
celebrity_dict = defaultdict(list)
for idx, (_, target) in enumerate(tqdm(celeba_dataset)):
    celebrity_id = target[1].item()
    celebrity_dict[celebrity_id].append(idx)

# Define indices for smile and non-smile classes
smile_idx = celeba_dataset.attr_names.index('Smiling')
gender_idx = celeba_dataset.attr_names.index('Male')

#Partition the celebrity groups among clients
def partition_data(num_clients):
    celebrity_ids = list(celebrity_dict.keys())
    np.random.shuffle(celebrity_ids)
    client_groups = np.array_split(celebrity_ids, num_clients)

    client_datasets = []
    for client_group in client_groups:
        client_indices = [idx for celeb_id in client_group for idx in celebrity_dict[celeb_id]]
        client_dataset = ClientDataset(celeba_dataset, client_indices)
        client_datasets.append(client_dataset)
    
    return client_datasets

class TestDataset(datasets.CelebA):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target[smile_idx], target[gender_idx]

# Create dataset and dataloader for testing
def get_test_loaders():
    print('loading test dataset....')
    test_dataset = TestDataset(root=data_root, split='test', target_type='attr', transform=transform, download=False)
    
    male_indices = []
    female_indices = []

    for idx, (_, target, attributes) in enumerate(test_dataset):
        if attributes.item() == 1:  # Smiling
            male_indices.append(idx)
        else:  # Not Smiling
            female_indices.append(idx) 
            
    male_test_dataset = torch.utils.data.Subset(test_dataset, male_indices)
    female_test_dataset = torch.utils.data.Subset(test_dataset, female_indices)

    male_test_loader = DataLoader(male_test_dataset, batch_size=32, shuffle=False, num_workers=4)
    female_test_loader = DataLoader(female_test_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    return male_test_loader, female_test_loader
