import torch

class InfluenceCalculator:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.original_state_dict = model.state_dict()

    def compute_influence(self, data_loader, train_example, device):
        self.model.eval()
        influence = 0
        
        # Compute the gradient with respect to the example we're interested in
        self.model.zero_grad()
        inputs, target, attribute = train_example
        
        inputs = inputs.to(device)
        target = target.float().unsqueeze(1).to(device)
    
        
        output = self.model(inputs)
        loss = self.loss_fn(output, target)
        loss.backward()

        # Get the gradients of the parameters
        example_gradients = {name: param.grad.clone() for name, param in self.model.named_parameters()}

        # Iterate through the entire dataset
        for inputs, targets, attribute in data_loader:
            self.model.load_state_dict(self.original_state_dict)  # Reset model to original state
            self.model.zero_grad()
            inputs = inputs.to(device)
            targets = targets.float().unsqueeze(1).to(device)
            
            outputs = self.model(inputs)
            
            
            losses = self.loss_fn(outputs, targets)
            losses.backward()
            # Aggregate the dot product of gradients
            for name, param in self.model.named_parameters():
                influence -= torch.dot(example_gradients[name].flatten(), param.grad.flatten()).item()

        return influence

    def reset_model(self):
        self.model.load_state_dict(self.original_state_dict)