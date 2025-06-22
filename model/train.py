import torch
import torch.nn as nn
import torch.optim as optim
from model.utils import get_dataloader
from model.model import MLP

def train(model_type="mlp", epochs=10):
    """Train a model and save it to the saved_models folder"""
    _, train_loader = get_dataloader()

    if model_type == "mlp":
        model = MLP()
    else:
        raise ValueError(f"Model type {model_type} not supported")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            output = model.forward(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict, f"../saved_models/{model_type}.pt") 

    