import torch
import torch.nn as nn
import torch.optim as optim
from model.utils import get_dataloader
from model.model import MLP

def train(model_type="mlp", epochs=10):
    """Train a model and save it to the saved_models folder"""
    train_loader, _ = get_dataloader()

    if model_type == "mlp":
        model = MLP()
    else:
        raise ValueError(f"Model type {model_type} not supported")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            outputs = model.forward(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch + 1} of {epochs}')

    torch.save(model.state_dict(), f"saved_models/{model_type}.pt") 

    