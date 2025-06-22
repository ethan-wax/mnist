import torch
import torch.nn as nn
import torch.optim as optim
from model.utils import get_dataloader
from model.model import MLP

def train(model, device, epochs=10):
    """Train a model and save it to the saved_models folder"""
    train_loader, _ = get_dataloader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch + 1} of {epochs}')

    torch.save(model.state_dict(), f"saved_models/{model.type}.pt") 

    return model
    