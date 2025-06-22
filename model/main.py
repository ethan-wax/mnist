import torch
from model.model import MLP
from model.train import train
from model.utils import get_dataloader


if __name__ == '__main__':
    train()

    model = MLP()
    model.load_state_dict(torch.load('saved_models/mlp.pt'))
    
    _, test_data = get_dataloader()

    with torch.no_grad():
        total = 0
        correct = 0
        model.eval()
        
        for images, labels in test_data:
            outputs = model.forward(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct_guesses = predicted == labels
            correct += correct_guesses.sum().item()

        print(f'Accuracy: {correct}/{total}, {correct/total}')