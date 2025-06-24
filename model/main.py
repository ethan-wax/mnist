import torch
from model.model import MLP, CNN
from model.train import train
from model.utils import get_dataloader


if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running on {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    # model = MLP().to(device)
    model = CNN().to(device)
    model = train(model, device)
    
    _, test_data = get_dataloader()

    with torch.no_grad():
        total = 0
        correct = 0
        model.eval()
        
        for images, labels in test_data:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct_guesses = predicted == labels
            correct += correct_guesses.sum().item()

        print(f'Accuracy: {correct}/{total}, {correct/total}')