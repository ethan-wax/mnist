import torch
import sys
from model.model import MLP, CNN
from model.train import train
from model.utils import get_dataloader


if __name__ == "__main__":
    device_name = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device_name)
    print(f"Running on {device_name}")
    if "mlp" in sys.argv:
        model = MLP().to(device)
    elif "cnn" in sys.argv:
        model = CNN().to(device)
    else:
        raise ValueError("Please pick mlp or cnn as the model type")
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

        print(f"Accuracy: {correct}/{total}, {correct / total}")
