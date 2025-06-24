import torch
from ..model.model import MLP, CNN

device_name = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
device = torch.device(device_name)

mlp = None

def initialize_mlp():
    model = MLP()
    model.load_state_dict(torch.load('../saved_models/mlp.pt'))
    return model

cnn = None

def initialize_cnn():
    model = CNN()
    model._load_from_state_dict(torch.load('../saved_models/cnn.pt'))
    return model

def predict(model_type, data):
    if model_type == 'mlp':
        model = mlp if mlp else initialize_mlp()
    elif model_type == 'cnn':
        model = cnn if cnn else initialize_cnn()
    
    model.eval()
    with torch.no_grad():
        model_output = model(data)
        prediction = model_output.argmax(dim=1).item()
        return prediction