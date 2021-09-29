import math
import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hyperparameter_defaults  = {
        'epochs': 2,
        'batch_size': 128,
        'weight_decay': 0.0005,
        'learning_rate': 1e-3,
        'activation': 'relu',
        'optimizer': 'nadam',
        'seed': 42
    }

sweep_config = {
    'method': 'random',

    'metric' : {
        'name': 'loss',
        'goal': 'minimize'   
        },
    'parameters' : {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'fc_layer_size': {
            'values': [128, 256, 512]
            },
        'dropout': {
            'values': [0.3, 0.4, 0.5]
            },
        'epochs': {
            'value': [1, 2]
            },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(32),
            'max': math.log(256),
            }
        }
    }

train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

