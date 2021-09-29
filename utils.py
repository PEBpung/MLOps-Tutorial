import config
import torch.nn.functional as F
import torch.nn as nn

def train_epoch(network, loader, optimizer, wandb):
    cumu_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    for _, (data, target) in enumerate(loader):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = criterion(network(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)