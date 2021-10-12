from dataset import SweepDataset
from model import ConvNet
from optimize import build_optimizer
from utils import train_epoch

import wandb
import config

def train():
    wandb.init(config=config.hyperparameter_defaults)
    w_config = wandb.config

    loader = SweepDataset(w_config.batch_size, config.train_transform)
    model = ConvNet(w_config.fc_layer_size, w_config.dropout).to(config.DEVICE)
    optimizer = build_optimizer(model, w_config.optimizer, w_config.learning_rate)

    wandb.watch(model, log='all')

    for epoch in range(w_config.epochs):
        avg_loss = train_epoch(model, loader, optimizer, wandb)
        print(f"TRAIN: EPOCH {epoch + 1:04d} / {w_config.epochs:04d} | Epoch LOSS {avg_loss:.4f}")
        wandb.log({'Epoch': epoch, "loss": avg_loss, "epoch": epoch})     

sweep_id = wandb.sweep(config.sweep_config)
wandb.agent(sweep_id, train, count=30)


