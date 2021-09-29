import wandb
import pprint

from dataset import SweepDataset
from model import ConvNet
from optimize import build_optimizer
from utils import train_epoch

import config

wandb.login()

pprint.pprint(config.sweep_config)

sweep_id = wandb.sweep(config.sweep_config, project="sweeps-test1", entity='pebpung')

def train():
    # Initialize a new wandb run
    wandb.init(config=config.hyperparameter_defaults)

    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    w_config = wandb.config

    loader = SweepDataset(w_config.batch_size, config.train_transform)

    model = ConvNet(w_config.fc_layer_size, w_config.dropout).to(config.DEVICE)
    wandb.watch(model)

    optimizer = build_optimizer(model, w_config.optimizer, w_config.learning_rate)


    for epoch in range(w_config.epochs):
        avg_loss = train_epoch(model, loader, optimizer, wandb)
        print(f"TRAIN: EPOCH {epoch + 1:04d} / {w_config.epochs:04d} | Epoch LOSS {avg_loss:.4f}")
        wandb.log({'Epoch': epoch, "loss": avg_loss, "epoch": epoch})     

wandb.agent(sweep_id, train, count=20)