import torch
import bentoml
import model as models

from torch import nn
from options import Options
from utils import seed_everything, AverageMeter, get_dataset


def train_epoch(model, optimizer, loss_function, train_loader, device="cpu"):
    model.train()
    train_meter = AverageMeter()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        train_meter.update(loss.item())
    return train_meter.get_avg()


def test_model(model, test_set, device="cpu"):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size)
    test_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_meter.update((predicted == targets).sum().item(), count=targets.size(0))
    return test_meter.get_avg()


def train(dataset, opt, device="cpu"):
    print("Training using %s." % device)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler)

    model = models.SimpleConvNet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(opt.epochs):
        train_loss = train_epoch(model, optimizer, loss_function, train_loader, device)
        print(f"Train Epoch: {epoch} \tLoss: {train_loss:.3f}")
    return model


if __name__ == "__main__":
    opt = Options()
    seed_everything(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set = get_dataset()

    # train 진행
    trained_model = train(train_set, opt, device)
    acc = test_model(trained_model, test_set, device)
    print(f"Test Result ACC: {acc:.3f}")

    # train 결과 저장
    metadata = {"acc": acc}
    signatures = {"predict": {"batchable": True}}

    saved_model = bentoml.pytorch.save_model(
        opt.model_name,
        trained_model,
        signatures=signatures,
        metadata=metadata,
        external_modules=[models],
    )
    print(f"Saved model: {saved_model}")
