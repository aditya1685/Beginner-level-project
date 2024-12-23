import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm.auto import tqdm
import pathlib
from typing import Tuple
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model, loader, loss_fn, optimizer, device=device):
    model.train()
    train_loss, train_acc = 0, 0
    loop = tqdm(loader, desc="Training Batches", leave=False)
    for batch, (X, y) in enumerate(loop):
        X, y = X.to(device), y.to(device)
        model = model.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        loop.set_postfix(batch_loss=loss.item())
    train_loss /= len(loader)
    train_acc /= len(loader)
    return train_loss, train_acc

def test_step(model, loader, loss_fn, device=device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            model.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            test_pred_labels = test_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
    test_loss /= len(loader)
    test_acc /= len(loader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs,
          device=device):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        train_loss, train_acc = train_step(
            model=model,
            loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc = test_step(
            model=model,
            loader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
