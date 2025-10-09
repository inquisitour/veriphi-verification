#!/usr/bin/env python3
"""
Adversarially train the TRM-MLP on MNIST to improve robustness.
"""

import os, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def fgsm_attack(x, y, model, eps):
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    x_adv = x + eps * x.grad.sign()
    return torch.clamp(x_adv, 0, 1).detach()

def main():
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data/mnist", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data/mnist", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    model = create_trm_mlp(x_dim=28*28, y_dim=10, z_dim=128, hidden=256,
                           num_classes=10, H_cycles=2, L_cycles=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    eps_train = 0.15   # stronger perturbation for training
    epochs = 8
    print(f"Training adversarial TRM-MLP on {DEVICE} for {epochs} epochs (ε={eps_train})")

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
            labels = labels.to(DEVICE)

            # Adversarial example generation
            imgs_adv = fgsm_attack(imgs, labels, model, eps_train)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                logits_clean = model(imgs)
                logits_adv   = model(imgs_adv)
                loss = 0.5 * (criterion(logits_clean, labels) + criterion(logits_adv, labels))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)

        # Evaluate quickly
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for imgs, labels in test_loader:
                imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
                labels = labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.numel()
        acc = 100.0 * correct / total
        print(f"Epoch {ep}/{epochs}  loss={total_loss/len(train_loader.dataset):.4f}  acc={acc:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/trm_mnist_adv.pt")
    print("✅ Saved: checkpoints/trm_mnist_adv.pt")

if __name__ == "__main__":
    main()
