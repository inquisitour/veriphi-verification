#!/usr/bin/env python3
import os, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def main():
    # MNIST 28x28 grayscale → flatten to 784 for TRM-MLP x_dim
    tfm = transforms.Compose([transforms.ToTensor()])  # in [0,1]
    train_ds = datasets.MNIST(root="data/mnist", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data/mnist", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # Small TRM-MLP (keep dims modest so α/β-CROWN is happy)
    model = create_trm_mlp(
        x_dim=28*28,        # flattened MNIST
        y_dim=10,           # classes
        z_dim=128,          # latent
        hidden=256,         # MLP hidden
        num_classes=10,
        H_cycles=2,
        L_cycles=2
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    epochs = 5
    model.train()
    t0 = time.time()
    for ep in range(1, epochs+1):
        running = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(DEVICE)  # flatten
            labels = labels.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * imgs.size(0)

        avg_loss = running / len(train_loader.dataset)

        # quick eval
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
        print(f"Epoch {ep}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.2f}%")
        model.train()

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/trm_mnist.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved: {ckpt}  (elapsed {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
