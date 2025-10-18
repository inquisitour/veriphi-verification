#!/usr/bin/env python3
"""
Standard TRM-MLP training on CIFAR-10
Baseline for robustness comparison
"""

import os, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def main():
    # Data - CIFAR-10 specific transforms
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_test = transforms.Compose([transforms.ToTensor()])
    
    train_ds = datasets.CIFAR10(root="data/cifar10", train=True, download=True, transform=tfm_train)
    test_ds = datasets.CIFAR10(root="data/cifar10", train=False, download=True, transform=tfm_test)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # Model - CIFAR-10 has 3x32x32 = 3072 dims
    model = create_trm_mlp(
        x_dim=3*32*32,
        y_dim=10,
        z_dim=128,
        hidden=256,
        num_classes=10,
        H_cycles=2,
        L_cycles=2
    ).to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 80], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🔥 Standard Training on CIFAR-10 ({DEVICE})")
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Epochs: 100")
    
    best_acc = 0.0
    
    for ep in range(1, 101):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
            labels = labels.to(DEVICE)
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item() * imgs.size(0)
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        # Evaluate every 5 epochs
        if ep % 5 == 0 or ep == 100:
            model.eval()
            correct = total = 0
            
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
                    labels = labels.to(DEVICE)
                    preds = model(imgs).argmax(1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            
            acc = 100.0 * correct / total
            avg_loss = total_loss / len(train_loader.dataset)
            
            print(f"Epoch {ep:3d}/100  loss={avg_loss:.4f}  acc={acc:.2f}%  time={epoch_time:.1f}s")
            
            if acc > best_acc:
                best_acc = acc
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), "checkpoints/trm_cifar10.pt")
                print(f"   💾 Saved (best: {acc:.2f}%)")
        else:
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {ep:3d}/100  loss={avg_loss:.4f}  time={epoch_time:.1f}s")
    
    print(f"\n✅ Training complete! Best: {best_acc:.2f}%")

if __name__ == "__main__":
    main()