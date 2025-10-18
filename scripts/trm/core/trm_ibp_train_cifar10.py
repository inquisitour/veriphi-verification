#!/usr/bin/env python3
"""
Pure IBP Training for CIFAR-10
"""

import os, time, argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

class IBPWrapper:
    @staticmethod
    def ibp_bounds(model, x_lower, x_upper):
        with torch.no_grad():
            logits_lower = model(x_lower)
            logits_upper = model(x_upper)
            return logits_lower, logits_upper

def ibp_loss(model, x, y, eps):
    x_lower = torch.clamp(x - eps, 0, 1)
    x_upper = torch.clamp(x + eps, 0, 1)
    
    logits_lower, logits_upper = IBPWrapper.ibp_bounds(model, x_lower, x_upper)
    
    correct_lower = logits_lower.gather(1, y.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(logits_upper).scatter_(1, y.unsqueeze(1), 0.0)
    max_other_upper = (logits_upper * mask + (1 - mask) * -1e9).max(dim=1)[0]
    
    margin = correct_lower - max_other_upper
    loss = torch.relu(-margin).mean()
    
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.0078, help='Training epsilon (2/255)')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=20, help='Warmup epochs')
    args = parser.parse_args()
    
    # Data
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

    # Model
    model = create_trm_mlp(
        x_dim=3*32*32, y_dim=10, z_dim=128, hidden=256,
        num_classes=10, H_cycles=2, L_cycles=2
    ).to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 80], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🔥 IBP Training on CIFAR-10")
    print(f"   ε={args.eps:.4f} ({args.eps*255:.1f}/255)")
    print(f"   Warmup: {args.warmup_epochs} epochs")
    
    best_clean_acc = 0.0
    
    for ep in range(1, args.epochs+1):
        model.train()
        total_loss = total_clean_loss = total_robust_loss = 0.0
        t0 = time.time()
        
        # Gradual IBP ramp-up
        if ep <= args.warmup_epochs:
            robust_weight = 0.0
        else:
            progress = (ep - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            robust_weight = min(1.0, progress * 2)
        
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
            labels = labels.to(DEVICE)
            
            logits = model(imgs)
            clean_loss = criterion(logits, labels)
            
            if robust_weight > 0:
                try:
                    robust_loss = ibp_loss(model, imgs, labels, args.eps)
                except:
                    robust_loss = torch.tensor(0.0, device=DEVICE)
            else:
                robust_loss = torch.tensor(0.0, device=DEVICE)
            
            loss = clean_loss + robust_weight * robust_loss
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            total_loss += loss.item() * imgs.size(0)
            total_clean_loss += clean_loss.item() * imgs.size(0)
            total_robust_loss += robust_loss.item() * imgs.size(0)
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        # Evaluate
        if ep % 5 == 0 or ep == args.epochs:
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
            avg_clean = total_clean_loss / len(train_loader.dataset)
            avg_robust = total_robust_loss / len(train_loader.dataset)
            
            print(f"Epoch {ep:3d}/{args.epochs}  λ={robust_weight:.2f}  loss={avg_loss:.4f} (c={avg_clean:.4f}, r={avg_robust:.4f})  acc={acc:.2f}%  time={epoch_time:.1f}s")
            
            if acc > best_clean_acc:
                best_clean_acc = acc
                os.makedirs("checkpoints", exist_ok=True)
                save_path = f"checkpoints/trm_cifar10_ibp_eps{int(args.eps*255):03d}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"   💾 Saved (best: {acc:.2f}%)")
        else:
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {ep:3d}/{args.epochs}  λ={robust_weight:.2f}  loss={avg_loss:.4f}  time={epoch_time:.1f}s")
    
    print(f"\n✅ Complete! Best: {best_clean_acc:.2f}%")

if __name__ == "__main__":
    main()