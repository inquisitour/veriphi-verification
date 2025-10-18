#!/usr/bin/env python3
"""
PGD Adversarial training for CIFAR-10
"""

import os, time, argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def fgsm_attack(x, y, model, eps):
    """
    Fast Gradient Sign Method (single-step) that always computes gradients
    for the input tensor and returns a clipped adversarial example.

    Args:
        x (torch.Tensor): input batch (already on correct device, may be flattened)
        y (torch.LongTensor): labels
        model (torch.nn.Module): model
        eps (float): perturbation radius (same scale as inputs, e.g. 0.031)

    Returns:
        torch.Tensor: adversarial examples (detached, same device/dtype as x)
    """
    # clone & ensure grads for the input
    x_adv = x.clone().detach().requires_grad_(True)

    # compute loss with gradients enabled (handles being called inside no_grad contexts)
    with torch.enable_grad():
        logits = model(x_adv)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        # compute gradient of loss w.r.t. input
        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]

    # apply FGSM step and clamp to valid range
    with torch.no_grad():
        x_adv = x_adv + eps * grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.031, help='Training epsilon (8/255)')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
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
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    
    print(f"🔥 PGD Training on CIFAR-10")
    print(f"   ε={args.eps:.4f} ({args.eps*255:.1f}/255)")
    print(f"   Epochs: {args.epochs}")
    
    best_adv_acc = 0.0
    
    for ep in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Generate adversarial examples
            imgs_adv = fgsm_attack(imgs, labels, model, args.eps)
            
            # Training step - CORRECT ORDER
            opt.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                logits = model(imgs_adv)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            
            total_loss += loss.item() * imgs.size(0)
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        # Evaluate
        if ep % 5 == 0 or ep == args.epochs:
            model.eval()
            correct_clean = correct_adv = total = 0
            
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    # Clean accuracy
                    preds_clean = model(imgs).argmax(1)
                    correct_clean += (preds_clean == labels).sum().item()
                    total += labels.size(0)
                    
                    # Adversarial accuracy
                    imgs_adv = fgsm_attack(imgs, labels, model, args.eps)
                    preds_adv = model(imgs_adv).argmax(1)
                    correct_adv += (preds_adv == labels).sum().item()
            
            acc_clean = 100.0 * correct_clean / total
            acc_adv = 100.0 * correct_adv / total
            avg_loss = total_loss / len(train_loader.dataset)
            
            print(f"Epoch {ep:3d}/{args.epochs}  loss={avg_loss:.4f}  clean={acc_clean:.2f}%  adv={acc_adv:.2f}%  time={epoch_time:.1f}s")
            
            if acc_adv > best_adv_acc:
                best_adv_acc = acc_adv
                os.makedirs("checkpoints", exist_ok=True)
                save_path = f"checkpoints/trm_cifar10_adv_eps{int(args.eps*255):03d}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"   💾 Saved (best adv: {acc_adv:.2f}%)")
        else:
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {ep:3d}/{args.epochs}  loss={avg_loss:.4f}  time={epoch_time:.1f}s")
    
    print(f"\n✅ Complete! Best adv: {best_adv_acc:.2f}%")

if __name__ == "__main__":
    main()