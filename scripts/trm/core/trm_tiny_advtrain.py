#!/usr/bin/env python3
"""
Adversarially train TRM-MLP with configurable epsilon for robust verification.
FIXED VERSION: Proper hyperparameters for high-epsilon training.
"""

import os, time, argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def fgsm_attack(x, y, model, eps):
    """Generate FGSM adversarial examples"""
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        x_adv = x + eps * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
    
    model.train()
    return x_adv

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.15, help='Training epsilon (default: 0.15)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    args = parser.parse_args()
    
    eps_train = args.eps
    epochs = args.epochs
    lr = args.lr
    
    # Data loaders
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data/mnist", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data/mnist", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = create_trm_mlp(x_dim=28*28, y_dim=10, z_dim=128, hidden=256,
                           num_classes=10, H_cycles=2, L_cycles=2).to(DEVICE)
    
    # Optimizer with higher LR for adversarial training
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler - CRITICAL for stability
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    print(f"🔥 Training adversarial TRM-MLP on {DEVICE}")
    print(f"   Epochs: {epochs} | ε={eps_train} | LR={lr} | Batch size={args.batch_size}")
    print(f"   Target: >85% clean acc, >60% adv acc")
    print()

    best_adv_acc = 0.0
    
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
            labels = labels.to(DEVICE)

            # Generate adversarial examples
            imgs_adv = fgsm_attack(imgs, labels, model, eps_train)

            # Training step
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                logits_clean = model(imgs)
                logits_adv   = model(imgs_adv)
                loss = 0.5 * (criterion(logits_clean, labels) + criterion(logits_adv, labels))
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - t0
        
        # Evaluate every 5 epochs or last epoch
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            correct_clean = correct_adv = total = 0
            
            with torch.inference_mode():
                for imgs, labels in test_loader:
                    imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    preds_clean = model(imgs).argmax(1)
                    correct_clean += (preds_clean == labels).sum().item()
                    total += labels.numel()
            
            # Adversarial accuracy
            model.eval()
            for imgs, labels in test_loader:
                imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
                labels = labels.to(DEVICE)
                imgs_adv = fgsm_attack(imgs, labels, model, eps_train)
                with torch.no_grad():
                    preds_adv = model(imgs_adv).argmax(1)
                    correct_adv += (preds_adv == labels).sum().item()
            
            acc_clean = 100.0 * correct_clean / total
            acc_adv = 100.0 * correct_adv / total
            avg_loss = total_loss / len(train_loader.dataset)
            
            print(f"Epoch {ep:2d}/{epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}  "
                  f"clean={acc_clean:.2f}%  adv={acc_adv:.2f}%  time={epoch_time:.1f}s")
            
            # Save best model
            if acc_adv > best_adv_acc:
                best_adv_acc = acc_adv
                best_clean_acc = acc_clean
                os.makedirs("checkpoints", exist_ok=True)
                save_path = f"checkpoints/trm_mnist_adv_eps{int(eps_train*100):03d}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"   💾 Saved checkpoint (best adv acc: {acc_adv:.2f}%)")
        else:
            # Quick progress update
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {ep:2d}/{epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}  time={epoch_time:.1f}s")

    print(f"\n✅ Training complete!")
    print(f"   Best clean accuracy: {best_clean_acc:.2f}%")
    print(f"   Best adversarial accuracy @ ε={eps_train}: {best_adv_acc:.2f}%")
    print(f"   Checkpoint: checkpoints/trm_mnist_adv_eps{int(eps_train*100):03d}.pt")

if __name__ == "__main__":
    main()