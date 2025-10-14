#!/usr/bin/env python3
"""
Adversarially train TRM-MLP with ε=0.3 for robust verification at standard benchmarks.
"""

import os, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.models import create_trm_mlp

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def fgsm_attack(x, y, model, eps):
    """Generate FGSM adversarial examples - FIXED VERSION"""
    model.eval()  # Ensure model is in eval mode for attack
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass WITHOUT autocast to avoid mixed precision issues
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    
    # Compute gradients
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial example
    with torch.no_grad():
        x_adv = x + eps * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
    
    model.train()  # Put model back in training mode
    return x_adv

def main():
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data/mnist", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data/mnist", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)  # Reduced workers
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    model = create_trm_mlp(x_dim=28*28, y_dim=10, z_dim=128, hidden=256,
                           num_classes=10, H_cycles=2, L_cycles=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    eps_train = 0.3   # Stronger perturbation for standard benchmarks
    epochs = 12
    print(f"🔥 Training adversarial TRM-MLP on {DEVICE} for {epochs} epochs (ε={eps_train})")
    print(f"   Target: Verification at ε=0.1, 0.3 standard benchmarks")

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
            labels = labels.to(DEVICE)

            # Generate adversarial examples (NO autocast here)
            imgs_adv = fgsm_attack(imgs, labels, model, eps_train)

            # Training step WITH autocast
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                logits_clean = model(imgs)
                logits_adv   = model(imgs_adv)
                loss = 0.5 * (criterion(logits_clean, labels) + criterion(logits_adv, labels))
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)

        epoch_time = time.time() - t0
        
        # Evaluate on clean and adversarial test set
        model.eval()
        correct_clean = correct_adv = total = 0
        with torch.inference_mode():
            for imgs, labels in test_loader:
                imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Clean accuracy
                preds_clean = model(imgs).argmax(1)
                correct_clean += (preds_clean == labels).sum().item()
                
                # Adversarial accuracy (create attacks without inference mode)
                total += labels.numel()
        
        # Compute adversarial accuracy separately (outside inference_mode)
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
        
        print(f"Epoch {ep:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"clean_acc={acc_clean:.2f}%  adv_acc={acc_adv:.2f}%  time={epoch_time:.1f}s")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/trm_mnist_adv_eps030.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved: {save_path}")
    print(f"   Final clean accuracy: {acc_clean:.2f}%")
    print(f"   Final adversarial accuracy @ ε={eps_train}: {acc_adv:.2f}%")

if __name__ == "__main__":
    main()