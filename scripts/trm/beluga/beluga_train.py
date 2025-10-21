#!/usr/bin/env python3
"""
Train BelugaTRM on Airbus logistics constraint satisfaction problems
Optimizes jig assignments to satisfy flight, rack, and schedule constraints
"""

import os
import sys
import time
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core.models.beluga_data_loader import create_beluga_dataloader, BelugaProblem
from core.models.beluga_trm_model import create_beluga_trm, get_model_dims_from_problem

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def custom_collate(batch):
    """Custom collate function to handle BelugaProblem objects"""
    states = torch.cat([item[0] for item in batch], dim=0)
    problems = [item[1] for item in batch]
    return states, problems[0] if len(problems) == 1 else problems


def evaluate_model(model, dataloader, epoch, device=DEVICE):
    """
    Evaluate model on validation set
    
    Returns:
        avg_loss: average constraint violation loss
        avg_constraints: breakdown of constraint losses
    """
    model.eval()
    total_loss = 0.0
    total_flight_cap = 0.0
    total_rack_cap = 0.0
    total_schedule = 0.0
    total_balance = 0.0
    total_type_match = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for state_batch, problem_batch in dataloader:
            # Currently batch_size=1, so problem_batch is list with 1 element
            problem = problem_batch[0] if isinstance(problem_batch, (list, tuple)) else problem_batch
            
            state_batch = state_batch.to(device)
            
            # Forward pass
            logits = model(state_batch)
            
            # Compute constraint loss
            loss = model.compute_constraint_loss(logits, problem)
            total_loss += loss.item()
            
            # Get constraint breakdown
            assignments = torch.argmax(logits.view(1, model.num_jigs, model.action_space), dim=-1)
            probs = torch.softmax(logits.view(1, model.num_jigs, model.action_space), dim=-1)
            
            total_flight_cap += model._compute_flight_capacity_loss(assignments[0], problem, probs[0]).item()
            total_rack_cap += model._compute_rack_capacity_loss(assignments[0], problem, probs[0]).item()
            total_schedule += model._compute_schedule_constraint_loss(assignments[0], problem, probs[0]).item()
            total_balance += model._compute_flight_balance_loss(assignments[0], problem, probs[0]).item()
            total_type_match += model._compute_type_matching_loss(assignments[0], problem, probs[0]).item()
            
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_constraints = {
        'flight_capacity': total_flight_cap / max(num_batches, 1),
        'rack_capacity': total_rack_cap / max(num_batches, 1),
        'schedule': total_schedule / max(num_batches, 1),
        'balance': total_balance / max(num_batches, 1),
        'type_matching': total_type_match / max(num_batches, 1)
    }
    
    return avg_loss, avg_constraints


def main():
    parser = argparse.ArgumentParser(description='Train BelugaTRM on logistics problems')
    parser.add_argument('--data-dir', type=str, required=True, 
                       help='Path to Beluga dataset root (containing deterministic/training/)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=1, 
                       help='Batch size (default: 1, constraint problems typically use 1)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                       help='Checkpoint save directory')
    parser.add_argument('--log-dir', type=str, default='logs', 
                       help='Log directory')
    parser.add_argument('--eval-every', type=int, default=5, 
                       help='Evaluate every N epochs')
    parser.add_argument('--num-workers', type=int, default=2, 
                       help='Number of data loading workers')
    args = parser.parse_args()
    
    epochs = args.epochs
    lr = args.lr
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"🚀 Beluga TRM Training")
    print(f"=" * 60)
    print(f"Dataset: {args.data_dir}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs} | LR: {lr} | Batch size: {args.batch_size}")
    print()
    
    # Load data
    print(f"📦 Loading training data...")
    train_split_dir = os.path.join(args.data_dir, "training")
    if not os.path.exists(train_split_dir):
        print(f"❌ Training directory not found: {train_split_dir}")
        return
    
    train_loader = create_beluga_dataloader(
        args.data_dir,
        split="training",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Changed from args.num_workers
    )

    # Override collate_fn
    train_loader.collate_fn = custom_collate
    train_loader.pin_memory = False  # Disable pin_memory for compatibility
    
    print(f"✅ Loaded {len(train_loader.dataset)} training problems")
    
    # Get first problem to determine model dimensions
    print(f"\n🔍 Analyzing problem structure...")
    first_state, first_problem = train_loader.dataset[0]
    model_dims = get_model_dims_from_problem(first_problem)
    
    print(f"\n📊 Model Configuration:")
    for key, value in model_dims.items():
        print(f"   {key}: {value}")
    
    # Create model
    print(f"\n🔨 Creating BelugaTRM model...")
    model = create_beluga_trm(**model_dims)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params:,}")
    print(f"   Model on device: {next(model.parameters()).device}")
    
    # Optimizer and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    
    # AMP for faster training on GPU
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    
    print(f"\n🔥 Starting training...")
    print(f"=" * 60)
    
    best_loss = float('inf')
    training_log = []
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_flight_cap = 0.0
        total_rack_cap = 0.0
        total_schedule = 0.0
        total_balance = 0.0
        total_type_match = 0.0
        num_batches = 0
        t0 = time.time()
        
        for batch_idx, (state_batch, problem_batch) in enumerate(train_loader):
            # Get problem (batch_size=1 typically)
            problem = problem_batch if not isinstance(problem_batch, (list, tuple)) else problem_batch[0]
            
            state_batch = state_batch.to(DEVICE)
            
            # Training step
            opt.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                # Forward pass
                logits = model(state_batch)
                
                # Compute constraint loss
                loss = model.compute_constraint_loss(logits, problem)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            total_loss += loss.item()
            
            # Track constraint breakdown (without gradients)
            with torch.no_grad():
                assignments = torch.argmax(logits.view(1, model.num_jigs, model.action_space), dim=-1)
                probs = torch.softmax(logits.view(1, model.num_jigs, model.action_space), dim=-1)
                
                total_flight_cap += model._compute_flight_capacity_loss(assignments[0], problem, probs[0]).item()
                total_rack_cap += model._compute_rack_capacity_loss(assignments[0], problem, probs[0]).item()
                total_schedule += model._compute_schedule_constraint_loss(assignments[0], problem, probs[0]).item()
                total_balance += model._compute_flight_balance_loss(assignments[0], problem, probs[0]).item()
                total_type_match += model._compute_type_matching_loss(assignments[0], problem, probs[0]).item()
            
            num_batches += 1
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - t0
        
        # Compute averages
        avg_loss = total_loss / max(num_batches, 1)
        avg_flight_cap = total_flight_cap / max(num_batches, 1)
        avg_rack_cap = total_rack_cap / max(num_batches, 1)
        avg_schedule = total_schedule / max(num_batches, 1)
        avg_balance = total_balance / max(num_batches, 1)
        avg_type_match = total_type_match / max(num_batches, 1)
        
        # Log
        log_entry = {
            'epoch': ep,
            'loss': avg_loss,
            'flight_capacity': avg_flight_cap,
            'rack_capacity': avg_rack_cap,
            'schedule': avg_schedule,
            'balance': avg_balance,
            'type_matching': avg_type_match,
            'lr': current_lr,
            'time': epoch_time
        }
        training_log.append(log_entry)
        
        # Print progress
        if ep % args.eval_every == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}  time={epoch_time:.1f}s")
            print(f"   Constraints: flight={avg_flight_cap:.3f} rack={avg_rack_cap:.3f} "
                  f"sched={avg_schedule:.3f} bal={avg_balance:.3f} type={avg_type_match:.3f}")
            
            # Save checkpoint if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(args.checkpoint_dir, 'beluga_trm_best.pt')
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'model_dims': model_dims,
                    'args': vars(args)
                }, save_path)
                print(f"   💾 Saved best checkpoint (loss: {avg_loss:.4f})")
        else:
            # Quick progress update
            print(f"Epoch {ep:3d}/{epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}  time={epoch_time:.1f}s")
    
    # Save final checkpoint
    final_save_path = os.path.join(args.checkpoint_dir, 'beluga_trm_final.pt')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'model_dims': model_dims,
        'args': vars(args)
    }, final_save_path)
    
    print(f"\n✅ Training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Final loss: {avg_loss:.4f}")
    print(f"   Best checkpoint: {os.path.join(args.checkpoint_dir, 'beluga_trm_best.pt')}")
    print(f"   Final checkpoint: {final_save_path}")
    
    # Save training log
    import csv
    log_path = os.path.join(args.log_dir, f'beluga_training_log_{int(time.time())}.csv')
    with open(log_path, 'w', newline='') as f:
        if training_log:
            writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
            writer.writeheader()
            writer.writerows(training_log)
    print(f"   Training log: {log_path}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Evaluate on validation set")
    print(f"   2. Visualize constraint satisfaction over epochs")
    print(f"   3. Test on held-out problems")


if __name__ == "__main__":
    main()