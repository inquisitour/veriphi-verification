#!/usr/bin/env python3
"""
Train BelugaTRM on Airbus logistics constraint satisfaction problems
Optimizes jig assignments to satisfy flight, rack, and schedule constraints

Correct AMP scaler usage following PyTorch best practices
"""

import os
import sys
import time
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import nvtx  

NVTX_COLORS = {
    "TRAIN": "blue",
    "EVAL": "orange",
    "EPOCH": "purple",
    "BATCH": "green",
    "H2D": "brown",
    "FWD": "royalblue",
    "LOSS": "red",
    "BWD": "crimson",
    "CLIP": "deeppink",
    "OPT": "teal",
    "SCHED": "slategray",
    "CKPT": "gold",
    "DATA": "gray",
}

def nv_range(name: str, color="gray", domain="BelugaTRM", **kwargs):
    """Minimal NVTX wrapper with optional metadata appended to the message."""
    meta = " ".join(f"{k}={v}" for k, v in kwargs.items())
    msg = f"{name}" + (f" [{meta}]" if meta else "")
    return nvtx.annotate(message=msg, color=color, domain=domain)

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
    Evaluate model on validation set with NVTX ranges
    """
    with nv_range("EVAL", color=NVTX_COLORS["EVAL"], epoch=epoch):
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
                with nv_range("BATCH", color=NVTX_COLORS["BATCH"], epoch=epoch):
                    # Currently batch_size=1, so problem_batch is list with 1 element
                    problem = problem_batch[0] if isinstance(problem_batch, (list, tuple)) else problem_batch

                    with nv_range("H2D", color=NVTX_COLORS["H2D"], epoch=epoch):
                        state_batch = state_batch.to(device)

                    with nv_range("FWD", color=NVTX_COLORS["FWD"], epoch=epoch):
                        # Forward pass
                        logits = model(state_batch)

                    with nv_range("LOSS", color=NVTX_COLORS["LOSS"], epoch=epoch):
                        # Compute constraint loss
                        loss = model.compute_constraint_loss(logits, problem)

                    total_loss += loss.item()
                    
                    # Get constraint breakdown
                    assignments = torch.argmax(logits.view(1, model.num_jigs, model.action_space), dim=-1)
                    probs = torch.softmax(logits.view(1, model.num_jigs, model.action_space), dim=-1)
                    
                    # Create jig mask for individual constraint loss computation
                    actual_num_jigs = problem.num_jigs
                    jig_mask = torch.zeros(model.num_jigs, dtype=torch.bool, device=device)
                    jig_mask[:actual_num_jigs] = True

                    total_flight_cap += model._compute_flight_capacity_loss(probs[0], problem, jig_mask).item()
                    total_rack_cap   += model._compute_rack_capacity_loss(probs[0], problem, jig_mask).item()
                    total_schedule   += model._compute_schedule_constraint_loss(probs[0], problem, jig_mask).item()
                    total_balance    += model._compute_flight_balance_loss(probs[0], problem, jig_mask).item()
                    total_type_match += model._compute_type_matching_loss(probs[0], problem, jig_mask).item()
                    
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

    with nv_range("DATA", color=NVTX_COLORS["DATA"], split="training"):
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
    with nv_range("DATA", color=NVTX_COLORS["DATA"], action="create_model"):
        model = create_beluga_trm(**model_dims)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params:,}")
    print(f"   Model on device: {next(model.parameters()).device}")

    # Optimizer and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    checkpoint_path = os.path.join(args.checkpoint_dir, 'beluga_trm_best.pt')
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        with nv_range("CKPT", color=NVTX_COLORS["CKPT"], action="load_best"):
            print(f"📂 Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"   Resuming from epoch {start_epoch}")

    # AMP disabled due to scaler initialization bug in PyTorch
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n🔥 Starting training...")
    print(f"   AMP enabled: {use_amp}")
    print(f"=" * 60)

    with nv_range("TRAIN", color=NVTX_COLORS["TRAIN"]):
        best_loss = float('inf')
        training_log = []

        for ep in range(start_epoch, epochs + 1):
            with nv_range("EPOCH", color=NVTX_COLORS["EPOCH"], epoch=ep):
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
                    with nv_range("BATCH", color=NVTX_COLORS["BATCH"], epoch=ep, batch=batch_idx):
                        # Get problem (batch_size=1 typically)
                        problem = problem_batch if not isinstance(problem_batch, (list, tuple)) else problem_batch[0]

                        with nv_range("H2D", color=NVTX_COLORS["H2D"], epoch=ep, batch=batch_idx):
                            state_batch = state_batch.to(DEVICE, non_blocking=False)

                        # Training step
                        opt.zero_grad(set_to_none=True)

                        # Correct AMP pattern following PyTorch docs
                        # https://pytorch.org/docs/stable/notes/amp_examples.html
                        with nv_range("FWD", color=NVTX_COLORS["FWD"], epoch=ep, batch=batch_idx):
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                # Forward pass
                                logits = model(state_batch)

                        with nv_range("LOSS", color=NVTX_COLORS["LOSS"], epoch=ep, batch=batch_idx):
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                # Compute constraint loss
                                loss = model.compute_constraint_loss(logits, problem)

                        with nv_range("BWD", color=NVTX_COLORS["BWD"], epoch=ep, batch=batch_idx):
                            # ----- backward + clip + step (no AMP) -----
                            loss.backward()

                        with nv_range("CLIP", color=NVTX_COLORS["CLIP"], epoch=ep, batch=batch_idx):
                            # single clipping step
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                            print(f"   grad_norm={float(grad_norm):.6f}")

                        with nv_range("OPT", color=NVTX_COLORS["OPT"], epoch=ep, batch=batch_idx):
                            opt.step()

                        # Track constraint breakdown (without gradients)
                        with torch.no_grad():
                            probs = torch.softmax(logits.view(1, model.num_jigs, model.action_space), dim=-1)
                            
                            # Create jig mask for individual constraint loss computation
                            actual_num_jigs = problem.num_jigs
                            jig_mask = torch.zeros(model.num_jigs, dtype=torch.bool, device=DEVICE)
                            jig_mask[:actual_num_jigs] = True

                            total_flight_cap += model._compute_flight_capacity_loss(probs[0], problem, jig_mask).item()
                            total_rack_cap   += model._compute_rack_capacity_loss(probs[0], problem, jig_mask).item()
                            total_schedule   += model._compute_schedule_constraint_loss(probs[0], problem, jig_mask).item()
                            total_balance    += model._compute_flight_balance_loss(probs[0], problem, jig_mask).item()
                            total_type_match += model._compute_type_matching_loss(probs[0], problem, jig_mask).item()
                            total_loss += loss.item()

                            num_batches += 1

                with nv_range("SCHED", color=NVTX_COLORS["SCHED"], epoch=ep):
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
                        with nv_range("CKPT", color=NVTX_COLORS["CKPT"], action="save_best", epoch=ep):
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
    with nv_range("CKPT", color=NVTX_COLORS["CKPT"], action="save_final"):
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

    # Save training log
    with nv_range("DATA", color=NVTX_COLORS["DATA"], action="write_csv"):
        import csv
        log_path = os.path.join(args.log_dir, f'beluga_training_log_{int(time.time())}.csv')
        with open(log_path, 'w', newline='') as f:
            if training_log:
                writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
                writer.writeheader()
                writer.writerows(training_log)
        print(f"   Training log: {log_path}")

    print(f"\n✅ Training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Final loss: {avg_loss:.4f}")
    print(f"   Best checkpoint: {os.path.join(args.checkpoint_dir, 'beluga_trm_best.pt')}")
    print(f"   Final checkpoint: {final_save_path}")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Evaluate on validation set")
    print(f"   2. Visualize constraint satisfaction over epochs")
    print(f"   3. Test on held-out problems")


if __name__ == "__main__":
    main()
