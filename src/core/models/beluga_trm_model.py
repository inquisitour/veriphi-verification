"""
TRM-MLP Architecture for Beluga Logistics Optimization
Adapts Tiny Recursive Model for constraint satisfaction problems
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Tuple
from collections import Counter

DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


class MLPBlock(nn.Module):
    """Standard MLP block with residual connection"""
    
    def __init__(self, dim: int, hidden: int, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            act(),
            nn.Linear(hidden, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BelugaTRM(nn.Module):
    """
    TRM-MLP adapted for Beluga logistics constraint satisfaction
    
    Architecture:
      x: [B, Dx]   (problem state: jigs, flights, racks, schedule)
      y: [B, Dy]   (solution state: assignment decisions)
      z: [B, Dz]   (latent reasoning state)
    
    Update schedule per improvement step k:
      - Repeat H times: z <- f_z([x, y, z])  (constraint reasoning)
      - Then y <- f_y([y, z])                (solution refinement)
    
    Output: assignment logits for each jig
    All operations are Linear/ReLU → α,β-CROWN friendly
    """
    
    def __init__(
        self,
        x_dim: int,          # Full problem state dimension
        y_dim: int = 256,    # Solution state dimension
        z_dim: int = 256,    # Latent reasoning dimension
        hidden: int = 512,   # MLP hidden layer size
        num_jigs: int = 821, # Max number of jigs (UPDATED: was 100, now 821 to match dataset max)
        action_space: int = 10,  # Possible actions per jig (flight assignment, rack, etc.)
        H_cycles: int = 3,   # Constraint reasoning loops
        L_cycles: int = 2,   # Solution improvement steps
        init_scale: float = 0.02,
        act=nn.ReLU,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.num_jigs = num_jigs
        self.action_space = action_space
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        
        # Input encoder - compress problem state
        self.x_enc = nn.Sequential(
            nn.Linear(x_dim, x_dim // 2),
            act(),
            nn.Linear(x_dim // 2, x_dim // 2)
        )
        self.x_compressed_dim = x_dim // 2
        
        # Initial states (learnable)
        self.y_init = nn.Parameter(torch.zeros(1, y_dim))
        self.z_init = nn.Parameter(torch.zeros(1, z_dim))
        
        # Latent reasoning update blocks (constraint checking)
        self.fz_in = nn.Linear(self.x_compressed_dim + y_dim + z_dim, z_dim)
        self.fz_mlp = MLPBlock(z_dim, hidden, act=act)
        
        # Solution state update blocks
        self.fy_in = nn.Linear(y_dim + z_dim, y_dim)
        self.fy_mlp = MLPBlock(y_dim, hidden, act=act)
        
        # Output head - assignment logits for each jig
        # Output shape: [batch, num_jigs * action_space]
        self.head = nn.Sequential(
            nn.Linear(y_dim, hidden),
            act(),
            nn.Linear(hidden, num_jigs * action_space)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_scale)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, x_dim] - flattened problem state from BelugaProblem.get_full_state_tensor()
        
        Returns:
            logits: [B, num_jigs * action_space] - assignment scores for each jig
        """
        B = x.size(0)
        
        # Encode problem state
        x_encoded = self.x_enc(x)  # [B, x_dim/2]
        
        # Initialize solution and reasoning states
        y = self.y_init.expand(B, -1)  # [B, y_dim]
        z = self.z_init.expand(B, -1)  # [B, z_dim]
        
        # Recursive reasoning with constraint checking
        for improvement_step in range(self.L_cycles):
            # Inner loop: constraint reasoning
            for reasoning_cycle in range(self.H_cycles):
                # Update latent state with full context
                z_input = torch.cat([x_encoded, y, z], dim=-1)
                z = self.fz_in(z_input)
                z = self.fz_mlp(z)
            
            # Outer loop: refine solution based on reasoning
            y_input = torch.cat([y, z], dim=-1)
            y = self.fy_in(y_input)
            y = self.fy_mlp(y)
        
        # Generate assignment logits
        logits = self.head(y)  # [B, num_jigs * action_space]
        
        return logits
    
    def get_assignments(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get discrete jig assignments from model
        
        Args:
            x: [B, x_dim] - problem state
        
        Returns:
            assignments: [B, num_jigs] - action index for each jig
            scores: [B, num_jigs] - confidence scores
        """
        logits = self.forward(x)  # [B, num_jigs * action_space]
        
        # Reshape to [B, num_jigs, action_space]
        logits_reshaped = logits.view(-1, self.num_jigs, self.action_space)
        
        # Get best action for each jig
        scores, assignments = torch.max(logits_reshaped, dim=-1)
        
        return assignments, scores
        
    def compute_constraint_loss(self, logits: torch.Tensor, problem: 'BelugaProblem') -> torch.Tensor:
        """
        Differentiable constraint loss using soft probabilities.
        Uses soft expectations over actions instead of hard argmax/item(),
        so gradients flow back into logits and the model can learn.
        """
        B = logits.size(0)
        assert B == 1, "Currently supports batch_size=1"

        # [B, num_jigs, action_space]
        logits_reshaped = logits.view(B, self.num_jigs, self.action_space)
        probs = torch.softmax(logits_reshaped, dim=-1)  # differentiable "assignments"

        # mask: only compute for actual jigs, ignore padding
        actual_num_jigs = problem.num_jigs
        jig_mask = torch.zeros(self.num_jigs, dtype=torch.bool, device=logits.device)
        jig_mask[:actual_num_jigs] = True

        total_loss = (
            self._compute_flight_capacity_loss(probs[0], problem, jig_mask)
            + self._compute_rack_capacity_loss(probs[0], problem, jig_mask)
            + self._compute_schedule_constraint_loss(probs[0], problem, jig_mask)
            + self._compute_flight_balance_loss(probs[0], problem, jig_mask)
            + self._compute_type_matching_loss(probs[0], problem, jig_mask)
        )
        return total_loss / self.num_jigs  # normalize by number of jigs
    
    def _compute_flight_capacity_loss(self, probs, problem, jig_mask):
        """Soft, differentiable flight capacity penalty."""
        loss = 0.0

        # only iterate over action indices that represent flights
        num_flight_actions = min(problem.num_flights, self.action_space)
        MAX_CAP = 150.0
        jig_names = sorted(problem.jigs.keys())

        for f_idx in range(num_flight_actions):
            flight_load = 0.0
            for j_idx, jig_name in enumerate(jig_names):
                if not jig_mask[j_idx]:
                    continue
                jig_info = problem.jigs[jig_name]
                size_loaded = problem.jig_types[jig_info['type']]['size_loaded']
                flight_load = flight_load + probs[j_idx, f_idx] * float(size_loaded)

            over = torch.relu(flight_load - MAX_CAP)
            loss = loss + (over / MAX_CAP) ** 2

        return loss * 10.0 # Weight factor
    
    def _compute_rack_capacity_loss(self, probs, problem, jig_mask):
        """
        Penalize exceeding rack storage capacity (differentiable version).
        Handles truncated action spaces safely.
        """
        loss = torch.tensor(0.0, device=probs.device)
        num_flights = problem.num_flights
        num_racks = problem.num_racks
        rack_offset = num_flights
        jig_names = sorted(problem.jigs.keys())

        max_action_idx = self.action_space - 1

        for r_idx, rack in enumerate(problem.racks):
            # keep rack_load as tensor on same device
            rack_load = torch.tensor(0.0, device=probs.device)

            rack_capacity = float(rack["size"])

            # fixed current load from existing jigs (constant)
            for jig_name in rack["jigs"]:
                info = problem.jigs[jig_name]
                t = info["type"]
                size_const = (
                    problem.jig_types[t]["size_empty"]
                    if info["empty"]
                    else problem.jig_types[t]["size_loaded"]
                )
                rack_load = rack_load + torch.tensor(float(size_const), device=probs.device)

            # expected load from soft assignments, clamped
            for j_idx, jig_name in enumerate(jig_names):
                if not jig_mask[j_idx]:
                    continue
                info = problem.jigs[jig_name]
                t = info["type"]
                size = (
                    problem.jig_types[t]["size_empty"]
                    if info["empty"]
                    else problem.jig_types[t]["size_loaded"]
                )
                action_idx = rack_offset + r_idx
                if action_idx > max_action_idx:
                    continue
                rack_load = rack_load + probs[j_idx, action_idx] * float(size)

            # ensure both are tensors for relu
            over = torch.relu(rack_load - torch.tensor(rack_capacity, device=probs.device))
            loss = loss + (over / rack_capacity) ** 2

        return loss * 5.0 # Weight factor
    
    def _compute_schedule_constraint_loss(
        self,
        probs: torch.Tensor,
        problem: 'BelugaProblem',
        jig_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft penalty for misaligned production schedule.
        High-priority (early) jigs should have lower expected action index.
        Clamps to model.action_space to avoid index overflow.
        """
        loss = torch.tensor(0.0, device=probs.device)

        # build mapping jig_name → (production_line, position)
        jig_schedule = {}
        for pl_idx, pl in enumerate(problem.production_lines):
            for pos, jig_name in enumerate(pl.get("schedule", [])):
                jig_schedule[jig_name] = (pl_idx, pos)

        # schedule statistics
        max_len = max((len(pl.get("schedule", [])) for pl in problem.production_lines), default=1)
        num_actions = self.action_space
        action_idx = torch.arange(num_actions, device=probs.device, dtype=torch.float32)
        denom = max(num_actions - 1, 1)

        jig_names = sorted(problem.jigs.keys())
        for j_idx, jig_name in enumerate(jig_names):
            if not jig_mask[j_idx] or jig_name not in jig_schedule:
                continue

            _, pos = jig_schedule[jig_name]
            priority = 1.0 - (float(pos) / float(max_len))  # scalar weight

            # expected normalized action index in [0,1]
            exp_action = (probs[j_idx, :num_actions] * action_idx).sum() / denom
            loss = loss + exp_action * priority

        return loss * 2.0  # weight factor
    
    def _compute_flight_balance_loss(self, probs, problem, jig_mask):
        """
        Soft load-balance constraint: match incoming and outgoing per flight.
        Clamps to model.action_space to avoid out-of-bounds indices.
        """
        loss = torch.tensor(0.0, device=probs.device)
        num_flights = min(problem.num_flights, self.action_space)
        jig_names = sorted(problem.jigs.keys())

        for f_idx, flight in enumerate(problem.flights[:num_flights]):
            # constant incoming load (float, independent of params)
            incoming_load = 0.0
            for j in flight["incoming"]:
                if j in problem.jigs:
                    t = problem.jigs[j]["type"]
                    incoming_load += float(problem.jig_types[t]["size_loaded"])

            outgoing_load = torch.tensor(0.0, device=probs.device)

            for j_idx, jig_name in enumerate(jig_names):
                if not jig_mask[j_idx]:
                    continue
                t = problem.jigs[jig_name]["type"]
                size_empty = float(problem.jig_types[t]["size_empty"])
                # f_idx is guaranteed < self.action_space
                outgoing_load = outgoing_load + probs[j_idx, f_idx] * size_empty

            imbalance = torch.abs(outgoing_load - incoming_load)
            max_load = max(incoming_load, 1.0)
            loss = loss + (imbalance / max_load) ** 2

        return loss * 0.01 # Weight factor
    
    def _compute_type_matching_loss(
        self,
        probs: torch.Tensor,
        problem: 'BelugaProblem',
        jig_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize assigning jigs of the wrong TYPE to a flight that expects specific outgoing types.
        Soft version: sum of probabilities assigned to flights where type ∉ required_types.
        Clamps to model.action_space to stay in range.
        """
        loss = torch.tensor(0.0, device=probs.device)
        num_flights = min(problem.num_flights, self.action_space)
        jig_names = sorted(problem.jigs.keys())

        for f_idx, flight in enumerate(problem.flights[:num_flights]):
            required_types = set(flight.get("outgoing", []))
            if not required_types:
                continue

            for j_idx, jig_name in enumerate(jig_names):
                if not jig_mask[j_idx]:
                    continue
                t = problem.jigs[jig_name]["type"]
                if t not in required_types:
                    # probability mass assigned to this flight for this jig
                    loss = loss + probs[j_idx, f_idx]

        return loss * 15.0  # strongest weight


def create_beluga_trm(
    x_dim: int,
    num_jigs: int = 821,  # UPDATED: Changed default from 100 to 821 (dataset maximum)
    y_dim: int = 256,
    z_dim: int = 256,
    hidden: int = 512,
    action_space: int = 10,
    H_cycles: int = 3,
    L_cycles: int = 2
) -> nn.Module:
    """
    Factory function to create Beluga TRM model
    
    UPDATED: Default num_jigs is now 821 to match dataset maximum
    
    Args:
        x_dim: Full problem state dimension (from data_loader)
        num_jigs: Maximum number of jigs in problems (default 821)
        y_dim: Solution state dimension
        z_dim: Latent reasoning dimension
        hidden: Hidden layer size
        action_space: Number of possible actions per jig
        H_cycles: Constraint reasoning loops
        L_cycles: Solution improvement steps
    
    Returns:
        BelugaTRM model on DEVICE
    """
    model = BelugaTRM(
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
        hidden=hidden,
        num_jigs=num_jigs,
        action_space=action_space,
        H_cycles=H_cycles,
        L_cycles=L_cycles
    )
    
    return model.to(DEVICE)


# Utility function for determining model dimensions from dataset
def get_model_dims_from_problem(problem: 'BelugaProblem', max_jigs: int = 821) -> dict:
    """
    Calculate appropriate model dimensions from a problem instance
    
    UPDATED: Added max_jigs parameter (default 821) to ensure model can handle all problems
    
    Args:
        problem: BelugaProblem instance
        max_jigs: Maximum number of jigs across entire dataset (default 821)
    
    Returns:
        dict with recommended model dimensions
    """
    state_tensor = problem.get_full_state_tensor().to(DEVICE)
    x_dim = state_tensor.shape[1]
    
    # UPDATED: Use max_jigs parameter instead of problem.num_jigs
    # This ensures the model can handle ANY problem in the dataset
    num_jigs = max_jigs
    
    # Estimate action space based on problem structure
    # Actions could be: assign to flight, assign to rack, assign to production line
    num_flights = problem.num_flights
    num_racks = problem.num_racks
    num_production_lines = problem.num_production_lines
    
    # Action space: flight assignments + rack assignments + special actions (hold, etc.)
    action_space = num_flights + num_racks + 5  # 5 for special actions
    
    return {
        'x_dim': x_dim,
        'num_jigs': num_jigs,  # UPDATED: Now uses max_jigs (821)
        'action_space': action_space,
        'y_dim': min(256, x_dim // 4),  # Scale y_dim with problem complexity
        'z_dim': min(256, x_dim // 4),
        'hidden': min(512, x_dim // 2),
        'H_cycles': 3,  # Fixed for now
        'L_cycles': 2
    }


# Test function
def test_beluga_trm():
    """Test TRM model with dummy data"""
    import sys
    from pathlib import Path
    
    # Add src to path for imports
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root))
    
    from core.models.beluga_data_loader import BelugaProblem
    
    print("🧪 Testing Beluga TRM Model")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python trm_beluga_model.py <path_to_problem.json>")
        return
    
    problem_path = sys.argv[1]
    
    # Load problem
    print(f"\n📂 Loading problem: {Path(problem_path).name}")
    problem = BelugaProblem.from_json(problem_path)
    
    # Get model dimensions
    model_dims = get_model_dims_from_problem(problem)
    print(f"\n📊 Model Dimensions:")
    for key, value in model_dims.items():
        print(f"   {key}: {value}")
    
    print(f"\n   NOTE: Model built with max_jigs=821 (dataset maximum)")
    print(f"   Current problem has {problem.num_jigs} jigs")
    print(f"   Padding will be handled via masking in constraint loss")
    
    # Create model
    print(f"\n🔨 Creating BelugaTRM model...")
    model = create_beluga_trm(**model_dims)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params:,}")
    print(f"   Device: {next(model.parameters()).device}")
    
    # Test forward pass
    print(f"\n🚀 Testing forward pass...")
    state_tensor = problem.get_full_state_tensor().to(DEVICE)  # [1, x_dim]
    
    print(f"   Input shape: {state_tensor.shape}")
    
    with torch.no_grad():
        logits = model(state_tensor)
        print(f"   Output logits shape: {logits.shape}")
        print(f"   Expected: [1, {model_dims['num_jigs']} * {model_dims['action_space']}] = [1, {model_dims['num_jigs'] * model_dims['action_space']}]")
        
        # Get discrete assignments
        assignments, scores = model.get_assignments(state_tensor)
        print(f"\n   Assignment shape: {assignments.shape}")
        print(f"   Score shape: {scores.shape}")
        
        # Show sample assignments (only for actual jigs, not padding)
        print(f"\n   Sample assignments (first 10 actual jigs, ignoring {model_dims['num_jigs'] - problem.num_jigs} padded positions):")
        for i in range(min(10, problem.num_jigs)):
            action_idx = assignments[0, i].item()
            score = scores[0, i].item()
            print(f"      Jig {i:02d} → Action {action_idx:02d} (score: {score:.3f})")
    
    print(f"\n✅ TRM model test complete!")
    
    # Test constraint loss computation
    print(f"\n🔍 Testing constraint loss computation with masking...")
    with torch.no_grad():
        loss = model.compute_constraint_loss(logits, problem)
        print(f"   Total constraint loss: {loss.item():.6f}")
        print(f"   (Loss computed only for {problem.num_jigs} actual jigs, ignoring {model_dims['num_jigs'] - problem.num_jigs} padded positions)")
        
        # Test individual constraint losses
        assignments_hard = torch.argmax(logits.view(1, model_dims['num_jigs'], model_dims['action_space']), dim=-1)
        probs = torch.softmax(logits.view(1, model_dims['num_jigs'], model_dims['action_space']), dim=-1)
        
        # Create mask for testing
        jig_mask = torch.zeros(model_dims['num_jigs'], dtype=torch.bool, device=DEVICE)
        jig_mask[:problem.num_jigs] = True
        
        flight_loss = model._compute_flight_capacity_loss(assignments_hard[0], problem, probs[0], jig_mask)
        rack_loss = model._compute_rack_capacity_loss(assignments_hard[0], problem, probs[0], jig_mask)
        schedule_loss = model._compute_schedule_constraint_loss(assignments_hard[0], problem, probs[0], jig_mask)
        balance_loss = model._compute_flight_balance_loss(assignments_hard[0], problem, probs[0], jig_mask)
        type_loss = model._compute_type_matching_loss(assignments_hard[0], problem, probs[0], jig_mask)
        
        print(f"\n   Constraint breakdown:")
        print(f"      Flight capacity:  {flight_loss.item():.6f}")
        print(f"      Rack capacity:    {rack_loss.item():.6f}")
        print(f"      Schedule:         {schedule_loss.item():.6f}")
        print(f"      Flight balance:   {balance_loss.item():.6f}")
        print(f"      Type matching:    {type_loss.item():.6f}")
    
    print(f"\n✅ All tests passed!")


if __name__ == "__main__":
    test_beluga_trm()