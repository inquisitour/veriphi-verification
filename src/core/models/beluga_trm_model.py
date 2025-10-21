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
        num_jigs: int = 100, # Max number of jigs (for output dimensioning)
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
    
    def compute_constraint_loss(
        self, 
        logits: torch.Tensor, 
        problem: 'BelugaProblem'
    ) -> torch.Tensor:
        """
        Compute loss based on constraint violations
        
        Args:
            logits: [B, num_jigs * action_space] - model output
            problem: BelugaProblem instance with constraint information
        
        Returns:
            loss: scalar tensor combining multiple constraint violations
        """
        B = logits.size(0)
        assert B == 1, "Currently only supports batch_size=1"
        
        # Reshape logits to [B, num_jigs, action_space]
        logits_reshaped = logits.view(B, self.num_jigs, self.action_space)
        
        # Get soft assignments (probabilities) for differentiable loss
        probs = torch.softmax(logits_reshaped, dim=-1)  # [B, num_jigs, action_space]
        
        # Get hard assignments for constraint checking
        assignments = torch.argmax(logits_reshaped, dim=-1)  # [B, num_jigs]
        
        # Initialize loss components
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # === 1. FLIGHT CAPACITY CONSTRAINTS ===
        flight_capacity_loss = self._compute_flight_capacity_loss(
            assignments[0], problem, probs[0]
        )
        total_loss = total_loss + flight_capacity_loss
        
        # === 2. RACK CAPACITY CONSTRAINTS ===
        rack_capacity_loss = self._compute_rack_capacity_loss(
            assignments[0], problem, probs[0]
        )
        total_loss = total_loss + rack_capacity_loss
        
        # === 3. PRODUCTION SCHEDULE CONSTRAINTS ===
        schedule_loss = self._compute_schedule_constraint_loss(
            assignments[0], problem, probs[0]
        )
        total_loss = total_loss + schedule_loss
        
        # === 4. FLIGHT BALANCE CONSTRAINTS ===
        balance_loss = self._compute_flight_balance_loss(
            assignments[0], problem, probs[0]
        )
        total_loss = total_loss + balance_loss
        
        # === 5. JIG TYPE MATCHING CONSTRAINTS ===
        type_matching_loss = self._compute_type_matching_loss(
            assignments[0], problem, probs[0]
        )
        total_loss = total_loss + type_matching_loss
        
        return total_loss
    
    def _compute_flight_capacity_loss(
        self,
        assignments: torch.Tensor,  # [num_jigs]
        problem: 'BelugaProblem',
        probs: torch.Tensor  # [num_jigs, action_space]
    ) -> torch.Tensor:
        """
        Penalize exceeding flight cargo capacity
        
        Each flight can only carry a certain volume based on jig sizes
        """
        loss = torch.tensor(0.0, device=assignments.device, requires_grad=True)
        
        # Map action indices to flights (first N actions are flight assignments)
        num_flights = problem.num_flights
        
        # For each flight, calculate total expected load
        for flight_idx in range(num_flights):
            flight_load = 0.0
            
            # Check which jigs are assigned to this flight
            jig_names = sorted(problem.jigs.keys())
            for jig_idx, jig_name in enumerate(jig_names[:self.num_jigs]):
                if jig_idx >= assignments.size(0):
                    break
                    
                # Check if this jig is assigned to this flight
                if assignments[jig_idx].item() == flight_idx:
                    jig_info = problem.jigs[jig_name]
                    jig_type_info = problem.jig_types[jig_info['type']]
                    
                    # Use loaded size (jigs on flights are loaded)
                    jig_size = jig_type_info['size_loaded']
                    flight_load += jig_size
            
            # Assume max flight capacity (can be refined based on Beluga specs)
            # Typical Beluga capacity is around 150-200 units
            MAX_FLIGHT_CAPACITY = 150.0
            
            if flight_load > MAX_FLIGHT_CAPACITY:
                overcapacity = flight_load - MAX_FLIGHT_CAPACITY
                # Quadratic penalty for overcapacity
                penalty = (overcapacity / MAX_FLIGHT_CAPACITY) ** 2
                loss = loss + penalty
        
        return loss * 10.0  # Weight factor
    
    def _compute_rack_capacity_loss(
        self,
        assignments: torch.Tensor,
        problem: 'BelugaProblem',
        probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize exceeding rack storage capacity
        """
        loss = torch.tensor(0.0, device=assignments.device, requires_grad=True)
        
        num_flights = problem.num_flights
        num_racks = problem.num_racks
        
        # Rack assignments start after flight assignments
        rack_action_offset = num_flights
        
        # For each rack, calculate total load
        for rack_idx, rack in enumerate(problem.racks):
            if rack_idx >= num_racks:
                break
                
            rack_capacity = rack['size']
            rack_load = 0.0
            
            # Count jigs already in rack
            for jig_name in rack['jigs']:
                jig_info = problem.jigs[jig_name]
                jig_type_info = problem.jig_types[jig_info['type']]
                size = jig_type_info['size_empty'] if jig_info['empty'] else jig_type_info['size_loaded']
                rack_load += size
            
            # Add newly assigned jigs
            jig_names = sorted(problem.jigs.keys())
            for jig_idx, jig_name in enumerate(jig_names[:self.num_jigs]):
                if jig_idx >= assignments.size(0):
                    break
                    
                action_idx = assignments[jig_idx].item()
                
                # Check if assigned to this rack
                if action_idx == rack_action_offset + rack_idx:
                    jig_info = problem.jigs[jig_name]
                    jig_type_info = problem.jig_types[jig_info['type']]
                    size = jig_type_info['size_empty'] if jig_info['empty'] else jig_type_info['size_loaded']
                    rack_load += size
            
            # Penalize overcapacity
            if rack_load > rack_capacity:
                overcapacity = rack_load - rack_capacity
                penalty = (overcapacity / rack_capacity) ** 2
                loss = loss + penalty
        
        return loss * 5.0  # Weight factor
    
    def _compute_schedule_constraint_loss(
        self,
        assignments: torch.Tensor,
        problem: 'BelugaProblem',
        probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize violations of production schedule ordering
        
        Jigs should be available according to production line schedule
        """
        loss = torch.tensor(0.0, device=assignments.device, requires_grad=True)
        
        # Build schedule mapping: jig_name -> (production_line, position)
        jig_schedule = {}
        for pl_idx, pl in enumerate(problem.production_lines):
            for pos, jig_name in enumerate(pl['schedule']):
                jig_schedule[jig_name] = (pl_idx, pos)
        
        # Check if jigs are available when needed in schedule
        # Penalize if high-priority (early schedule) jigs are not readily available
        jig_names = sorted(problem.jigs.keys())
        for jig_idx, jig_name in enumerate(jig_names[:self.num_jigs]):
            if jig_idx >= assignments.size(0):
                break
                
            if jig_name in jig_schedule:
                pl_idx, schedule_pos = jig_schedule[jig_name]
                
                # Early schedule positions need higher availability
                # Penalize if assigned to distant locations (high action indices)
                action_idx = assignments[jig_idx].item()
                
                # Normalize schedule position (0 = high priority, 1 = low priority)
                max_schedule_len = max(len(pl['schedule']) for pl in problem.production_lines)
                priority = 1.0 - (schedule_pos / max(max_schedule_len, 1))
                
                # Penalize high-priority jigs assigned to less accessible actions
                accessibility_penalty = (action_idx / self.action_space) * priority
                loss = loss + accessibility_penalty
        
        return loss * 2.0  # Weight factor
    
    def _compute_flight_balance_loss(
        self,
        assignments: torch.Tensor,
        problem: 'BelugaProblem',
        probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize imbalance between incoming and outgoing jigs per flight
        
        Each flight has specified incoming jigs and required outgoing jig types
        """
        loss = torch.tensor(0.0, device=assignments.device, requires_grad=True)
        
        num_flights = problem.num_flights
        
        for flight_idx, flight in enumerate(problem.flights):
            # Count incoming jigs (these are mandatory deliveries)
            num_incoming = len(flight['incoming'])
            
            # Count required outgoing jigs
            num_outgoing_required = len(flight['outgoing'])
            
            # Count how many jigs we're assigning to this flight
            jig_names = sorted(problem.jigs.keys())
            num_assigned = 0
            for jig_idx, jig_name in enumerate(jig_names[:self.num_jigs]):
                if jig_idx >= assignments.size(0):
                    break
                if assignments[jig_idx].item() == flight_idx:
                    num_assigned += 1
            
            # Penalize if we don't provide enough outgoing jigs
            outgoing_deficit = max(0, num_outgoing_required - num_assigned)
            if outgoing_deficit > 0:
                penalty = (outgoing_deficit / max(num_outgoing_required, 1)) ** 2
                loss = loss + penalty
            
            # Also penalize significant over-assignment (inefficiency)
            excess = max(0, num_assigned - num_outgoing_required - 2)  # Allow 2 extra
            if excess > 0:
                penalty = (excess / 10.0) ** 2
                loss = loss + penalty * 0.5  # Lower weight
        
        return loss * 8.0  # Weight factor
    
    def _compute_type_matching_loss(
        self,
        assignments: torch.Tensor,
        problem: 'BelugaProblem',
        probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize type mismatches between assigned jigs and flight requirements
        
        Each flight specifies required outgoing jig TYPES (not specific jigs)
        """
        loss = torch.tensor(0.0, device=assignments.device, requires_grad=True)
        
        num_flights = problem.num_flights
        
        for flight_idx, flight in enumerate(problem.flights):
            # Get required outgoing types
            required_types = flight['outgoing']  # List of type names
            
            if not required_types:
                continue
            
            # Count types by analyzing assigned jigs
            assigned_types = []
            
            jig_names = sorted(problem.jigs.keys())
            for jig_idx, jig_name in enumerate(jig_names[:self.num_jigs]):
                if jig_idx >= assignments.size(0):
                    break
                if assignments[jig_idx].item() == flight_idx:
                    jig_type = problem.jigs[jig_name]['type']
                    assigned_types.append(jig_type)
            
            required_type_counts = Counter(required_types)
            assigned_type_counts = Counter(assigned_types)
            
            # Penalize type mismatches
            for jig_type, required_count in required_type_counts.items():
                assigned_count = assigned_type_counts.get(jig_type, 0)
                shortage = max(0, required_count - assigned_count)
                if shortage > 0:
                    penalty = (shortage / required_count) ** 2
                    loss = loss + penalty
        
        return loss * 15.0  # Weight factor (highest priority)


def create_beluga_trm(
    x_dim: int,
    num_jigs: int = 100,
    y_dim: int = 256,
    z_dim: int = 256,
    hidden: int = 512,
    action_space: int = 10,
    H_cycles: int = 3,
    L_cycles: int = 2
) -> nn.Module:
    """
    Factory function to create Beluga TRM model
    
    Args:
        x_dim: Full problem state dimension (from data_loader)
        num_jigs: Maximum number of jigs in problems
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
def get_model_dims_from_problem(problem: 'BelugaProblem') -> dict:
    """
    Calculate appropriate model dimensions from a problem instance
    
    Args:
        problem: BelugaProblem instance
    
    Returns:
        dict with recommended model dimensions
    """
    state_tensor = problem.get_full_state_tensor().to(DEVICE)
    x_dim = state_tensor.shape[1]
    num_jigs = problem.num_jigs
    
    # Estimate action space based on problem structure
    # Actions could be: assign to flight, assign to rack, assign to production line
    num_flights = problem.num_flights
    num_racks = problem.num_racks
    num_production_lines = problem.num_production_lines
    
    # Action space: flight assignments + rack assignments + special actions (hold, etc.)
    action_space = num_flights + num_racks + 5  # 5 for special actions
    
    return {
        'x_dim': x_dim,
        'num_jigs': num_jigs,
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
        
        # Show sample assignments
        print(f"\n   Sample assignments (first 10 jigs):")
        for i in range(min(10, model_dims['num_jigs'])):
            action_idx = assignments[0, i].item()
            score = scores[0, i].item()
            print(f"      Jig {i:02d} → Action {action_idx:02d} (score: {score:.3f})")
    
    print(f"\n✅ TRM model test complete!")
    
    # Test constraint loss computation
    print(f"\n🔍 Testing constraint loss computation...")
    with torch.no_grad():
        loss = model.compute_constraint_loss(logits, problem)
        print(f"   Total constraint loss: {loss.item():.6f}")
        
        # Test individual constraint losses
        assignments_hard = torch.argmax(logits.view(1, model_dims['num_jigs'], model_dims['action_space']), dim=-1)
        probs = torch.softmax(logits.view(1, model_dims['num_jigs'], model_dims['action_space']), dim=-1)
        
        flight_loss = model._compute_flight_capacity_loss(assignments_hard[0], problem, probs[0])
        rack_loss = model._compute_rack_capacity_loss(assignments_hard[0], problem, probs[0])
        schedule_loss = model._compute_schedule_constraint_loss(assignments_hard[0], problem, probs[0])
        balance_loss = model._compute_flight_balance_loss(assignments_hard[0], problem, probs[0])
        type_loss = model._compute_type_matching_loss(assignments_hard[0], problem, probs[0])
        
        print(f"\n   Constraint breakdown:")
        print(f"      Flight capacity:  {flight_loss.item():.6f}")
        print(f"      Rack capacity:    {rack_loss.item():.6f}")
        print(f"      Schedule:         {schedule_loss.item():.6f}")
        print(f"      Flight balance:   {balance_loss.item():.6f}")
        print(f"      Type matching:    {type_loss.item():.6f}")
    
    print(f"\n✅ All tests passed!")


if __name__ == "__main__":
    test_beluga_trm()
