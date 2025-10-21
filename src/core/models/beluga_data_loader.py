"""
Beluga Logistics Dataset Parser
Parses JSON constraint satisfaction problems for TRM training
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

#DEVICE = torch.device(os.environ.get("VERIPHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DEVICE = torch.device("cpu")

@dataclass
class BelugaProblem:
    """Represents a single Beluga logistics problem instance"""
    
    # Raw data
    trailers_beluga: List[Dict]
    trailers_factory: List[Dict]
    hangars: List[str]
    jig_types: Dict[str, Dict]
    racks: List[Dict]
    jigs: Dict[str, Dict]
    production_lines: List[Dict]
    flights: List[Dict]
    
    # Metadata
    problem_id: str
    num_jigs: int
    num_flights: int
    num_racks: int
    num_production_lines: int
    
    # Feature tensors (computed lazily)
    _jig_features: Optional[torch.Tensor] = None
    _flight_features: Optional[torch.Tensor] = None
    _rack_features: Optional[torch.Tensor] = None
    _constraint_matrix: Optional[torch.Tensor] = None
    
    @classmethod
    def from_json(cls, filepath: str) -> 'BelugaProblem':
        """Load problem from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        problem_id = Path(filepath).stem
        
        return cls(
            trailers_beluga=data['trailers_beluga'],
            trailers_factory=data['trailers_factory'],
            hangars=data['hangars'],
            jig_types=data['jig_types'],
            racks=data['racks'],
            jigs=data['jigs'],
            production_lines=data['production_lines'],
            flights=data['flights'],
            problem_id=problem_id,
            num_jigs=len(data['jigs']),
            num_flights=len(data['flights']),
            num_racks=len(data['racks']),
            num_production_lines=len(data['production_lines'])
        )
    
    def get_jig_features(self) -> torch.Tensor:
        """
        Encode all jigs as feature tensors
        Returns: [num_jigs, jig_feature_dim]
        
        Features per jig:
        - type_one_hot: 5 dims (typeA-E)
        - empty_flag: 1 dim (binary)
        - size_empty: 1 dim (normalized)
        - size_loaded: 1 dim (normalized)
        Total: 8 dims per jig
        """
        if self._jig_features is not None:
            return self._jig_features
        
        # Type mapping
        type_to_idx = {name: i for i, name in enumerate(['typeA', 'typeB', 'typeC', 'typeD', 'typeE'])}
        
        # Max sizes for normalization
        max_size_empty = max(jt['size_empty'] for jt in self.jig_types.values())
        max_size_loaded = max(jt['size_loaded'] for jt in self.jig_types.values())
        
        features = []
        for jig_name in sorted(self.jigs.keys()):
            jig = self.jigs[jig_name]
            jig_type_info = self.jig_types[jig['type']]
            
            # One-hot encode type
            type_onehot = [0.0] * 5
            type_onehot[type_to_idx[jig['type']]] = 1.0
            
            # Binary empty flag
            empty_flag = [1.0 if jig['empty'] else 0.0]
            
            # Normalized sizes
            size_empty_norm = [jig_type_info['size_empty'] / max_size_empty]
            size_loaded_norm = [jig_type_info['size_loaded'] / max_size_loaded]
            
            # Concatenate all features
            jig_feat = type_onehot + empty_flag + size_empty_norm + size_loaded_norm
            features.append(jig_feat)
        
        self._jig_features = torch.tensor(features, dtype=torch.float32, device=DEVICE)
        return self._jig_features
    
    def get_flight_features(self) -> torch.Tensor:
        """
        Encode all flights as feature tensors
        Returns: [num_flights, flight_feature_dim]
        
        Features per flight:
        - incoming_count: 1 dim (number of incoming jigs)
        - outgoing_count: 1 dim (number of outgoing jig types requested)
        - total_incoming_volume: 1 dim (sum of sizes, normalized)
        - total_outgoing_volume_required: 1 dim (normalized)
        - cargo_balance: 1 dim (outgoing - incoming, can be negative)
        Total: 5 dims per flight
        """
        if self._flight_features is not None:
            return self._flight_features
        
        # Compute max volumes for normalization
        max_volume = 0.0
        for flight in self.flights:
            incoming_vol = sum(
                self.jig_types[self.jigs[jig_name]['type']]['size_loaded']
                for jig_name in flight['incoming']
            )
            outgoing_vol = sum(
                self.jig_types[jig_type]['size_empty']
                for jig_type in flight['outgoing']
            )
            max_volume = max(max_volume, incoming_vol, outgoing_vol)
        
        if max_volume == 0:
            max_volume = 1.0  # Avoid division by zero
        
        features = []
        for flight in self.flights:
            incoming_count = len(flight['incoming'])
            outgoing_count = len(flight['outgoing'])
            
            # Calculate volumes
            incoming_vol = sum(
                self.jig_types[self.jigs[jig_name]['type']]['size_loaded']
                for jig_name in flight['incoming']
            ) if incoming_count > 0 else 0.0
            
            outgoing_vol = sum(
                self.jig_types[jig_type]['size_empty']
                for jig_type in flight['outgoing']
            ) if outgoing_count > 0 else 0.0
            
            # Normalize and compute balance
            incoming_vol_norm = incoming_vol / max_volume
            outgoing_vol_norm = outgoing_vol / max_volume
            cargo_balance = outgoing_vol_norm - incoming_vol_norm
            
            flight_feat = [
                float(incoming_count),
                float(outgoing_count),
                incoming_vol_norm,
                outgoing_vol_norm,
                cargo_balance
            ]
            features.append(flight_feat)
        
        self._flight_features = torch.tensor(features, dtype=torch.float32, device=DEVICE)
        return self._flight_features
    
    def get_rack_features(self) -> torch.Tensor:
        """
        Encode all racks as feature tensors
        Returns: [num_racks, rack_feature_dim]
        
        Features per rack:
        - capacity: 1 dim (normalized)
        - current_load: 1 dim (sum of jig sizes, normalized)
        - available_space: 1 dim (capacity - load, normalized)
        - num_jigs_stored: 1 dim (count)
        - utilization: 1 dim (load / capacity)
        Total: 5 dims per rack
        """
        if self._rack_features is not None:
            return self._rack_features
        
        max_capacity = max(rack['size'] for rack in self.racks)
        
        features = []
        for rack in self.racks:
            capacity = rack['size']
            
            # Calculate current load
            current_load = 0.0
            for jig_name in rack['jigs']:
                jig_type = self.jigs[jig_name]['type']
                jig_empty = self.jigs[jig_name]['empty']
                size = self.jig_types[jig_type]['size_empty'] if jig_empty else self.jig_types[jig_type]['size_loaded']
                current_load += size
            
            available_space = capacity - current_load
            num_jigs = len(rack['jigs'])
            utilization = current_load / capacity if capacity > 0 else 0.0
            
            rack_feat = [
                capacity / max_capacity,
                current_load / max_capacity,
                available_space / max_capacity,
                float(num_jigs),
                utilization
            ]
            features.append(rack_feat)
        
        self._rack_features = torch.tensor(features, dtype=torch.float32, device=DEVICE)
        return self._rack_features
    
    def get_production_schedule_features(self) -> torch.Tensor:
        """
        Encode production schedule as temporal features
        Returns: [num_jigs, schedule_feature_dim]
        
        Features per jig:
        - production_line_id: 1 dim (which line, normalized)
        - schedule_position: 1 dim (position in queue, normalized)
        - is_scheduled: 1 dim (binary flag)
        Total: 3 dims per jig
        """
        features = []
        jig_to_position = {}
        
        # Build schedule mapping
        for pl_idx, pl in enumerate(self.production_lines):
            for pos, jig_name in enumerate(pl['schedule']):
                jig_to_position[jig_name] = (pl_idx, pos)
        
        max_position = max((pos for _, pos in jig_to_position.values()), default=1)
        num_lines = len(self.production_lines)
        
        for jig_name in sorted(self.jigs.keys()):
            if jig_name in jig_to_position:
                pl_idx, pos = jig_to_position[jig_name]
                jig_feat = [
                    pl_idx / max(num_lines - 1, 1),  # Normalized line ID
                    pos / max_position,  # Normalized position
                    1.0  # Is scheduled
                ]
            else:
                jig_feat = [0.0, 0.0, 0.0]  # Not scheduled
            
            features.append(jig_feat)
        
        return torch.tensor(features, dtype=torch.float32, device=DEVICE)
    
    def get_full_state_tensor(self) -> torch.Tensor:
        """
        Combine all features into a single flattened state tensor
        Returns: [1, total_feature_dim]
        
        This is the main input to the TRM model
        """
        jig_feats = self.get_jig_features()  # [num_jigs, 8]
        flight_feats = self.get_flight_features()  # [num_flights, 5]
        rack_feats = self.get_rack_features()  # [num_racks, 5]
        schedule_feats = self.get_production_schedule_features()  # [num_jigs, 3]
        
        # Global features
        global_feats = torch.tensor([
            float(len(self.trailers_beluga)),
            float(len(self.trailers_factory)),
            float(len(self.hangars)),
            float(self.num_jigs),
            float(self.num_flights),
            float(self.num_racks),
            float(self.num_production_lines)
        ], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, 7]
        
        # Flatten and concatenate
        jig_flat = jig_feats.flatten().unsqueeze(0)  # [1, num_jigs * 8]
        schedule_flat = schedule_feats.flatten().unsqueeze(0)  # [1, num_jigs * 3]
        flight_flat = flight_feats.flatten().unsqueeze(0)  # [1, num_flights * 5]
        rack_flat = rack_feats.flatten().unsqueeze(0)  # [1, num_racks * 5]
        
        full_state = torch.cat([
            global_feats,
            jig_flat,
            schedule_flat,
            flight_flat,
            rack_flat
        ], dim=1)  # [1, total_dim]
        
        return full_state
    
    def get_constraint_violations(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Compute constraint violations for a proposed solution
        Used as loss function during training
        
        Args:
            solution: [num_jigs, action_space] - proposed jig assignments
        
        Returns:
            violation_score: scalar tensor (0 = all constraints satisfied)
        """
        # Placeholder - will implement constraint checking logic
        # For now, return dummy value
        return torch.tensor(0.0, device=DEVICE)


class BelugaDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Beluga problems"""
    
    def __init__(self, data_dir: str, split: str = "training"):
        """
        Args:
            data_dir: Path to Beluga dataset root (e.g., 'data/beluga/deterministic')
            split: "training" or "validation"
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Find all JSON files in the split directory
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        self.problem_files = sorted(list(split_dir.glob("*.json")))
        
        if len(self.problem_files) == 0:
            raise ValueError(f"No JSON files found in {split_dir}")

        # Get first problem to check dimensions
        with open(self.problem_files[0], 'r') as f:
            first_data = json.load(f)
            self.target_jigs = len(first_data['jigs'])
            self.target_flights = len(first_data['flights'])
            self.target_racks = len(first_data['racks'])
        
        # Filter to only problems with matching dimensions
        filtered_files = []
        for pf in self.problem_files:
            with open(pf, 'r') as f:
                data = json.load(f)
                if (len(data['jigs']) == self.target_jigs and 
                    len(data['flights']) == self.target_flights and
                    len(data['racks']) == self.target_racks):
                    filtered_files.append(pf)
        
        self.problem_files = filtered_files
        
        print(f"✅ Loaded {len(self.problem_files)} problems from {split} split (filtered by dimensions)")
            
    def __len__(self) -> int:
        return len(self.problem_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, BelugaProblem]:
        """
        Returns:
            state_tensor: [1, total_feature_dim] - input for TRM
            problem: BelugaProblem instance for constraint checking
        """
        problem = BelugaProblem.from_json(str(self.problem_files[idx]))
        state_tensor = problem.get_full_state_tensor()
        
        return state_tensor, problem


def create_beluga_dataloader(
    data_dir: str,
    split: str = "training",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 2
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for Beluga dataset
    
    Args:
        data_dir: Path to Beluga dataset root (e.g., 'data/beluga/deterministic')
        split: "training" or "validation"
        batch_size: Batch size (typically 1 for constraint problems)
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader instance
    """
    dataset = BelugaDataset(data_dir, split)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda")
    )


def test_data_loader():
    """Test data loading with sample files"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python beluga_data_loader.py <path_to_problem_directory>")
        print("Example: python beluga_data_loader.py data/beluga/deterministic/training")
        return
    
    data_dir = sys.argv[1]
    
    print(f"🧪 Testing Beluga Data Loader")
    print(f"Dataset directory: {data_dir}")
    print("=" * 60)
    
    # Load single problem
    problem_files = list(Path(data_dir).glob("*.json"))
    if not problem_files:
        print("❌ No problem files found")
        print(f"   Looking in: {data_dir}")
        return
    
    print(f"\n📂 Found {len(problem_files)} problem files")
    print(f"Loading first problem: {problem_files[0].name}")
    
    problem = BelugaProblem.from_json(str(problem_files[0]))
    
    print(f"\n📊 Problem Statistics:")
    print(f"   Problem ID: {problem.problem_id}")
    print(f"   Jigs: {problem.num_jigs}")
    print(f"   Flights: {problem.num_flights}")
    print(f"   Racks: {problem.num_racks}")
    print(f"   Production Lines: {problem.num_production_lines}")
    
    # Test feature extraction
    print(f"\n🔢 Feature Extraction:")
    jig_feats = problem.get_jig_features()
    print(f"   Jig features: {jig_feats.shape}")
    
    flight_feats = problem.get_flight_features()
    print(f"   Flight features: {flight_feats.shape}")
    
    rack_feats = problem.get_rack_features()
    print(f"   Rack features: {rack_feats.shape}")
    
    schedule_feats = problem.get_production_schedule_features()
    print(f"   Schedule features: {schedule_feats.shape}")
    
    full_state = problem.get_full_state_tensor()
    print(f"   Full state tensor: {full_state.shape}")
    print(f"   Total input dimensions: {full_state.shape[1]}")
    
    print(f"\n✅ Data loader test complete!")


if __name__ == "__main__":
    test_data_loader()