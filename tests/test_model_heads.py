import torch 
import torch.nn as nn 

class ToyBackboneAndHeads(nn.Module):
    """A minimal pedagogical model showing FAIRChem's 
    Backbone + Heads patern.""" 

    def __init__(self, hidden_dim=16):
        super().__init__()
        # Backbone: Embeds atomic numbers (Z=1 to 100)
        # into hidden features 
        self.embedding = nn.Embedding(100, hidden_dim)

        # Energy Head: Predicts a scalar energy per atom
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Direct Force Head: Predicts a 3D vector (x, y, z) per atom
        self.direct_force_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, atomic_numbers, pos): 
        # 1. Backbone produces per-atom embeddings
        h = self.embedding(atomic_numbers)

        # Add coordinate interaction so energy depends on position 
        # (in real models, this is equivariant message passing)
        dist_from_origin = torch.norm(pos, dim=-1, keepdim=True) 
        h = h * dist_from_origin 

        # 2. Energy Head: Sum atomic energies to get total system Energy E 
        per_atom_energy = self.energy_head(h)
        total_energy = per_atom_energy.sum() 

        # 3. Direct Force Head: Predicts 3D forces directly 
        direct_forces = self.direct_force_head(h)

        return total_energy, direct_forces 


def test_conserving_vs_direct_forces():
    """
    Verify how Conserving forces (Autograd) differ from 
    Direct head forces. 
    """
    model = ToyBackboneAndHeads()

    # 3 atoms: O, H, H
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    # Posiitons must require grad so Pytorch can differentiate 
    # w.r.t. coordinates 
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.75, 0.58],
        [0.0, -0.75, 0.58]
    ], requires_grad=True)

    # Forward pass 
    total_energy, direct_forces = model(atomic_numbers, pos)

    # Conserving Forces: Negative gradient of Energy w.r.t. positions (-dE/dr)
    conserving_forces = -torch.autograd.grad(
        outputs=total_energy, 
        inputs=pos, 
        create_graph=True,
    )[0]

    # Both heads output forces with shape (3 atoms, 3 dimensions: x, y, z)
    assert conserving_forces.shape == (3, 3)
    assert direct_forces.shape == (3, 3)

    # Crucial assertion: Total energy is a scalar 
    assert total_energy.ndim == 0 
