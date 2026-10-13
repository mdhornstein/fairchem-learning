from ase.constraints import fix_cartesian
import torch 
from ase import Atoms 
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch 

def test_ase_to_atomic_data_conversion():
    """Verify conversion of an ASE Atoms object into 
    FairChem AtomicData PyTorch tensors.""" 

    # Water molecule 
    positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586]
    ]
    symbols = ["O", "H", "H"]
    atoms = Atoms(
        symbols=symbols, 
        positions=positions, 
        pbc=False)

    
    # Convert to FAIRChem AtomicData 
    data = AtomicData.from_ase(atoms)

    assert isinstance(data.pos, torch.Tensor)
    assert data.pos.shape == (3, 3)

    # Check atomic numbers 
    assert torch.equal(data.atomic_numbers, torch.tensor([8, 1, 1], dtype=torch.long))

    # Check number of atoms tensor 
    assert data.natoms.item() == 3 


def test_batching_multiple_molecules():
    """Verify how FAIRChem batches variable-sized graphs
    into a disjoint union""" 

    # Molecule 0: Water (3 atoms)
    h2o = Atoms(
        symbols=["O", "H", "H"], 
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.58], 
            [0.0, -0.75, 0.58] 
        ],
        pbc=False
    )

    # Molecule 1: Hydrogen gas (2 atoms)
    h2 = Atoms(
        symbols=["H", "H"], 
        positions=[
            [0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.74]
        ],
        pbc=False 
    )

    data_h2o = AtomicData.from_ase(h2o)
    data_h2 = AtomicData.from_ase(h2)

    # Combine into a single batch 
    batch = atomicdata_list_to_batch([
        data_h2o, data_h2
    ])

    # Total atoms = 3 + 2 = 5 
    assert batch.pos.shape == (5, 3)

    # Batch indicator tensor assigns each atom to its molecule index 
    expected_batch_indices = torch.tensor(
        [0, 0, 0, 1, 1], dtype=torch.long)
    
    assert torch.equal(
        batch.batch,
        expected_batch_indices
    )

    # Number of atoms per molecule 
    expected_natoms = torch.tensor([3, 2], dtype=torch.long)
    assert torch.equal(
        batch.natoms, expected_natoms
    )
