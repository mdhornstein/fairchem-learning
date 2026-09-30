import numpy as np 
from ase import Atoms 

def test_water_molecule_representation():
    """Verify how ASE represents an isolated molecule in 3D space.""" 
    # Water: 1 Oxygen at origin, 2 Hydrogens angled 
    positions = [
        [0.0, 0.0, 0.0], 
        [0.0, 0.757, 0.586], 
        [0.0, -0.757, 0.586]
    ]
    symbols = ["O", "H", "H"]

    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)

    assert len(atoms) == 3
    assert atoms.get_chemical_formula() == "H2O"

    np.testing.assert_array_equal(
        atoms.get_atomic_numbers(), 
        [8, 1, 1]
    )

    np.testing.assert_array_equal(
        atoms.pbc, [False, False, False]
    )

    d_oh = atoms.get_distance(0, 1)
    assert np.isclose(d_oh, 0.957, atol=1e-3)