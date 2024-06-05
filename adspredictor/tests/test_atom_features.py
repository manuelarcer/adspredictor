import pytest
from ase.io import read
from adspredictor.featurizer.featurizer import AtomFeatures

@pytest.fixture
def atoms():
    return read('OUTCAR_Ag_01', format='vasp-out', index=-1)

def test_create_neighbor_list(atoms):
    af = AtomFeatures(atoms)
    nl = af.create_neighbor_list()
    assert nl is not None

def test_get_surface_atoms(atoms):
    af = AtomFeatures(atoms, cutoff_cn=11)
    surface_atoms = af.get_surface_atoms()
    assert len(surface_atoms) > 0

def test_get_atom_index(atoms):
    af = AtomFeatures(atoms)
    index = af.get_atom_index('C')
    assert index == 55

    index = af.get_atom_index(1)
    assert index == 1

    with pytest.raises(ValueError):
        af.get_atom_index('X')

def test_determine_neigbors(atoms):
    af = AtomFeatures(atoms)
    neighbors = af.determine_neigbors('C')
    assert len(neighbors) > 0

def test_get_neighbors_cutoff(atoms):
    af = AtomFeatures(atoms)
    cutoff_range = (1.5, 2.5)
    condition = af.get_neighbors_cutoff(55, cutoff_range)
    assert condition.any()

