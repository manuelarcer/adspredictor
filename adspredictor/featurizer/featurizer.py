
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

class AtomFeatures:
    def __init__(self, atoms, natural_cutoff_factor=1):
        self.atoms = atoms
        self.natural_cutoff_factor = natural_cutoff_factor
        self.neighbor_list = self.create_neighbor_list()
    
    def get_atom_index(self, interest):
        """
        interest: str or int
            If str, the symbol of the atom of interest.
            If int, the index of the atom of interest.
        """
        if isinstance(interest, str):
            atom_index = next((i for i, atom in enumerate(self.atoms) if atom.symbol == interest), None)
            if atom_index is None:
                raise ValueError(f"No atom with symbol {interest} found.")
        elif isinstance(interest, (int, np.integer)):
            atom_index = interest
        else:
            raise ValueError("Interest must be a string (atom symbol) or an integer (atom index).")
        return atom_index
    
    def create_neighbor_list(self):
        """
        Create a NeighborList object for the atoms object.
            natural_cutoff_factor: float
        """
        cutoffs = natural_cutoffs(self.atoms)
        nl = NeighborList([c * self.natural_cutoff_factor for c in cutoffs], self_interaction=False, bothways=True)
        nl.update(self.atoms)
        return nl
    
    def filter_neighbors(self, atom_index, avoid):
        """
        Filter the neighboring atoms of the atom of interest.
            nl: NeighborList object
            atom_index: int
            atoms: list of ase.Atom objects
            avoid: list of str
                List of atom symbols to avoid. Useful for filtering adsorbate atoms.
        """
        indices, _ = self.neighbor_list.get_neighbors(atom_index)
        filtered_indices = [i for i in indices if self.atoms[i].symbol not in avoid]
        neighboring_atoms = [self.atoms[i].symbol for i in filtered_indices]
        return neighboring_atoms, filtered_indices
    
    def find_neighboring_atoms(self, interest, avoid=[]):
        atom_index = self.get_atom_index(interest)
        neighboring_atoms, filtered_indices = self.filter_neighbors(atom_index, avoid)
        return neighboring_atoms, filtered_indices

    def get_surface_atoms(self, cutoff_cn=12):
        """ 
        Identify surface atoms based on coordination number. 
            natural_cutoff_factor: float - factor to multiply the natural cutoffs by
            cutoff_cn: int - coordination number cutoff for surface atoms
        """
        surface_atoms = []
        for i, atom in enumerate(self.atoms):
            if len(self.neighbor_list.get_neighbors(i)[0]) < cutoff_cn:
                surface_atoms.append(i)
        return surface_atoms

class FeatureCreator:
    def __init__(self, df):
        self.df = df

    def binding_atoms_per_type(self, atoms, ads, metaltype):
        """
        Count the number of specified metal atoms bonded to the adsorbate atom.
            ads: str or int - symbol or index of the adsorbate atom
            metaltype: str - symbol of the metal atom
        """
        af = AtomFeatures(atoms)
        binding_metals, _ = af.find_neighboring_atoms(interest=ads, avoid=[])
        count = len([m for m in binding_metals if m == metaltype])
        return count
    
    def neighbor_count_per_type(self, atoms, ads, metal):
        """
        Count the number of specified metal atoms neighboring the adsorbate atom's neighbors.
            ads: str or int - symbol or index of the adsorbate atom
            metal: str - symbol of the metal atom
        """
        af = AtomFeatures(atoms)
        _, bm_indices = af.find_neighboring_atoms(interest=ads, avoid=[])
        neigh_indexes = []
        metal_symb = []
        for i in bm_indices:
            mneigh, mn_indices = af.find_neighboring_atoms(interest=i, avoid=[ads])
            for j, index in enumerate(mn_indices):
                if index not in neigh_indexes and index not in bm_indices:
                    neigh_indexes.append(index)
                    metal_symb.append(mneigh[j])
        count = len([m for m in metal_symb if m == metal])
        return count

    def add_bonding_features(self, metals=['Ag', 'Au', 'Cu', 'Pd', 'Pt'], ads='N'):
        """
        Add bonding features to the dataframe for specified metals.
        """
        for metal in metals:
            self.df[f'bonding_{metal}'] = self.df.apply(lambda x: self.binding_atoms_per_type(x.Atoms, ads=ads, metaltype=metal), axis=1)

    def add_neigh_features(self, metals=['Ag', 'Au', 'Cu', 'Pd', 'Pt'], ads='N'):
        """
        Add neighboring features to the dataframe for specified metals.
        """
        for metal in metals:
            self.df[f'neigh_{metal}'] = self.df.apply(lambda x: self.neighbor_count_per_type(x.Atoms, ads=ads, metal=metal), axis=1)