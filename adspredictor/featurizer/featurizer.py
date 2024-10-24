import numpy as np
import pandas as pd
from ase.neighborlist import NeighborList, natural_cutoffs

class AtomFeatures:
    def __init__(self, atoms, natural_cutoff_factor=1, cutoff_cn=11):
        self.atoms = atoms
        self.natural_cutoff_factor = natural_cutoff_factor
        self.cutoff_cn = cutoff_cn
        self.neighbor_list = self.create_neighbor_list()
        self.surface_atoms = self.get_surface_atoms()
        self.distances_matrix = self.atoms.get_all_distances(mic=True)

    def create_neighbor_list(self):
        """
        Create a NeighborList object for the atoms object.
            natural_cutoff_factor: float
        """
        cutoffs = natural_cutoffs(self.atoms)
        nl = NeighborList([c * self.natural_cutoff_factor for c in cutoffs], self_interaction=False, bothways=True)
        nl.update(self.atoms)
        return nl
    
    def get_surface_atoms(self):
        """ 
        Identify surface atoms based on coordination number. 
            natural_cutoff_factor: float - factor to multiply the natural cutoffs by
            cutoff_cn: int - coordination number cutoff for surface atoms
        """
        surface_atoms = []
        for atom in self.atoms:
            if len(self.neighbor_list.get_neighbors(atom.index)[0]) < self.cutoff_cn:
                surface_atoms.append(atom.index)
        return surface_atoms
    
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
    
    def determine_neigbors(self, interest, indexonly=True):
        """
        interest: str or int
            If str, the symbol of the atom of interest.
            If int, the index of the atom of interest.
        avoid: list of str
            List of atom symbols to avoid. Useful for filtering adsorbate atoms.
        """
        index = self.get_atom_index(interest)
        neigh_i, _ = self.neighbor_list.get_neighbors(index)
        if indexonly:
            return neigh_i
        else:
            neigh_symb = self.atoms[neigh_i].get_chemical_symbols()
            return neigh_i, neigh_symb
    
    def get_neighbors_cutoff(self, ads, cutoff):
        # returns a boolean array with the atoms that are within the cutoff range
        condition = (self.distances_matrix[ads] >= cutoff[0]) & (self.distances_matrix[ads] < cutoff[1])
        return condition
    
class FeatureCreator:
    def __init__(self, df, ads, listmetals):
        self.df = df       # DataFrame with 'Atoms' column
        self.ads = ads
        self.listmetals = listmetals
        self.bindingsites_idx = self.bindingsites_indexes()
        self.bindingsites_symb = self.bindingsites_symbols()

#    def neigbors_serie(self, interest):
#        # interest: str (Symbol) or int (Index) of atom of interest
#        return self.df['Atoms'].apply(lambda x: AtomFeatures(x).determine_neigbors(interest, indexonly=False))

    def bindingsites_indexes(self):
        serie = self.df['Atoms'].apply(lambda x: AtomFeatures(x).determine_neigbors(self.ads, indexonly=False))
        return serie.apply(lambda x: x[0])

    def bindingsites_symbols(self):
        serie = self.df['Atoms'].apply(lambda x: AtomFeatures(x).determine_neigbors(self.ads, indexonly=False))
        return serie.apply(lambda x: x[1])

    def create_feature_binding_site(self):
        for metal in self.listmetals:
            self.df[f'bonding_{metal}'] = self.bindingsites_symb.apply(lambda x: count_atoms_x_type(x, metal))
        return self.df

    def second_neighbors(self):
        dummy_df = pd.DataFrame(self.bindingsites_idx)
        second_neigh_serie = dummy_df.apply(
                lambda x: [find_neigh(self.df.Atoms.loc[x.name], i, avoid=[self.ads]) for i in x.iloc[0]], axis=1)
        # Combine the lists in each row inside second_neighbors and then remove duplicates
        ## Needed for x-fold type of adsorption
        second_neigh_serie = second_neigh_serie.apply(lambda x: [item for sublist in x for item in sublist])
        second_neigh_serie = second_neigh_serie.apply(lambda x: list(set(x)))
        second_neigh_serie = pd.DataFrame(second_neigh_serie)
        second_neigh_serie = second_neigh_serie.apply(
                lambda x: [i for i in x[0] if i not in self.bindingsites_idx.loc[x.name]], axis=1
                )
        return second_neigh_serie
    
    def create_feature_second_neighbors(self, distinguishsurface=False):
        second_neigh_serie = self.second_neighbors()
        second_neigh_serie = pd.DataFrame(second_neigh_serie)
        # Count the number of each atom type in the list
        if distinguishsurface:
            symbols_surf = second_neigh_serie.apply(lambda x: [
                        self.df.Atoms.loc[x.name][i].symbol for i in x.iloc[0] if i in AtomFeatures(self.df.Atoms.loc[x.name]).get_surface_atoms()
                        ], axis=1)
            symbols_bulk = second_neigh_serie.apply(lambda x: [
                        self.df.Atoms.loc[x.name][i].symbol for i in x.iloc[0] if i not in AtomFeatures(self.df.Atoms.loc[x.name]).get_surface_atoms()
                        ], axis=1)
            for metal in self.listmetals:
                self.df[f'neigh_surf_{metal}'] = symbols_surf.apply(lambda x: count_atoms_x_type(x, metal))
                self.df[f'neigh_bulk_{metal}'] = symbols_bulk.apply(lambda x: count_atoms_x_type(x, metal))
        else:
            symbols = second_neigh_serie.apply(lambda x: [
                        self.df.Atoms.loc[x.name][i].symbol for i in x.iloc[0]
                        ], axis=1)
            for metal in self.listmetals:
                self.df[f'neigh_{metal}'] = symbols.apply(lambda x: count_atoms_x_type(x, metal))
        return self.df

    def create_features_based_on_cutoff(self, cutoffs = []):
        # cutoffs: list of float (List of cutoffs to use for the neighbors)
        for i, cutoff in enumerate(cutoffs):
            if i == 0:
                limits = [(0, cutoff)]
            else:
                limits.append((cutoffs[i-1], cutoff))
        
        for i, limpair in enumerate(limits):
                symbolsserie = self.df.Atoms.apply(
                        lambda x: x[AtomFeatures(x).get_neighbors_cutoff(self.ads, limpair)].get_chemical_symbols())
                for metal in self.listmetals:    
                    self.df[f'R{i}_{metal}'] = symbolsserie.apply(lambda x: count_atoms_x_type(x, metal))
        return self.df

def count_atoms_x_type(listsymbols, metalsymb, avoid=[]):
    # listsymbols: list of str (List of atom symbols)
    # metalsymb: str (Symbol of metal atom for feature)
    # avoid: list of str (List of atom symbols to avoid)

    # How many items in neigh_symb match metalsymb
    count = sum([1 for symb in listsymbols if symb == metalsymb and symb not in avoid])
    return count

def find_neigh(atoms, interest, avoid=[]):
    # atoms: Atoms object
    # interest: str or int (Symbol or index of the atom of interest)
    af = AtomFeatures(atoms)
    index_to_avoid = [af.get_atom_index(a) for a in avoid]
    all_neigh = af.determine_neigbors(interest)
    # remove avoid indexes
    return [i for i in all_neigh if i not in index_to_avoid]