import numpy as np
import pandas as pd
from ase.neighborlist import NeighborList, natural_cutoffs

class AtomFeatures:
    def __init__(self, atoms, natural_cutoff_factor=1.1, cutoff_cn=11, isparticle=False):
        self.atoms = atoms
        self.natural_cutoff_factor = natural_cutoff_factor
        self.cutoff_cn = cutoff_cn
        self.isparticle = isparticle
        self.neighbor_list = self.create_neighbor_list()
        self.surface_atoms = self.get_surface_atoms()
        self.distances_matrix = self.atoms.get_all_distances(mic=True)

    def create_neighbor_list(self):
        """
        Create a NeighborList object for the atoms object.
            natural_cutoff_factor: float
        """

        cutoffs = natural_cutoffs(self.atoms, mult=1)
        nl = NeighborList([c * self.natural_cutoff_factor for c in cutoffs], self_interaction=False, bothways=True)
        nl.update(self.atoms)

        return nl
    
    def get_surface_atoms(self):                # TODO: it does not seem to work entirely
        """ 
        Identify surface atoms based on coordination number. 
            natural_cutoff_factor: float - factor to multiply the natural cutoffs by
            cutoff_cn: int - coordination number cutoff for surface atoms
        """
        surface_atoms = []

        if not self.isparticle:
            ave_z = np.mean([atom.position[2] for atom in self.atoms])

        for atom in self.atoms:
            if len(self.neighbor_list.get_neighbors(atom.index)[0]) < self.cutoff_cn:
                if self.isparticle:
                    surface_atoms.append(atom.index)
                else:
                    if atom.position[2] > ave_z:
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
    
    def determine_neigbors(self, interest, indexonly=True, avoid=[]):
        """
        interest: str or int
            If str, the symbol of the atom of interest.
            If int, the index of the atom of interest.
        avoid: list of str
            List of atom symbols to avoid. Useful for filtering adsorbate atoms.
        """
        index = self.get_atom_index(interest)
        avoid_index = [self.get_atom_index(a) for a in avoid]
        neigh_i, _ = self.neighbor_list.get_neighbors(index)
        neigh_i = [i for i in neigh_i if i not in avoid_index]
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
    def __init__(self, df, ads, listmetals, avoid=[], isparticle=False, atomscol='Atoms'):
        self.df = df       # DataFrame with 'Atoms' column (ase.Atoms objects)
        self.ads = ads
        self.avoid = avoid
        self.atomscol = atomscol
        self.isparticle = isparticle
        self.listmetals = listmetals
        self.bindingsites_idx = self.bindingsites_indexes()
        self.bindingsites_symb = self.bindingsites_symbols()

    def bindingsites_indexes(self):
        serie = self.df[self.atomscol].apply(lambda x: AtomFeatures(x).determine_neigbors(self.ads, indexonly=False, avoid=self.avoid))
        return serie.apply(lambda x: x[0])

    def bindingsites_symbols(self):
        serie = self.df[self.atomscol].apply(lambda x: AtomFeatures(x).determine_neigbors(self.ads, indexonly=False, avoid=self.avoid))
        return serie.apply(lambda x: x[1])

    def create_feature_binding_site(self):
        for metal in self.listmetals:
            self.df[f'bonding_{metal}'] = self.bindingsites_symb.apply(lambda x: count_atoms_x_type(x, metal))
        return self.df

    def second_neighbors(self):
        # Binding sites to Dataframe
        dummy_df = pd.DataFrame(self.bindingsites_idx)
        # Find the second neighbors of the binding sites
        second_neigh_serie = dummy_df.apply(
                lambda x: [find_neigh(self.df[self.atomscol].loc[x.name], i, avoid=self.avoid, natural_cutoff_factor=1.1) for i in x.iloc[0]], axis=1)
        # Combine the lists in each row inside second_neighbors and then remove duplicates
        ## Needed for x-fold type of adsorption
        second_neigh_serie = second_neigh_serie.apply(lambda x: [item for sublist in x for item in sublist])
        second_neigh_serie = second_neigh_serie.apply(lambda x: list(set(x)))
        second_neigh_serie = pd.DataFrame(second_neigh_serie)
        # Remove the binding sites from the second neighbors
        second_neigh_serie = second_neigh_serie.apply(
                lambda x: [i for i in x[0] if i not in self.bindingsites_idx.loc[x.name]], axis=1
                )
        return second_neigh_serie
    
    def create_feature_second_neighbors(self, distinguishsurface=False):
        second_neigh_serie = self.second_neighbors()    # Shape: (n_samples, 1)
        second_neigh_serie = pd.DataFrame(second_neigh_serie)
        # Count the number of each atom type in the list
        if distinguishsurface:

            # Define a function to get the symbols of the neighbors
            def get_symbols(row, df, atomscol, isparticle, surf=True):
                # Extract the row's Atoms object and create AtomFeatures
                row_atoms = df[atomscol].loc[row.name]
                atom_features = AtomFeatures(row_atoms, isparticle=isparticle)
                surf_atoms = atom_features.get_surface_atoms()

                # Extract indices from the first element of the row
                indices = row.iloc[0]

                # Depending on surf parameter, filter indices
                if surf:
                    return [row_atoms[i].symbol for i in indices if i in surf_atoms]
                else:
                    return [row_atoms[i].symbol for i in indices if i not in surf_atoms]

            symbols_surf = second_neigh_serie.apply(lambda x: get_symbols(x, self.df, self.atomscol, self.isparticle, surf=True), axis=1)
            symbols_bulk = second_neigh_serie.apply(lambda x: get_symbols(x, self.df, self.atomscol, self.isparticle, surf=False), axis=1)

            #symbols_surf = second_neigh_serie.apply(lambda x: [
            #            self.df[self.atomscol].loc[x.name][i].symbol for i in x.iloc[0] if i in AtomFeatures(self.df[self.atomscol].loc[x.name],isparticle=self.isparticle).get_surface_atoms()
            #            ], axis=1)
            #symbols_bulk = second_neigh_serie.apply(lambda x: [
            #            self.df[self.atomscol].loc[x.name][i].symbol for i in x.iloc[0] if i not in AtomFeatures(self.df[self.atomscol].loc[x.name],isparticle=self.isparticle).get_surface_atoms()
            #            ], axis=1)
            
            for metal in self.listmetals:
                self.df[f'neigh_S_{metal}'] = symbols_surf.apply(lambda x: count_atoms_x_type(x, metal))
                self.df[f'neigh_B_{metal}'] = symbols_bulk.apply(lambda x: count_atoms_x_type(x, metal))
        else:
            symbols = second_neigh_serie.apply(lambda x: [
                        self.df[self.atomscol].loc[x.name][i].symbol for i in x.iloc[0]], axis=1)
            for metal in self.listmetals:
                self.df[f'neigh_{metal}'] = symbols.apply(lambda x: count_atoms_x_type(x, metal))
        return self.df

    def create_features_based_on_cutoff(self, cutoffs = [], surfdistinc=False):
        # cutoffs: list of float (List of cutoffs to use for the neighbors)
        # surfdistinc: bool (If True, distinguish between surface and bulk atoms), this doubles the number of features
        for i, cutoff in enumerate(cutoffs):
            if i == 0:
                limits = [(0, cutoff)]
            else:
                limits.append((cutoffs[i-1], cutoff))
        
        for i, limpair in enumerate(limits):
                # Instantiate AtomFeatures once per row
                atom_features_series = self.df[self.atomscol].apply( lambda x: AtomFeatures(x, isparticle=True) )

                # Retrieve neighbor boolean arrays within the cutoff range
                neighbors_condition = atom_features_series.apply(lambda af: af.get_neighbors_cutoff(self.ads, limpair))

                # Retrieve chemical symbols of the neighbors within the cutoff
                symbolsserie = self.df.apply(
                        lambda row: row[self.atomscol][neighbors_condition[row.name]].get_chemical_symbols(), axis=1 )
                
                if surfdistinc:
                    # Retrieve surface atom indices for each row
                    surf_atoms_series = atom_features_series.apply( lambda af: af.get_surface_atoms() )
                    
                    # Define a function to filter symbols based on surface and bulk
                    def filter_symbols(row):
                        neighbor_condition = neighbors_condition[row.name]
                        neighbor_indices = np.where(neighbor_condition)[0]
                        neighbor_symbols = row[self.atomscol][neighbor_condition].get_chemical_symbols()

                        surf_atoms = surf_atoms_series[row.name]
                        symbols_surf = [symbol for idx, symbol in zip(neighbor_indices, neighbor_symbols) if idx in surf_atoms]
                        symbols_bulk = [symbol for idx, symbol in zip(neighbor_indices, neighbor_symbols) if idx not in surf_atoms]
                        return symbols_surf, symbols_bulk

                    # Apply the filtering function row-wise
                    filtered_symbols = self.df.apply(filter_symbols, axis=1)

                    # Extract surface and bulk symbols
                    symbols_surf = filtered_symbols.apply(lambda x: x[0])
                    symbols_bulk = filtered_symbols.apply(lambda x: x[1])

                    for metal in self.listmetals:
                        # Count metal atoms in surface neighbors
                        self.df[f'R{i}_S_{metal}'] = symbols_surf.apply(
                            lambda x: count_atoms_x_type(x, metal, avoid=self.avoid))
                        # Count metal atoms in bulk neighbors
                        self.df[f'R{i}_B_{metal}'] = symbols_bulk.apply(
                            lambda x: count_atoms_x_type(x, metal, avoid=self.avoid))

                else:
                    for metal in self.listmetals:
                        self.df[f'R{i}_{metal}'] = symbolsserie.apply(lambda x: count_atoms_x_type(x, metal, avoid=self.avoid))    
                    
        return self.df
    
def ads_riadial_distribution(list_atoms, index):
    # Calculate the radial distribution function from a particular atom 
    list_distances = np.array([])
    for atoms in list_atoms:
        d_matrix = atoms.get_all_distances(mic=True)
        distances = d_matrix[index]
        list_distances = np.concatenate([list_distances, distances])
    return list_distances

def count_atoms_x_type(listsymbols, metalsymb, avoid=[]):
    # listsymbols: list of str (List of atom symbols)
    # metalsymb: str (Symbol of metal atom for feature)
    # avoid: list of str (List of atom symbols to avoid)

    # How many items in neigh_symb match metalsymb
    count = sum([1 for symb in listsymbols if symb == metalsymb and symb not in avoid])
    return count

def find_neigh(atoms, interest, avoid=[], natural_cutoff_factor=1.1):
    # atoms: Atoms object
    # interest: str or int (Symbol or index of the atom of interest)
    af = AtomFeatures(atoms, natural_cutoff_factor=natural_cutoff_factor)
    index_to_avoid = [af.get_atom_index(a) for a in avoid]
    all_neigh = af.determine_neigbors(interest)
    # remove avoid indexes
    return [i for i in all_neigh if i not in index_to_avoid]