import numpy as np
import pandas as pd
from ase.neighborlist import NeighborList, natural_cutoffs
from joblib import Parallel, delayed
from tqdm import tqdm

tqdm.pandas()

class AtomFeatures:
    def __init__(self, atoms, natural_cutoff_factor=1.1, cutoff_cn=11, isparticle=False):
        self.atoms = atoms
        self.natural_cutoff_factor = natural_cutoff_factor
        self.cutoff_cn = cutoff_cn
        self.isparticle = isparticle
        # Cache the neighbor list, distances matrix, and surface atoms.
        self._neighbor_list = None
        self._distances_matrix = None
        self._surface_atoms = None
        self.create_neighbor_list()
        self.get_surface_atoms()
        self.get_all_distances()

    def create_neighbor_list(self):
        cutoffs = natural_cutoffs(self.atoms, mult=1)
        self._neighbor_list = NeighborList([c * self.natural_cutoff_factor for c in cutoffs],
                                           self_interaction=False, bothways=True)
        self._neighbor_list.update(self.atoms)

    def get_neighbor_list(self):
        return self._neighbor_list

    def get_all_distances(self):
        # Cache the distances matrix
        if self._distances_matrix is None:
            self._distances_matrix = self.atoms.get_all_distances(mic=True)
        return self._distances_matrix

    @property
    def distances_matrix(self):
        return self.get_all_distances()

    def get_surface_atoms(self):
        # Cache surface atoms if already computed
        if self._surface_atoms is not None:
            return self._surface_atoms

        surface_atoms = []
        nl = self.get_neighbor_list()
        if not self.isparticle:
            ave_z = np.mean([atom.position[2] for atom in self.atoms])
        for atom in self.atoms:
            if len(nl.get_neighbors(atom.index)[0]) < self.cutoff_cn:
                if self.isparticle:
                    surface_atoms.append(atom.index)
                else:
                    if atom.position[2] > ave_z:
                        surface_atoms.append(atom.index)
        self._surface_atoms = surface_atoms
        return self._surface_atoms

    def get_atom_index(self, interest):
        if isinstance(interest, str):
            for i, atom in enumerate(self.atoms):
                if atom.symbol == interest:
                    return i
            raise ValueError(f"No atom with symbol {interest} found.")
        elif isinstance(interest, (int, np.integer)):
            return interest
        else:
            raise ValueError("Interest must be a string (atom symbol) or an integer (atom index).")

    def determine_neigbors(self, interest, indexonly=True, avoid=[]):
        nl = self.get_neighbor_list()
        if isinstance(interest, list):
            indices = [self.get_atom_index(i) for i in interest]
            avoid_index = [self.get_atom_index(a) for a in avoid]
            all_neighbors = set()
            for idx in indices:
                neigh, _ = nl.get_neighbors(idx)
                filtered = [n for n in neigh if n not in avoid_index]
                all_neighbors.update(filtered)
            all_neighbors = list(all_neighbors)
            if indexonly:
                return all_neighbors
            else:
                neigh_symb = [self.atoms[i].symbol for i in all_neighbors]
                return all_neighbors, neigh_symb
        else:
            index = self.get_atom_index(interest)
            avoid_index = [self.get_atom_index(a) for a in avoid]
            neigh, _ = nl.get_neighbors(index)
            neigh = [i for i in neigh if i not in avoid_index]
            if indexonly:
                return neigh
            else:
                neigh_symb = [self.atoms[i].symbol for i in neigh]
                return neigh, neigh_symb

    def get_neighbors_cutoff(self, ads, cutoff):
        d_matrix = self.distances_matrix
        if isinstance(ads, list):
            ads_indices = [self.get_atom_index(a) if not isinstance(a, int) else a for a in ads]
            conditions = [(d_matrix[ad] >= cutoff[0]) & (d_matrix[ad] < cutoff[1])
                          for ad in ads_indices]
            # Combine conditions: True if any adsorbate meets the condition.
            combined_condition = np.any(conditions, axis=0)
            return combined_condition
        else:
            ad_index = self.get_atom_index(ads) if not isinstance(ads, int) else ads
            condition = (d_matrix[ad_index] >= cutoff[0]) & (d_matrix[ad_index] < cutoff[1])
            return condition

def count_atoms_x_type(listsymbols, metalsymb, avoid=[]):
    # Convert to numpy array for vectorized counting.
    arr = np.array(listsymbols)
    count = np.sum((arr == metalsymb) & (~np.isin(arr, avoid)))
    return int(count)

def find_neigh(atoms, interest, avoid=[], natural_cutoff_factor=1.1):
    af = AtomFeatures(atoms, natural_cutoff_factor=natural_cutoff_factor)
    index_to_avoid = [af.get_atom_index(a) for a in avoid]
    all_neigh = af.determine_neigbors(interest)
    return [i for i in all_neigh if i not in index_to_avoid]

def ads_riadial_distribution(list_atoms, index):
    # Calculate the radial distribution function from a particular atom 
    list_distances = np.array([])
    for atoms in list_atoms:
        d_matrix = atoms.get_all_distances(mic=True)
        distances = d_matrix[index]
        list_distances = np.concatenate([list_distances, distances])
    return list_distances

class FeatureCreator:
    def __init__(self, df, ads, listmetals, avoid=[], isparticle=False, atomscol='Atoms',
                 natural_cutoff_factor=1.1):
        self.df = df.copy()  # DataFrame with 'Atoms' column (ase.Atoms objects)
        self.ads = ads
        self.avoid = avoid
        self.atomscol = atomscol
        self.isparticle = isparticle
        self.listmetals = listmetals
        self.natural_cutoff_factor = natural_cutoff_factor

        # Precompute AtomFeatures for each Atoms object once.
        print("Precomputing AtomFeatures...")
        self.df['atom_features'] = self.df[self.atomscol].progress_apply(
            lambda atoms: AtomFeatures(atoms, natural_cutoff_factor=self.natural_cutoff_factor,
                                 isparticle=self.isparticle)
        )
        #self.df['atom_features'] = self.df[self.atomscol].apply(
        #    lambda atoms: AtomFeatures(atoms, natural_cutoff_factor=self.natural_cutoff_factor,
        #                                 isparticle=self.isparticle)
        #)
        self.bindingsites_idx = self.bindingsites_indexes()
        self.bindingsites_symb = self.bindingsites_symbols()

    def bindingsites_indexes(self):
        # Use precomputed atom_features to get binding site indices.
        return self.df['atom_features'].apply(
            lambda af: af.determine_neigbors(self.ads, indexonly=False, avoid=self.avoid)[0]
        )

    def bindingsites_symbols(self):
        return self.df['atom_features'].apply(
            lambda af: af.determine_neigbors(self.ads, indexonly=False, avoid=self.avoid)[1]
        )

    def create_feature_binding_site(self):
        print("Creating features based on binding sites...")
        for metal in self.listmetals:
            self.df[f'bonding_{metal}'] = self.bindingsites_symb.progress_apply(
                lambda symbols: count_atoms_x_type(symbols, metal, avoid=self.avoid)
            )
        return self.df

    def _compute_second_neighbors(self, row):
        # Helper function for parallel processing.
        af = row['atom_features']
        binding_indices = af.determine_neigbors(self.ads, indexonly=True, avoid=self.avoid)
        second_neigh = []
        # Use the cached neighbor list to get neighbors of each binding site.
        for idx in binding_indices:
            neigh = af.determine_neigbors(idx, indexonly=True, avoid=self.avoid)
            second_neigh.extend(neigh)
        # Remove duplicates and exclude the binding indices.
        second_neigh = list(set(second_neigh) - set(binding_indices))
        return second_neigh

    def second_neighbors(self):
        # Compute second neighbors in parallel.
        # TODO this part is very slow
        second_neigh_list = Parallel(n_jobs=-1)(
            delayed(self._compute_second_neighbors)(row)
                for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing second neighbors")
        )
        self.df['second_neighbors'] = second_neigh_list
        return self.df['second_neighbors']

    def create_feature_second_neighbors(self, distinguishsurface=False):
        print("Creating features based on second neighbors...")
        self.second_neighbors()  # Ensure second_neighbors column is computed.
        if distinguishsurface:
            def get_symbols(row):
                af = row['atom_features']
                atoms = row[self.atomscol]
                surf_atoms = af.get_surface_atoms()
                indices = row['second_neighbors']
                symbols_surf = [atoms[i].symbol for i in indices if i in surf_atoms]
                symbols_bulk = [atoms[i].symbol for i in indices if i not in surf_atoms]
                return symbols_surf, symbols_bulk

            symbols = self.df.apply(get_symbols, axis=1)
            symbols_surf = symbols.apply(lambda x: x[0])
            symbols_bulk = symbols.apply(lambda x: x[1])
            for metal in self.listmetals:
                self.df[f'neigh_S_{metal}'] = symbols_surf.apply(
                    lambda x: count_atoms_x_type(x, metal, avoid=self.avoid)
                )
                self.df[f'neigh_B_{metal}'] = symbols_bulk.apply(
                    lambda x: count_atoms_x_type(x, metal, avoid=self.avoid)
                )
        else:
            symbols = self.df.apply(
                lambda row: [row[self.atomscol][i].symbol for i in row['second_neighbors']],
                axis=1
            )
            for metal in self.listmetals:
                self.df[f'neigh_{metal}'] = symbols.apply(
                    lambda x: count_atoms_x_type(x, metal, avoid=self.avoid)
                )
        return self.df

    def create_features_based_on_cutoff(self, cutoffs=[], surfdistinc=False):
        print("Creating features based on cutoffs...")
        # Build limits from the list of cutoff values.
        limits = []
        for i, cutoff in enumerate(cutoffs):
            if i == 0:
                limits.append((0, cutoff))
            else:
                limits.append((cutoffs[i-1], cutoff))
        # Reuse the precomputed AtomFeatures.
        atom_features_series = self.df['atom_features']
        for i, limpair in enumerate(limits):
            neighbors_condition = atom_features_series.apply(
                lambda af: af.get_neighbors_cutoff(self.ads, limpair)
            )
            symbolsserie = self.df.apply(
                lambda row: row[self.atomscol][neighbors_condition[row.name]].get_chemical_symbols(),
                axis=1
            )
            if surfdistinc:
                surf_atoms_series = atom_features_series.apply(lambda af: af.get_surface_atoms())

                def filter_symbols(row):
                    neighbor_condition = neighbors_condition[row.name]
                    neighbor_indices = np.where(neighbor_condition)[0]
                    neighbor_symbols = row[self.atomscol][neighbor_condition].get_chemical_symbols()
                    surf_atoms = surf_atoms_series[row.name]
                    symbols_surf = [sym for idx, sym in zip(neighbor_indices, neighbor_symbols) if idx in surf_atoms]
                    symbols_bulk = [sym for idx, sym in zip(neighbor_indices, neighbor_symbols) if idx not in surf_atoms]
                    return symbols_surf, symbols_bulk

                filtered = self.df.apply(filter_symbols, axis=1)
                symbols_surf = filtered.apply(lambda x: x[0])
                symbols_bulk = filtered.apply(lambda x: x[1])
                for metal in self.listmetals:
                    self.df[f'R{i}_S_{metal}'] = symbols_surf.apply(
                        lambda x: count_atoms_x_type(x, metal, avoid=self.avoid)
                    )
                    self.df[f'R{i}_B_{metal}'] = symbols_bulk.apply(
                        lambda x: count_atoms_x_type(x, metal, avoid=self.avoid)
                    )
            else:
                for metal in self.listmetals:
                    self.df[f'R{i}_{metal}'] = symbolsserie.apply(
                        lambda x: count_atoms_x_type(x, metal, avoid=self.avoid)
                    )
        return self.df
