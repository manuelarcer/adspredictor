import pytest, os
import pandas as pd
from ase import Atoms
from ase.io import read
from adspredictor.featurizer.featurizer import FeatureCreator
from asetools.analysis import check_outcar_convergence

@pytest.fixture
def df():
    """
    Fixture to create a DataFrame by reading OUTCAR files.
     
    Returns:
    pd.DataFrame: DataFrame containing information from OUTCAR files
    """
    outcar_files = [f for f in os.listdir('.') if f.startswith('OUTCAR_')]
    df = pd.DataFrame()

    for outcar in outcar_files:
        try:
            convergence, _ = check_outcar_convergence(outcar, verbose=False)
            if convergence:
                atoms = read(outcar, format='vasp-out', index=-1)
                _df = pd.DataFrame.from_dict({
                                        'Config': outcar.split('_')[-1], 
                                        'Convergence': convergence,
                                        'Energy': atoms.get_potential_energy(),
                                        'Atoms': atoms}, orient='index').T
                df = pd.concat([df, pd.DataFrame(_df, index=[0])], ignore_index=True)
        except (FileNotFoundError, IOError, ValueError) as e:
            print(f"Error processing {outcar}: {e}")
            continue
    return df

@pytest.fixture
def ads():
    return 55

@pytest.fixture
def listmetals():
    return ['Ag', 'Au', 'Cu', 'Pd', 'Pt']

def test_bindingsites_indexes(df, ads, listmetals):
    fc = FeatureCreator(df, ads, listmetals)
    indexes = fc.bindingsites_indexes()
    assert not indexes.isnull().values.any()

def test_bindingsites_symbols(df, ads, listmetals):
    fc = FeatureCreator(df, ads, listmetals)
    symbols = fc.bindingsites_symbols()
    assert not symbols.isnull().values.any()

def test_create_feature_binding_site(df, ads, listmetals):
    fc = FeatureCreator(df, ads, listmetals)
    df_with_features = fc.create_feature_binding_site()
    assert all([f'bonding_{metal}' in df_with_features.columns for metal in listmetals])

def test_create_feature_second_neighbors(df, ads, listmetals):
    fc = FeatureCreator(df, ads, listmetals)
    df_with_features = fc.create_feature_second_neighbors()
    assert all([f'neigh_{metal}' in df_with_features.columns for metal in listmetals])

def test_create_features_based_on_cutoff(df, ads, listmetals):
    fc = FeatureCreator(df, ads, listmetals)
    cutoffs = [2, 3.5, 5]
    df_with_features = fc.create_features_based_on_cutoff(cutoffs)
    for i in range(len(cutoffs)):
        for metal in listmetals:
            assert f'R{i}_{metal}' in df_with_features.columns

