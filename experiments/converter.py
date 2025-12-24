from rdkit import Chem
import selfies as sf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def inchi_to_selfies(inchi):
    mol = Chem.inchi.MolFromInchi(inchi, sanitize=True, treatWarningAsError=False)
    smiles = Chem.MolToSmiles(mol)
    selfies = sf.encoder(smiles)
    return selfies