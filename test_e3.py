# E5. enumeration robustness
from enum import EnumDict
from ape_tokenizer import APETokenizer
from collections import Counter
from SmilesPE.tokenizer import *
from utils import iter_smiles
import trie_funcs as tf
import numpy as np
import matplotlib.pyplot as plt
import json, pickle, codecs, tqdm, time, math, random

SLICE = "data/chembl_test.parquet"
APE_DIR = "ape_chembl"  # 48452.2s for training
TRIE_FILE = "trie_chembl.pkl"  # 12.9s for training
TRIE_TTG_FILE = "trie_ttg_chembl.pkl"  # 11.6s for training
SPE_FILE = "spe_chembl.txt"  # 82.5s for training


def token_vect(token, ttg, default):
    """
    Look up the token vector (v_Ri).
    `token` is a standard token parsed by the tokenizer.
    `ttg` is a matrix of all token transition probabilities (ij entry is probability token j follows token i).
    Output is a vector of conditional probabilities of transition probabilities from `token` → arbitrary token.
    """
    return ttg.get(token, default)


def token_list(vector, keys):
    """
    Given a token vector, which is a dictionary of key to probabilities,
    and a list of keys, organize the token vector into a list where entries are sorted according to `keys`
    """
    return np.array([vector.get(key, 0) for i, key in enumerate(keys)])


def molecule_repr(token_repr, ttg, keys):
    """
    Molecule-level representation (z_m)
    `token_repr` is vector (R_i1, R_i2, ..., R_iL)
    This function computes the mean of the constituent token vectors.
    """
    l = 0
    z = np.zeros(keys.shape)
    uniform = np.ones(keys.shape[0]) / keys.shape[0]
    for ri in token_repr:
        z += token_list(token_vect(ri, ttg, uniform), keys)
        l += 1
    return z / l

    
cosine_similarity = lambda z1, z2: np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))
dedup = lambda similarity_matrix: similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)]

    
def main():
    q = 10
    all_mols = list(iter_smiles(SLICE))
    random_mols = random.sample(all_mols, q)

    t0 = time.time()
    
    ttg = dict()
    keys = []
    
    with open("ttg/matrix.pkl", "rb") as f:
        ttg = pickle.load(f)
    with open("ttg/keys.json", "r") as f:
        keys = np.array(json.load(f))
   
    uniform = np.ones(keys.shape[0]) / keys.shape[0]
    token_matrix = np.array([token_list(token_vect(key, ttg, uniform), keys) for key in keys])
    
    dedup_matrix = dedup(token_matrix)
    mu = np.mean(dedup_matrix)
    sigma = np.std(dedup_matrix)
    
    t1 = time.time()
    print(f"E3: Cosine similarity statistics on token vectors")
    print(f"Mean: {mu:.4f}, Standard Deviation: {sigma:.4f}")
    print(f"Time to generate statistics: {t1 - t0:.4f} seconds")


if __name__ == "__main__":
    main()
