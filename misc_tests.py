# misc_tests.py
# E5. enumeration robustness
from utils import iter_smiles
from ape_tokenizer import APETokenizer
from SmilesPE.tokenizer import *
from collections import Counter
import trie_funcs as tf
import numpy as np
import codecs
import tqdm
import time
import math
import matplotlib.pyplot as plt

SLICE = "data/chembl_test.parquet"
APE_DIR = "ape_chembl"  # 48452.2s for training
TRIE_FILE = "trie_chembl.pkl"  # 12.9s for training
TRIE_TTG_FILE = "trie_ttg_chembl.pkl"  # 11.6s for training
SPE_FILE = "spe_chembl.txt"  # 82.5s for training


def mean_len(generator, fn, desc):
    tot = n = 0
    for s in tqdm.tqdm(generator, desc=desc):
        tot += fn(s)
        n += 1
    return tot / n
   

def token_vect(token, ttg):
    """
    looks up the token vector (v_Ri)
    `token` is a standard token parsed by the tokenizer
    `ttg` is a matrix of all token transition probabilities (ij entry is probability token j follows token i)
    output is a vector of conditional probabilities of transition probabilities from `token` → arbitrary token
    """
    return ttg[token, :]


def molecule_repr(token_repr):
    """
    molecule-level representation (z_m)
    `token_repr` is vector (R_i1, R_i2, ..., R_iL)
    this function computes the mean of the constituent token vectors
    """
    l = 0
    z = np.array()
    for ri in token_repr:
        z += token_vect(ri)
        l += 1
    return z / l
    
    
def main():
    ape = APETokenizer.from_pretrained(APE_DIR)
    trie_state = tf.load_state(TRIE_FILE)
    trie_ttg_state = tf.load_state(TRIE_TTG_FILE)
    spe_vocab = codecs.open(SPE_FILE)
    spe = SPE_Tokenizer(spe_vocab)
    all_mols = list(iter_smiles(SLICE))
    q = 10
    random_mols = random.sample(all_mols, q)

    t0 = time.time()
    
    print(trie_ttg_state) 
    trie_ttg_entropy = entropy(list(iter_smiles(SLICE)),
                               lambda s: tf.compress_and_return(s, trie_ttg_state))

if __name__ == "__main__":
    main()