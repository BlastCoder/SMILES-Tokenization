# bench_lengths.py
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
    

def var_len(generator, fn, desc, mean):
    tot = n = 0
    for s in tqdm.tqdm(generator, desc=desc):
        tot += (fn(s) - mean) ** 2
        n += 1
    return tot / n
    
    
def entropy(mols, fn):
    token_freq = Counter()
    tot_tokens = 0
    for mol in mols:
        tokens = fn(mol)
        token_freq.update(tokens)
        tot_tokens += len(tokens)
    
    entropy = 0
    for count in token_freq.values():
        p = count / tot_tokens
        entropy -= p * math.log2(p)
    return entropy
    
    
def main():
    ape = APETokenizer.from_pretrained(APE_DIR)
    trie_state = tf.load_state(TRIE_FILE)
    trie_ttg_state = tf.load_state(TRIE_TTG_FILE)
    spe_vocab = codecs.open(SPE_FILE)
    spe = SPE_Tokenizer(spe_vocab)

    t0 = time.time()
    mol = "ClC=CBr"
    tokenized = tf.compress_and_return(mol, trie_state)
     
    print("Molecule string: ", mol) 
    print("Semantic tokens: ", tf.tokenize(mol))
    print("Trie tokens:", tokenized)
    
    print(len(trie_state.idx_to_token.items()))
    print(len(trie_ttg_state.idx_to_token.items()))
    print(len(ape.vocabulary))

if __name__ == "__main__":
    main()
