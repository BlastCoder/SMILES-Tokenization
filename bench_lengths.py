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
    
    ape_avg = mean_len(iter_smiles(SLICE),
                       lambda s: len(ape.encode(s)),
                       "APE  ")
    ape_var = var_len(iter_smiles(SLICE),
                       lambda s: len(ape.encode(s)),
                       "APE variance  ",
                       ape_avg)
    ape_entropy = entropy(list(iter_smiles(SLICE)),
                          lambda s: ape.encode(s))
    trie_avg = mean_len(iter_smiles(SLICE),
                        lambda s: tf.compress_and_len(s, trie_state),
                        "Trie ")
    trie_var = var_len(iter_smiles(SLICE),
                        lambda s: tf.compress_and_len(s, trie_state),
                        "Trie variance  ",
                        trie_avg)
    trie_entropy = entropy(list(iter_smiles(SLICE)),
                           lambda s: tf.compress_and_return(s, trie_state))
    trie_ttg_avg = mean_len(iter_smiles(SLICE),
                            lambda s: tf.compress_and_len(s, trie_ttg_state),
                            "Trie-TTG ")
    trie_ttg_var = var_len(iter_smiles(SLICE),
                            lambda s: tf.compress_and_len(s, trie_ttg_state),
                            "Trie-TTG variance  ",
                            trie_ttg_avg)
    trie_ttg_entropy = entropy(list(iter_smiles(SLICE)),
                               lambda s: tf.compress_and_return(s, trie_ttg_state))
    spe_avg = mean_len(iter_smiles(SLICE),
                       lambda s: len(spe.tokenize(s).split(" ")),
                       "SPE  ")
    spe_var = var_len(iter_smiles(SLICE),
                      lambda s: len(spe.tokenize(s).split(" ")),
                      "SPE variance  ",
                      spe_avg)
    spe_entropy = entropy(list(iter_smiles(SLICE)),
                          lambda s: spe.tokenize(s).split(" "))
    
    print(f"\nAPE mean tokens : {ape_avg:5.2f}")
    print(f"APE variance in tokens/mol : {ape_var:5.2f}")
    print(f"APE entropy in bits : {ape_entropy:5.2f}")
    print(f"Trie mean tokens : {trie_avg:5.2f}")
    print(f"Trie variance in tokens/mol : {trie_var:5.2f}")
    print(f"Trie entropy in bits : {trie_entropy:5.2f}")
    print(f"SPE mean tokens : {spe_avg:5.2f}")
    print(f"SPE variance in tokens/mol : {spe_var:5.2f}")
    print(f"SPE entropy in bits : {spe_entropy:5.2f}")
    print(f"Trie+TTG mean tokens : {trie_ttg_avg:5.2f}")
    print(f"Trie+TTG variance in tokens/mol : {trie_ttg_var:5.2f}")
    print(f"Trie+TTG entropy in bits : {trie_ttg_entropy:5.2f}")
    print(f"Reduction (APE vs Trie)        : {(ape_avg-trie_avg)/ape_avg*100:4.1f}%")
    print(f"Reduction (SPE vs Trie)        : {(spe_avg-trie_avg)/spe_avg*100:4.1f}%")
    print(f"Reduction (Trie+TTG vs Trie)   : {(trie_avg-trie_ttg_avg)/trie_avg*100:4.1f}%")
    print(f"Total wall-time  : {time.time()-t0:.1f}s")

    total_saved = [0]

    for x in tqdm.tqdm(iter_smiles(SLICE), desc="APE vs Trie"):
        trie_len = tf.compress_and_len(x, trie_state)
        ape_len = len(ape.encode(x))
        total_saved.append((total_saved[-1] + (ape_len - trie_len)))

    total_saved = total_saved[1:]

    plt.plot(range(1, len(total_saved)+1), total_saved)
    plt.xlabel("Dataset Size")
    plt.ylabel("Total tokens saved")
    plt.show()


if __name__ == "__main__":
    main()
