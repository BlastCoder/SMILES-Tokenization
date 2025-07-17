# bench_lengths.py
from utils import iter_smiles
from ape_tokenizer import APETokenizer
import trie_funcs as tf
import numpy as np
import tqdm
import time
import matplotlib.pyplot as plt

SLICE = "data/pubchem_100K.parquet"
APE_DIR = "ape_pubchem100K"
TRIE_FILE = "trie_pubchem100K.pkl"


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


def main():
    ape = APETokenizer.from_pretrained(APE_DIR)
    trie_state = tf.load_state(TRIE_FILE)

    t0 = time.time()
    ape_avg = mean_len(iter_smiles(SLICE),
                       lambda s: len(ape.encode(s)),
                       "APE  ")
    ape_var = var_len(iter_smiles(SLICE),
                       lambda s: len(ape.encode(s)),
                       "APE variance  ",
                       ape_avg)
    trie_avg = mean_len(iter_smiles(SLICE),
                        lambda s: tf.compress_and_len(s, trie_state),
                        "Trie ")
    trie_var = var_len(iter_smiles(SLICE),
                        lambda s: tf.compress_and_len(s, trie_state),
                        "Trie variance  ",
                        trie_avg)

    print(f"\nAPE mean tokens : {ape_avg:5.2f}")
    print(f"APE variance in tokens/mol : {ape_var:5.2f}")
    print(f"Trie mean tokens : {trie_avg:5.2f}")
    print(f"Trie variance in tokens/mol : {trie_var:5.2f}")
    print(f"Reduction        : {(ape_avg-trie_avg)/ape_avg*100:4.1f}%")
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
