# train_trie.py
from utils import iter_smiles
import trie_funcs as tf
import time, os
import pandas as pd

OUT   = "trie_ttg_peptide.pkl"

data = pd.read_csv("./PeptideCLM/clustered_data/all_clusters.csv")
SMILES = data[data['cluster'] != 1]['SMILES'].to_list()

print('Number of SMILES:', len(SMILES))

def main():
    print("Building trie compressor …")
    t0 = time.time()
    state = tf.prepare_compressor_with_ttg(SMILES, K=12, freq_thr=2, entropy_thr=3.5)
    tf.save_state(state, OUT)
    print(f"✔ Trie saved → {OUT}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
