# train_trie.py
from utils import iter_smiles
import trie_funcs as tf
import time, os

SLICE = "data/pubchem_1M.parquet"
OUT   = "trie_pubchem1M.pkl"

def main():
    print("Building trie compressor …")
    t0 = time.time()
    state = tf.prepare_compressor(iter_smiles(SLICE), K=8, freq_thr=4)
    tf.save_state(state, OUT)
    print(f"✔ Trie saved → {OUT}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
