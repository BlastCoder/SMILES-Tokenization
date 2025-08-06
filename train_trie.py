# train_trie.py
from utils import iter_smiles
import trie_funcs as tf
import time, os

SLICE = "data/chembl_train_100K.parquet"
OUT   = "trie_ttg_chembl.pkl"

def main():
    print("Building trie compressor …")
    t0 = time.time()
    state = tf.prepare_compressor_with_ttg(iter_smiles(SLICE), K=12, freq_thr=2, entropy_thr=3.5)
    tf.save_state(state, OUT)
    print(f"✔ Trie saved → {OUT}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
