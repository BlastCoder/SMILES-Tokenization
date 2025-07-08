# train_ape.py
from utils import iter_smiles
from ape_tokenizer import APETokenizer
import time, os

SLICE = "data/pubchem_1M.parquet"          # use make_slice.py once
OUT   = "ape_pubchem1M"                    # folder will be created

def main():
    os.makedirs(OUT, exist_ok=True)
    ape = APETokenizer()
    print("Training APE …")
    t0 = time.time()
    ape.train(iter_smiles(SLICE), max_vocab_size=8000, min_freq_for_merge=800)
    ape.save_pretrained(OUT)
    print(f"✔ APE saved → {OUT}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
