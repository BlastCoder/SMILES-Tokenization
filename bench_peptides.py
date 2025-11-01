# bench_lengths.py
from utils import iter_smiles
from SmilesPE.tokenizer import SPE_Tokenizer
from collections import Counter
import trie_funcs as tf
import codecs
import tqdm
import math
import pandas as pd
from PeptideCLM.tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

SLICE = "data/peptide/peptides_1M.parquet"
APE_DIR = "ape_peptide"  # 48452.2s for training
TRIE_FILE = "./trie_ttg_peptide.pkl"  # 12.9s for training
TRIE_TTG_FILE = "trie_ttg_peptide.pkl"  # 11.6s for training
SPE_FILE = "spe_peptide.txt"  # 82.5s for training
VOCAB_FILE = "PeptideCLM/tokenizer/new_vocab.txt"
SPLITS_FILE = "PeptideCLM/tokenizer/new_splits.txt"

def gen(mols, fn, desc):
    tot = n = tot_v = ent = 0
    token_freq = Counter()
    for mol in tqdm.tqdm(mols, desc=f"{desc} - Pass 1"):
        tokens = fn(mol)
        n += 1
        tot += len(tokens)
        token_freq.update(tokens)

    mean = tot / n

    for mol in tqdm.tqdm(mols, desc=f"{desc} - Pass 2"):
        tot_v += (len(fn(mol)) - mean) ** 2

    var = tot_v / n

    for count in tqdm.tqdm(token_freq.values(), desc=f"{desc} - Pass 3"):
        p = count / tot
        ent -= p * math.log2(p)
        
    return mean, var, ent

def gen_pep(mols, tokenized):
    n = len(mols)
    token_freq = Counter()
    tot = ent = tot_v = 0
    for mol in tqdm.tqdm(tokenized, "Peptides - Pass 1"):
        token_freq.update(mol)
        tot += len(mol)

    mean = tot / n

    for mol in tqdm.tqdm(tokenized, desc="Peptides - Pass 2"):
        tot_v += (len(mol) - mean) ** 2

    var = tot_v / n

    for count in tqdm.tqdm(token_freq.values(), desc="Peptides - Pass 3"):
        p = count / tot
        ent -= p * math.log2(p)

    return mean, var, ent
  
def main():
    trie_ttg_state = tf.load_state(TRIE_TTG_FILE)
    spe_vocab = codecs.open(SPE_FILE)
    spe = SPE_Tokenizer(spe_vocab)
    peptideclm = SMILES_SPE_Tokenizer(VOCAB_FILE, SPLITS_FILE)

    #data = pd.read_csv("./PeptideCLM/clustered_data/all_clusters.csv")
    mols = list(iter_smiles(SLICE))[:100000]
    tokenized = peptideclm(mols)['input_ids']
     
    trie_ttg_avg, trie_ttg_var, trie_ttg_entropy = gen(mols,
                                        lambda s: tf.compress_and_return(s, trie_ttg_state),
                                        "Trie-TTG")
    spe_avg, spe_var, spe_entropy = gen(mols,
                                        lambda s: spe.tokenize(s).split(" "),
                                        "SPE")
    pep_avg, pep_var, pep_entropy = gen_pep(mols, tokenized)
   
    print(f"SPE mean tokens: {spe_avg}")
    print(f"SPE variance in tokens/mol: {spe_var}")
    print(f"SPE entropy in bits: {spe_entropy}")
    print(f"Trie+TTG mean tokens: {trie_ttg_avg}")
    print(f"Trie+TTG variance in tokens/mol: {trie_ttg_var}")
    print(f"Trie+TTG entropy in bits: {trie_ttg_entropy}")
    print(f"PeptideCLM mean tokens: {pep_avg}")
    print(f"PeptideCLM variance in tokens/mol: {pep_var}")
    print(f"PeptideCLM entropy in bits: {pep_entropy}")

    len_orig = 0
    len_trie = 0
    len_spe = 0
    len_pep = 0
    
    for x in range(0,len(mols)):
        len_trie += tf.compress_and_len(mols[x], trie_ttg_state)
        len_spe += len(spe.tokenize(mols[x]).split(" "))
        len_orig += len(tf.tokenize(mols[x]))
        len_pep += len(tokenized[x])
        
    print(f"Compression Ratio - Trie: {len_orig/len_trie}")
    print(f"Compression Ratio - SPE: {len_orig/len_spe}")
    print(f"Compression Ratio - Peptide CLM: {len_orig/len_pep}")

if __name__ == "__main__":
    main()
