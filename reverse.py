from SmilesPE.tokenizer import *
import trie_funcs as tf
import time
import re
from utils import iter_smiles

SLICE = "data/pubchem_250K.parquet"
TRIE_FILE = "./trie_pubchem.pkl"
TRIE_TTG_FILE  = "./trie_ttg_pubchem.pkl"

flat = lambda lst: list((item for sublist in lst for item in sublist))
TOKEN_PATTERN = re.compile(r"(\[[^\[\]]+\]|Br?|Cl?|[A-Z][a-z]?|\d+|=|\/|\\|\+|\-|\(|\)|@|\[|\])")

def bench(smiles, trie_state):
    t0 = time.time()
    tries = []
    for mol in smiles:
        tries.append(tf.compress_and_return(mol, trie_state))
    t = time.time() - t0

    return t, tries

def dfs(find, root, prefix):
    if root.replacement == find:
        return prefix
    else:
        for child in root.children:
            val = dfs(find, root.children[child], prefix + child)
            if val:
                return val
        return False

def bench_rev(orig, tokens, trie_state):
    t0 = time.time()
    smiles = []
    for mol in tokens:
        smile = ""
        for token in mol:
            rep = dfs(token, trie_state.replace_root, "")
            if rep: smile += rep
            elif token.isdigit() and int(token) in trie_state.idx_to_token.keys():
                smile += trie_state.idx_to_token[int(token)]
            else: smile += token
        smiles.append(smile)
    t = time.time() - t0

    return t

def bench_avg(smiles, trie_state):
    t1, tok1 = bench(smiles, trie_state)
    
    ret1 = [t1, t1/len(smiles)]

    r1 = bench_rev(smiles, tok1, trie_state)

    ret2 = [r1, r1/len(flat(tok1)), r1/len(smiles)]

    return ret1, ret2

def main():
    trie_state = tf.load_state(TRIE_FILE)
    trie_ttg_state  =  tf.load_state(TRIE_TTG_FILE)

    smiles = list(filter(lambda a: a != None, map(tf.canonicalize_smiles, list(iter_smiles(SLICE))[100000:110000])))

    trie, rev = bench_avg(smiles, trie_ttg_state)
    ttg, ttg_rev = bench_avg(smiles, trie_ttg_state)
    print(f"TRIE\n=============\nAverage: {trie[0]}s\ns/mol:   {trie[1]}s\n")
    print(f"REV\n==============\nAverage: {rev[0]}s\ns/mol:   {rev[2]}s\ns/tok:   {rev[1]}s\n")
    print(f"TTG\n=============\nAverage: {ttg[0]}s\ns/mol:   {ttg[1]}s\n")
    print(f"REV\n==============\nAverage: {ttg_rev[0]}s\ns/mol:   {ttg_rev[2]}s\ns/tok:   {ttg_rev[1]}s\n")

if __name__ == "__main__":
    main()
