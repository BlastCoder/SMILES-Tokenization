from SmilesPE.tokenizer import *
import trie_funcs as tf
import time
import re
from utils import iter_smiles

SLICE = "data/chembl_test.parquet"
TRIE_FILE = "./trie_chembl.pkl"
TRIE_TTG_FILE  = "./trie_ttg_chembl.pkl"
TRIE_REV = "./trie_rev_chembl.pkl"

flat = lambda lst: list((item for sublist in lst for item in sublist))
TOKEN_PATTERN = re.compile(r"(\[[^\[\]]+\]|Br?|Cl?|[A-Z][a-z]?|\d+|=|\/|\\|\+|\-|\(|\)|@|\[|\])")

def bench(smiles, trie_state):
    t0 = time.time()
    tries = []
    for mol in smiles:
        tries.append(tf.compress_and_return(mol, trie_state))
    t = time.time() - t0

    return t, tries

def bench_rev(orig, tokens, rev, trie_state):
    t0 = time.time()
    smiles = []
    for mol in tokens:
        smile = ""
        for token in mol:
            if token.startswith("<R"): smile += rev[token]
            elif TOKEN_PATTERN.match(token, pos=0, endpos=len(token)): smile += token
            elif token.isdigit() and int(token) in trie_state.idx_to_token.keys(): smile += trie_state.idx_to_token[int(token)]
        smiles.append(smile)
    t = time.time() - t0

    e = 0
    for mol in smiles:
        if mol not in orig:
            e+=1
    
    return t, e

def bench_avg(smiles, trie_state, rev_state):
    t1, tok1 = bench(smiles, trie_state)
    t2, tok2 = bench(smiles, trie_state)
    t3, tok3 = bench(smiles, trie_state)

    ret1 = [(t1+t2+t3)/3, (t1+t2+t3)/(3*len(smiles))]

    r1, e1 = bench_rev(smiles, tok1, rev_state, trie_state)
    r2, e2 = bench_rev(smiles, tok1, rev_state, trie_state)
    r3, e3 = bench_rev(smiles, tok1, rev_state, trie_state)

    ret2 = [(r1+r2+r3)/3, (r1+r2+r3)/(3*len(flat(tok1))), (r1+r2+r3)/(3*len(smiles)), (e1+e2+e3)/3]

    return ret1, ret2

def main():
    trie_state = tf.load_state(TRIE_FILE)
    #trie_ttg_state  =  tf.load_state(TRIE_TTG_FILE)
    trie_rev_state  = tf.load_state(TRIE_REV)
    #trie_ttg_rev_state = tf.load_state(TRIE_TTG_REV)

    smiles = list(iter_smiles(SLICE))[0:10000]

    trie, rev = bench_avg(smiles, trie_state, trie_rev_state)
    #ttg = bench_avg(smiles, trie_ttg_state)
    print(f"TRIE\n=============\nAverage: {trie[0]}s\ns/mol:   {trie[1]}s\n")
    print(f"REV\n==============\nAverage: {rev[0]}s\ns/mol:   {rev[2]}s\ns/tok:   {rev[1]}s\nAvg. Error: {rev[3]}\n")
    #print(f"TTG\n=============\nAverage: {ttg[0]}s\ns/mol:   {ttg[1]}s")

if __name__ == "__main__":
    main()
