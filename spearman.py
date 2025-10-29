from SmilesPE.tokenizer import *
import trie_funcs as tf
import time
import re
from utils import iter_smiles
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from operator import mul

SLICE = "data/chembl_test.parquet"
STATE = "trie_ttg_chembl.pkl"
TRAIN = "data/chembl_train_100K.parquet"

cap = 100000

fpgen = AllChem.GetMorganGenerator(radius=2)

def main():
    mols = list(iter_smiles(SLICE))[:5000]
    train = list(iter_smiles(TRAIN))
    state = tf.load_state(STATE)
    
    d = []
    
    z = []
    ecfp = []

    ttg = tf.TokenTransitionGraph()
    ttg.build_from_corpus(train)
    ttg._compute_transition_probs()
    
    for mol in mols:
        m = tf.tokenize(tf.canonicalize_smiles(mol))
        v = []
        for tok in m:
            if tok not in ttg._transition_probs.keys():
                ttg._transition_probs[tok] = {}
            for tok1 in m:
                if tok1 not in ttg._transition_probs[tok].keys():
                    ttg._transition_probs[tok][tok1] = 0
            v.append([ttg._transition_probs[tok][x] for x in ttg._transition_probs[tok].keys()])

        z.append([sum(i)/len(v) for i in zip(*v)])
        ecfp.append(fpgen.GetFingerprint(Chem.MolFromSmiles(mol)))

    n = 0

    for i in range(len(z)):
        if n >= cap: break
        for j in range(len(z)):
            if i >= j: continue
            if n >= cap: break
            cos = sum(map(mul, z[i], z[j])) / (sum(map(lambda x: x**2, z[i]))**0.5 * sum(map(lambda x: x**2, z[j]))**0.5)
            tanimoto = DataStructs.TanimotoSimilarity(ecfp[i], ecfp[j])

            d.append((i,j,1-cos,1-tanimoto))
            n+=1

    rank_ttg = sorted(d, key=lambda p: p[2])
    rank_ecfp = sorted(d, key=lambda p: p[3])

    d2_i = []

    for i in range(len(rank_ttg)):
        d2_i.append((i - rank_ecfp.index(rank_ttg[i]))**2)

    print(f"p = {1 - ((6 * sum(d2_i))/(len(rank_ttg) * (len(rank_ttg)**2 - 1)))}")
    
if __name__ == "__main__":
    main()
