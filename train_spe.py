import codecs
import pyarrow.dataset as ds, pyarrow as pa, pyarrow.parquet as pq, os
from SmilesPE.learner import *
from utils import iter_smiles

file_name = "data/pubchem_100K.parquet"

SMILES = list(iter_smiles(file_name))
print('Number of SMILES:', len(SMILES))

output = codecs.open('spe_pubchem100K.txt', 'w')
learn_SPE(SMILES, output, 30000, min_frequency=2000, augmentation=1, verbose=True, total_symbols=True)
