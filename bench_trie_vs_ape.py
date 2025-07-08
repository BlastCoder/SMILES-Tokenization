from datasets import load_dataset

# 2️⃣  STREAMING avoids loading 2 GB into memory:
ds_iter = load_dataset(
    "mikemayuare/PubChem10M_SMILES_SELFIES",
    split="train",
    streaming=True
)

# Grab a reproducible 1 M-mol *training* slice for both tokenisers
train_iter = ds_iter.take(1000000)

# Grab a disjoint 100 k-mol *eval* slice for average-length stats
eval_iter  = ds_iter.skip(1000000).take(100000)
