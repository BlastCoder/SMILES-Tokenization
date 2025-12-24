import pyarrow.dataset as ds

def iter_smiles(path: str, column: str = "SMILES"):
    """
    Yield SMILES strings lazily from a Parquet *directory* or *file*.
    """
    dataset = ds.dataset(path, format="parquet")      # auto-detect shards or file
    for rb in dataset.scanner(columns=[column]).to_batches():
        for s in rb.column(column).to_pylist():
            yield s
