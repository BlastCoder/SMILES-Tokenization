# make_slice.py
import pyarrow.dataset as ds, pyarrow as pa, pyarrow.parquet as pq, os

SRC_DIR = "data/chembl"                           # the 5-shard directory
DST_FN  = "data/chembl_train_100K.parquet"        # 100K-row slice to create

def main():
    rows = []
    for batch in ds.dataset(SRC_DIR, format="parquet")\
                  .scanner(columns=["SMILES"]).to_batches():
        rows.extend(batch.column("SMILES").to_pylist())
        if len(rows) >= 100000:
            break
    pq.write_table(pa.Table.from_pydict({"SMILES": rows[:100000]}), DST_FN)
    print("✔ wrote", DST_FN)

if __name__ == "__main__":
    main()
