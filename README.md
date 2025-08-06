# SMILES Tokenization

The files `train_ape.py`, `train_spe.py`, and `train_trie.py` contain the code for building the Atom-Pair Encoding, SMILES-Pair Encoding, and Trie-based encoding compressors from the PubChem parquet data files under the `data/` folder.

## Usage

To run, first install the necessary requirements:

```sh
pip install -r requirements.txt
```

Then, run the training files (`train_ape.py`, etc.) for the desired compressors and run `bench_lengths.py` to compare the compressions generated form the SPE, APE, Trie, and Trie+TTG algorithms.

## Results

The results will be visible in both the standard output of `bench_lenghts.py` and the `visualisations` folder, which contains JSON formatted data for the histograms and the histograms themselves for each of the compression algorithms. Other outputs are stored in `data/`, such as runtime.
