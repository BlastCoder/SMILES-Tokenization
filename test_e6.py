import pandas as pd
import numpy as np
from rdkit import Chem
import os

from utils import iter_smiles
from ape_tokenizer import APETokenizer

# -------------------------------------------------------------------
# ESOL constants
# -------------------------------------------------------------------
ESOL_URL = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
TARGET_COL = "measured log solubility in mols per litre"
SMILES_COL = "smiles"


# -------------------------------------------------------------------
# Canonicalization helpers
# -------------------------------------------------------------------
def canonicalize_smiles(smiles: str):
    if len(smiles) > 90:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_esol_and_save_canonical(path_or_url: str = ESOL_URL, save_path="data/esol_canonical.parquet"):
    """
    Download ESOL, canonicalize SMILES, save parquet,
    and return canonical smiles + y.
    """
    df = pd.read_csv(path_or_url)

    required = {SMILES_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in ESOL CSV: {missing}")

    # Canonicalize
    df["smiles_canonical"] = df[SMILES_COL].apply(canonicalize_smiles)
    df = df.dropna(subset=["smiles_canonical"]).reset_index(drop=True)

    # Save canonical SMILES to parquet for tokenizer training
    just_smiles = df[["smiles_canonical"]].copy()
    just_smiles.rename(columns={"smiles_canonical": "SMILES"}, inplace=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    just_smiles.to_parquet(save_path, index=False)

    # Extract arrays
    X_smiles = df["smiles_canonical"].tolist()
    y = df[TARGET_COL].values.astype(float)

    return X_smiles, y, df


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Load ESOL + canonicalize + save parquet
    X_smiles, y, df_esol = load_esol_and_save_canonical()
    print(f"[INFO] Molecules: {len(X_smiles)}")
    print("[INFO] Example canonical SMILES:", X_smiles[0])
    print("[INFO] Example target (logS):", y[0])

    # 2. Train APE tokenizer on canonical ESOL
    SLICE = "data/esol_canonical.parquet"
    ape = APETokenizer()

    print("\n[INFO] Training APE tokenizer...")
    ape.train(
        iter_smiles(SLICE),
        max_vocab_size=8000,
        min_freq_for_merge=800
    )

    # 3. Save tokenizer
    out_dir = "data/exp6/ape"
    os.makedirs(out_dir, exist_ok=True)
    ape.save_pretrained(out_dir)
    print(f"[INFO] APE tokenizer saved to: {out_dir}")
