# E2. ttg and pipeline schematic (clarity)
from enum import EnumDict
from ape_tokenizer import APETokenizer
from collections import Counter
from SmilesPE.tokenizer import *
from utils import iter_smiles
import trie_funcs as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json, pickle, codecs, tqdm, time, math, random

SLICE = "../data/chembl_test.parquet"
APE_DIR = "../train/ape_chembl"  # 48452.2s for training
TRIE_FILE = "../train/trie_chembl.pkl"  # 12.9s for training
TRIE_TTG_FILE = "../train/trie_ttg_chembl.pkl"  # 11.6s for training
SPE_FILE = "../train/spe_chembl.txt"  # 82.5s for training


def token_vect(token, ttg, default):
    """
    Look up the token vector (v_Ri).
    `token` is a standard token parsed by the tokenizer.
    `ttg` is a matrix of all token transition probabilities (ij entry is probability token j follows token i).
    Output is a vector of conditional probabilities of transition probabilities from `token` → arbitrary token.
    """
    return ttg.get(token, default)


def token_list(vector, keys):
    """
    Given a token vector, which is a dictionary of key to probabilities,
    and a list of keys, organize the token vector into a list where entries are sorted according to `keys`
    """
    return np.array([vector.get(key, 0) for i, key in enumerate(keys)])


def molecule_repr(token_repr, ttg, keys):
    """
    Molecule-level representation (z_m)
    `token_repr` is vector (R_i1, R_i2, ..., R_iL)
    This function computes the mean of the constituent token vectors.
    """
    l = 0
    z = np.zeros(keys.shape)
    uniform = np.ones(keys.shape[0]) / keys.shape[0]
    for ri in token_repr:
        z += token_list(token_vect(ri, ttg, uniform), keys)
        l += 1
    return z / l


cosine_similarity = lambda z1, z2: np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))
dedup = lambda similarity_matrix: similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)]
first = lambda keys, n: list(keys)[:n]


def visualize_tree(tree, title="Token Transition Tree"):
    """
    Visualize the tree structure using networkx and matplotlib.
    Each node shows the token name and each edge shows the transition probability.
    """
    G = nx.DiGraph()
    pos = {}
    edge_labels = {}

    # Add nodes and edges recursively
    def add_nodes_edges(node_dict, parent=None, level=0, pos_x=0, width=6):
        node_count = len(node_dict)
        if node_count == 0:
            return

        for i, (token, data) in enumerate(node_dict.items()):
            # Calculate position for this node
            if node_count == 1:
                node_pos_x = pos_x
            else:
                node_pos_x = pos_x + (i - (node_count - 1) / 2) * width / max(1, node_count - 1)

            pos[token] = (node_pos_x, -level)

            # Add node
            G.add_node(token)

            # Add edge from parent if exists
            if parent is not None:
                G.add_edge(parent, token)
                edge_labels[(parent, token)] = f"{data['p']:.3f}"

            # Recursively add children
            if data["children"]:
                add_nodes_edges(data["children"], token, level + 1, node_pos_x, width / max(1, len(data["children"])))

    # Start building the graph
    add_nodes_edges(tree)

    # Create the visualization
    plt.figure(figsize=(14, 10))

    # Draw the network
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=9, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray',
            arrowstyle='->', connectionstyle='arc3,rad=0.1')

    # Draw edge labels (probabilities)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, font_color='red')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("../visualisations/output_e2.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    q = 10
    all_mols = list(iter_smiles(SLICE))
    random_mols = random.sample(all_mols, q)

    t0 = time.time()

    ttg = dict()
    keys = []

    with open("../train/ttg/matrix.pkl", "rb") as f:
        ttg = pickle.load(f)
    with open("../train/ttg/keys.json", "r") as f:
        keys = np.array(json.load(f))

    uniform = np.ones(keys.shape[0]) / keys.shape[0]
    token_matrix = np.array([token_list(token_vect(key, ttg, uniform), keys) for key in keys])

    dedup_matrix = dedup(token_matrix)
    mu = np.mean(dedup_matrix)
    sigma = np.std(dedup_matrix)

    # form a tree starting with a single token (use keys to find one)
    # then, choose some random `n` tokens it connects to and form the next layer of the tree
    # do this again and stop
    tree = {}
    root = str(np.random.choice(keys))
    tree[root] = {
      "name": root,
      "p": 1,
      "children": {}
    }

    for child in first(ttg[root].keys(), 3):
        tree[root]["children"][child] = {
            "name": child,
            "p": ttg[root][child],
            "children": {}
        }
        for grandchild in first(ttg[child].keys(), 3):
            tree[root]["children"][child]["children"][grandchild] = {
                "name": grandchild,
                "p": ttg[child][grandchild],
                "children": {}
            }

    t1 = time.time()
    print(f"E2: TTG construction matrix")
    print(tree)
    print(f"Time to generate statistics: {t1 - t0:.4f} seconds")

    # Visualize the tree
    visualize_tree(tree, "Token Transition Tree Visualization")


if __name__ == "__main__":
    main()
