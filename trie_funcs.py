import re, pickle, collections, math
from collections import defaultdict
from typing import List, Tuple, Iterator

# ────────────────────────────────────────────────────────────────────────────────
# 1. Basic SMILES tokenisation (Paper §3)
TOKEN_PATTERN = re.compile(r"(\[[^\[\]]+\]|Br?|Cl?|[A-Z][a-z]?|\d+|=|\/|\\|\+|\-|\(|\)|@|\[|\])")

def tokenize(smiles: str) -> List[str]:
    """Return list[str] tokens for a SMILES string."""
    return TOKEN_PATTERN.findall(smiles)

# ────────────────────────────────────────────────────────────────────────────────
# 2. Counted‑trie node (Algorithms 3‑5)
class Node:
    __slots__ = ("children", "count", "replacement")
    def __init__(self, sigma: int):
        self.children: List[Node | None] = [None] * sigma  # fixed fan‑out
        self.count: int = 0
        self.replacement: str | None = None  # populated only in ReplaceTrie

# Build counted trie and collect frequent substrings (length ≤ K)

def build_trie(seqs: List[List[str]], token_to_idx: dict[str, int], K: int, sigma: int) -> Node:
    root = Node(sigma)
    for toks in seqs:
        n = len(toks)
        for i in range(n):
            node = root
            for j in range(i, min(i + K, n)):
                idx = token_to_idx[toks[j]]
                if node.children[idx] is None:
                    node.children[idx] = Node(sigma)
                node = node.children[idx]
                node.count += 1
    return root


def collect_freq(root: Node, k: int, idx_to_tok: dict[int, str], sigma: int, thr: int) -> List[Tuple[Tuple[str, ...], int]]:
    """DFS to gather substrings of length *k* with frequency ≥ *thr*."""
    out: List[Tuple[Tuple[str, ...], int]] = []
    buf: List[str] = []

    def dfs(node: Node, depth: int):
        if depth == k:
            if node.count >= thr:
                out.append((tuple(buf), node.count))
            return
        for idx in range(sigma):
            child = node.children[idx]
            if child is None:
                continue
            buf.append(idx_to_tok[idx])
            dfs(child, depth + 1)
            buf.pop()

    dfs(root, 0)
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ────────────────────────────────────────────────────────────────────────────────
# 3. Replacement‑trie for on‑the‑fly compression (Algorithm 7)
class ReplaceTrie:
    __slots__ = ("children", "replacement")
    def __init__(self):
        self.children: dict[str, "ReplaceTrie"] = {}
        self.replacement: str | None = None


def insert_replace(root: ReplaceTrie, pattern: Tuple[str, ...], new_token: str) -> None:
    node = root
    for tok in pattern:
        node = node.children.setdefault(tok, ReplaceTrie())
    node.replacement = new_token


def compress(tokens: List[str], rt_root: ReplaceTrie) -> List[str]:
    out: List[str] = []
    i, n = 0, len(tokens)
    while i < n:
        node = rt_root
        j = i
        last_rep: str | None = None
        last_len: int = 0
        # traverse as far as possible
        while j < n and tokens[j] in node.children:
            node = node.children[tokens[j]]
            if node.replacement is not None:
                last_rep, last_len = node.replacement, j - i + 1
            j += 1
        if last_rep is not None:
            out.append(last_rep)
            i += last_len
        else:
            out.append(tokens[i])
            i += 1
    return out

# ────────────────────────────────────────────────────────────────────────────────
# 4. State container
class _State(collections.namedtuple("_State", "token_to_idx idx_to_token replace_root")):
    __slots__ = ()

# ────────────────────────────────────────────────────────────────────────────────
# 5. Baseline compressor (Algorithms 3‑7)

def prepare_compressor(smiles_iter: Iterator[str], K: int = 8, freq_thr: int = 4) -> _State:
    seqs: List[List[str]] = []
    alphabet: set[str] = set()
    for s in smiles_iter:
        toks = tokenize(s)
        seqs.append(toks)
        alphabet.update(toks)

    alphabet = sorted(alphabet)
    token_to_idx = {t: i for i, t in enumerate(alphabet)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}
    sigma = len(alphabet)

    counted_root = build_trie(seqs, token_to_idx, K, sigma)

    replace_root = ReplaceTrie()
    rep_counter = 0
    for k in range(K, 1, -1):  # longest first
        for patt, freq in collect_freq(counted_root, k, idx_to_token, sigma, freq_thr):
            insert_replace(replace_root, patt, f"<R{rep_counter}>")
            rep_counter += 1

    return _State(token_to_idx, idx_to_token, replace_root)

# ────────────────────────────────────────────────────────────────────────────────
# 6. Token Transition Graph 
class TokenTransitionGraph:
    def __init__(self):
        self.START, self.END = "<START>", "<END>"
        self.transitions: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._transition_probs: dict[str, dict[str, float]] | None = None
        self._token_entropies: dict[str, float] | None = None

    # 6.1 — build raw counts
    def build_from_corpus(self, smiles_list: List[str]):
        for s in smiles_list:
            toks = tokenize(s)
            if not toks:
                continue
            self.transitions[self.START][toks[0]] += 1
            for a, b in zip(toks, toks[1:]):
                self.transitions[a][b] += 1
            self.transitions[toks[-1]][self.END] += 1

    # helper — compute P(y|x)
    def _compute_transition_probs(self):
        if self._transition_probs is not None:
            return
        self._transition_probs = {}
        for x, d in self.transitions.items():
            total = sum(d.values())
            self._transition_probs[x] = {y: c / total for y, c in d.items()}

    # 6.2 — H(x)
    def _compute_token_entropies(self):
        if self._token_entropies is not None:
            return
        self._compute_transition_probs()
        self._token_entropies = {}
        for x, probs in self._transition_probs.items():
            h = 0.0
            for p in probs.values():
                h -= p * math.log2(p)
            self._token_entropies[x] = h

    # average path entropy Ḣ(s)
    def average_path_entropy(self, pattern: Tuple[str, ...]) -> float:
        """Return mean H(x) for every source token along *pattern* (Paper Eq. 11)."""
        if not pattern:
            return float("inf")
        self._compute_token_entropies()
        entropies = [self._token_entropies.get(tok, 0.0) for tok in pattern]
        return sum(entropies) / len(entropies)

# ────────────────────────────────────────────────────────────────────────────────
# 7. Algorithm 8 — TTG‑guided refinement
def prepare_compressor_with_ttg(
    smiles_iter: Iterator[str],
    K: int = 8,
    freq_thr: int = 4,
    entropy_thr: float = 2.0,
) -> _State:

    # materialise corpus (needed twice)
    smiles_list = list(smiles_iter)

    # pass 1 — tokenise and build alphabet
    seqs: List[List[str]] = []
    alphabet: set[str] = set()
    for s in smiles_list:
        toks = tokenize(s)
        seqs.append(toks)
        alphabet.update(toks)

    alphabet = sorted(alphabet)
    token_to_idx = {t: i for i, t in enumerate(alphabet)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}
    sigma = len(alphabet)

    # pass 2 — counted‑trie mining of *all* substrings length 2…K with freq ≥ thr
    counted_root = build_trie(seqs, token_to_idx, K, sigma)
    pattern_freq: dict[Tuple[str, ...], int] = {}
    for k in range(K, 1, -1):
        for patt, freq in collect_freq(counted_root, k, idx_to_token, sigma, freq_thr):
            pattern_freq[patt] = freq

    if not pattern_freq:
        return prepare_compressor(smiles_list, K, freq_thr)

    # pass 3 — build TTG and compute Ḣ(s)
    ttg = TokenTransitionGraph()
    ttg.build_from_corpus(smiles_list)

    filtered_patterns = [p for p in pattern_freq if ttg.average_path_entropy(p) <= entropy_thr]
    filtered_patterns.sort(key=lambda p: (len(p), pattern_freq[p]), reverse=True)

    replace_root = ReplaceTrie()
    rep_counter = 0
    for patt in filtered_patterns:
        insert_replace(replace_root, patt, f"<R{rep_counter}>")
        rep_counter += 1

    return _State(token_to_idx, idx_to_token, replace_root)

# ────────────────────────────────────────────────────────────────────────────────
# 8. Misc helper utilities

def compress_and_len(smiles: str, state: _State) -> int:
    return len(compress(tokenize(smiles), state.replace_root))

def save_state(state: _State, fname: str) -> None:
    with open(fname, "wb") as f:
        pickle.dump(state, f)

def load_state(fname: str) -> _State:
    with open(fname, "rb") as f:
        return pickle.load(f)

def compress_and_return(smiles: str, state: _State) -> List[str]:
    """Return token list after replacement‑trie compression — handy for debugging."""
    return compress(tokenize(smiles), state.replace_root)
