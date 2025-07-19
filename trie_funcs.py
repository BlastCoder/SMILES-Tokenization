"""
Token-based trie compressor — stand-alone, minimal, no external deps.
Designed for the PubChem 1-M slice benchmark.

Public surface:
    prepare_compressor(iterator, K=8, freq_thr=4) -> state
    compress_and_len(smiles_str, state)           -> int
    save_state(state, fname);  load_state(fname)  -> state
"""

import re, pickle, collections, math
from collections import defaultdict

# ────────────────────────────────────────────────────────────────────────────────
# 1. basic tokenisation
TOKEN_PATTERN = re.compile(
    r"(\[[^\[\]]+\]|Br?|Cl?|[A-Z][a-z]?|\d+|=|\/|\\|\+|\-|\(|\)|@|\[|\])"
)

def tokenize(smiles: str):
    return TOKEN_PATTERN.findall(smiles)

# ────────────────────────────────────────────────────────────────────────────────
# 2. array-trie node
class Node:
    __slots__ = ("children", "count", "replacement")
    def __init__(self, sigma: int):
        self.children   = [None] * sigma   # fixed-size table
        self.count      = 0
        self.replacement = None            # None or str

# ────────────────────────────────────────────────────────────────────────────────
# 3. build counted trie (Algorithms 3-5 in the paper)
def build_trie(seqs, token_to_idx, K, sigma):
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

# ────────────────────────────────────────────────────────────────────────────────
def collect_freq(root, k, idx_to_tok, sigma, thr):
    """
    DFS to gather substrings of length k with freq ≥ thr.
    Returns list[(tuple[str], freq)] sorted by freq desc.
    """
    out = []
    buf = []

    def dfs(node, depth):
        if depth == k:
            if node.count >= thr:
                out.append((tuple(buf), node.count))
            return
        for idx in range(sigma):
            child = node.children[idx]
            if child is None: continue
            buf.append(idx_to_tok[idx])
            dfs(child, depth + 1)
            buf.pop()

    dfs(root, 0)
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ────────────────────────────────────────────────────────────────────────────────
# 4. build replacement trie (Algorithm 7)
class ReplaceTrie:
    __slots__ = ("children", "replacement")
    def __init__(self):
        self.children   = dict()       # ragged dict for compactness
        self.replacement = None

def insert_replace(rt_root, pattern, new_token):
    node = rt_root
    for tok in pattern:
        node = node.children.setdefault(tok, ReplaceTrie())
    node.replacement = new_token

def compress(tokens, rt_root):
    out, i, n = [], 0, len(tokens)
    while i < n:
        node = rt_root
        j, last_rep, last_len = i, None, 0
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
# 5. high-level driver
class _State(collections.namedtuple("_State",
        "token_to_idx idx_to_token replace_root")):
    __slots__ = ()

def prepare_compressor(smiles_iter, K=8, freq_thr=4):
    # 5.1 collect first pass — tokenise & build alphabet
    seqs = []
    alphabet = set()
    for s in smiles_iter:
        toks = tokenize(s)
        seqs.append(toks)
        alphabet.update(toks)

    alphabet = sorted(alphabet)
    token_to_idx = {t: i for i, t in enumerate(alphabet)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}
    sigma = len(alphabet)

    # 5.2 counted trie
    counted_root = build_trie(seqs, token_to_idx, K, sigma)

    # 5.3 find frequent substrings and assign replacement tokens
    replace_root = ReplaceTrie()
    rep_counter  = 0
    for k in range(K, 1, -1):       # longest first
        for patt, freq in collect_freq(counted_root, k, idx_to_token, sigma, freq_thr):
            new_tok = f"<R{rep_counter}>"
            rep_counter += 1
            insert_replace(replace_root, patt, new_tok)

    return _State(token_to_idx, idx_to_token, replace_root)

# ────────────────────────────────────────────────────────────────────────────────
# 6. helpers for the benchmark script
def compress_and_len(smiles: str, state: _State) -> int:
    return len(compress(tokenize(smiles), state.replace_root))

def save_state(state, fname):
    with open(fname, "wb") as f:
        pickle.dump(state, f)

def load_state(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
    
def compress_and_return(s, state):
    """
    Return the list of tokens produced by the trie tokenizer.
    """
    return state.replace_root.tokenize(s)


# ────────────────────────────────────────────────────────────────────────────────
# 7. token transition graph
class TokenTransitionGraph:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
        
        # special tokens
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"
        
        # vocab tracking
        self.vocabulary = set()
        self.total_sequences = 0
        
        # computed statistics (lazy computation)
        self._transition_probs = None
        self._token_entropies = None
        self._successor_counts = None
    
    def build_from_corpus(self, smiles_iter):
        """Build TTG by processing each SMILES sequence"""
        
        for smiles_string in smiles_iter:
            tokens = tokenize(smiles_string)  # Your existing function
            if not tokens:
                continue
                
            # Add all tokens to vocabulary
            self.vocabulary.update(tokens)
            
            # Process transitions in this sequence
            self._add_sequence_transitions(tokens)
            self.total_sequences += 1
    
    def _add_sequence_transitions(self, tokens):
        """Add all transitions from a single token sequence"""
        
        # START -> first_token
        if tokens:
            self.transitions[self.START_TOKEN][tokens[0]] += 1
        
        # token[i] -> token[i+1] for all adjacent pairs
        for i in range(len(tokens) - 1):
            from_token = tokens[i]
            to_token = tokens[i + 1]
            self.transitions[from_token][to_token] += 1
        
        # last_token -> END
        if tokens:
            self.transitions[tokens[-1]][self.END_TOKEN] += 1
    
    def _compute_transition_probabilities(self):
        """Convert transition counts to probabilities P(y|x)"""
        
        self._transition_probs = {}
        self._successor_counts = {}
        
        for from_token in self.transitions:
            # calculate total outgoing transitions for this token
            total_out = sum(self.transitions[from_token].values())
            self._successor_counts[from_token] = total_out
            
            # convert each transition to probability
            self._transition_probs[from_token] = {}
            for to_token, count in self.transitions[from_token].items():
                self._transition_probs[from_token][to_token] = count / total_out
    
    def _compute_token_entropies(self):
        """Calculate H(x) = -Σ P(y|x) * log2(P(y|x))"""
        
        if self._transition_probs is None:
            self._compute_transition_probabilities()
        
        self._token_entropies = {}
        
        for from_token in self._transition_probs:
            entropy = 0.0
            
            for to_token, prob in self._transition_probs[from_token].items():
                if prob > 0: 
                    entropy -= prob * math.log2(prob)
            
            self._token_entropies[from_token] = entropy
    
    def get_high_confidence_paths(self, max_length, min_frequency=None, 
                            min_probability=None, max_entropy=None):
        """Generate token sequences meeting confidence criteria"""
        
        if self._transition_probs is None:
            self._compute_transition_probabilities()
        if self._token_entropies is None:
            self._compute_token_entropies()
        
        high_confidence_paths = []
        
        # start DFS from each token in vocabulary
        for start_token in self.vocabulary:
            if start_token in self.transitions:
                if self._token_passes_filters(start_token, max_entropy):
                    paths = self._dfs_paths(start_token, max_length, 
                                        min_frequency, min_probability, max_entropy)
                    high_confidence_paths.extend(paths)
        
        return high_confidence_paths

    def _dfs_paths(self, current_token, remaining_length, 
               min_freq, min_prob, max_entropy, current_path=None):
        """Recursively find high-confidence paths"""
        
        if current_path is None:
            current_path = []
        
        current_path = current_path + [current_token]
        
        # base case
        if remaining_length <= 1:
            if len(current_path) >= 3: 
                return [current_path]
            else:
                return []
        
        paths = []
        
        # explore all valid successors
        for next_token in self.transitions[current_token]:
            if self._transition_passes_filters(current_token, next_token, 
                                            min_freq, min_prob, max_entropy):
                
                sub_paths = self._dfs_paths(next_token, remaining_length - 1,
                                        min_freq, min_prob, max_entropy, 
                                        current_path)
                paths.extend(sub_paths)
        
        return paths
    
    def _transition_passes_filters(self, from_token, to_token, 
                             min_freq, min_prob, max_entropy):
        """Check if a transition meets all criteria"""
        
        # ensure probabilities are computed
        if self._transition_probs is None:
            self._compute_transition_probabilities()
        
        # frequency filter
        if min_freq is not None:
            if self.transitions[from_token][to_token] < min_freq:
                return False
        
        # probability filter  
        if min_prob is not None:
            prob = self._transition_probs[from_token][to_token]
            if prob < min_prob:
                return False
        
        # source token entropy filter
        if max_entropy is not None:
            entropy = self._token_entropies[from_token]
            if entropy > max_entropy:
                return False
        
        return True

    def _token_passes_filters(self, token, max_entropy):
        """Check if a token meets entropy criteria"""
        
        if max_entropy is not None:
            if self._token_entropies.get(token, float('inf')) > max_entropy:
                return False
        
        return True
    
    def get_path_probability(self, token_sequence):
        """Calculate P(t1|START) * P(t2|t1) * ... * P(END|tn)"""
        
        if self._transition_probs is None:
            self._compute_transition_probabilities()
        
        if not token_sequence:
            return 0.0
        
        prob = 1.0
        
        # START -> first_token
        first_token = token_sequence[0]
        if first_token in self._transition_probs[self.START_TOKEN]:
            prob *= self._transition_probs[self.START_TOKEN][first_token]
        else:
            return 0.0
        
        # token[i] -> token[i+1]
        for i in range(len(token_sequence) - 1):
            from_token = token_sequence[i]
            to_token = token_sequence[i + 1]
            
            if to_token in self._transition_probs.get(from_token, {}):
                prob *= self._transition_probs[from_token][to_token]
            else:
                return 0.0
        
        # last_token -> END
        last_token = token_sequence[-1]
        if self.END_TOKEN in self._transition_probs.get(last_token, {}):
            prob *= self._transition_probs[last_token][self.END_TOKEN]
        else:
            return 0.0
        
        return prob

# ────────────────────────────────────────────────────────────────────────────────
# 8. helpers for ttg
def count_subsequence_in_sequence(subseq, sequence):
    """Count non-overlapping occurrences of subseq in sequence"""
    count = 0
    i = 0
    while i <= len(sequence) - len(subseq):
        if sequence[i:i+len(subseq)] == subseq:
            count += 1
            i += len(subseq)  # Non-overlapping
        else:
            i += 1
    return count

# built with the help of Claude for efficiency
def prepare_compressor_with_ttg(smiles_iter, K=8, freq_thr=4, 
                               ttg_min_prob=0.1, ttg_max_entropy=2.0):
    """Enhanced compressor using TTG guidance"""
    
    # Convert iterator to list for multiple passes
    smiles_list = list(smiles_iter)
    
    # Step 1: Build TTG from corpus
    ttg = TokenTransitionGraph()
    ttg.build_from_corpus(smiles_list)
    
    # Step 2: Get high-confidence token sequences
    candidate_paths = ttg.get_high_confidence_paths(
        max_length=K,
        min_probability=ttg_min_prob,
        max_entropy=ttg_max_entropy
    )
    
    # Step 3: Build alphabet from original corpus (unchanged)
    seqs = []
    alphabet = set()
    for s in smiles_list:
        toks = tokenize(s)
        seqs.append(toks)
        alphabet.update(toks)
    
    alphabet = sorted(alphabet)
    token_to_idx = {t: i for i, t in enumerate(alphabet)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}
    sigma = len(alphabet)
    
    # Step 4: Build filtered trie using only high-confidence paths
    replace_root = ReplaceTrie()
    rep_counter = 0
    
    # Convert candidate paths to frequency counts
    path_frequencies = defaultdict(int)
    for path in candidate_paths:
        path_tuple = tuple(path)
        # Count how many times this path appears in actual corpus
        for seq in seqs:
            path_frequencies[path_tuple] += count_subsequence_in_sequence(path, seq)
    
    # Process paths by decreasing length (longest first)
    for k in range(K, 1, -1):
        valid_paths = [(path, freq) for path, freq in path_frequencies.items() 
                      if len(path) == k and freq >= freq_thr]
        
        # Sort by frequency (highest first)
        valid_paths.sort(key=lambda x: x[1], reverse=True)
        
        for path, freq in valid_paths:
            new_tok = f"<R{rep_counter}>"
            rep_counter += 1
            insert_replace(replace_root, path, new_tok)
    
    return _State(token_to_idx, idx_to_token, replace_root)