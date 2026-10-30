"""Zero-dependency Fuzzy Fielded Okapi BM25F + Character 3-Gram scorer for tool & skill discovery.

Replaces naive substring counting (`sum(1 for term in query.split() if term in text)`)
with Robertson & Zaragoza's Fielded BM25F (k1=1.5, b=0.75), identifier sub-tokenization
(snake_case, kebab-case, camelCase, PascalCase), deterministic suffix stemming, and
bounded character 3-gram cosine soft-matching for typo/prefix resilience.
"""

from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple

_SPLIT_IDENT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]+")

_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
    "for", "from", "how", "i", "in", "into", "is", "it", "me", "my",
    "of", "on", "or", "please", "so", "that", "the", "their", "them",
    "this", "to", "up", "use", "using", "want", "we", "what", "when",
    "where", "which", "who", "with", "would", "you", "your",
}


def _stem_token(tok: str) -> str:
    """Lightweight, deterministic morphological stemmer preserving root discriminability."""
    n = len(tok)
    if n <= 3:
        return tok
    if tok.endswith("ies") and n > 4:
        return tok[:-3] + "y"
    if tok.endswith("ing") and n > 5:
        base = tok[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        return base
    if tok.endswith("ations") and n > 7:
        return tok[:-6]
    if tok.endswith("ation") and n > 6:
        return tok[:-5]
    if tok.endswith("ments") and n > 7:
        return tok[:-5]
    if tok.endswith("ment") and n > 6:
        return tok[:-4]
    if tok.endswith("ed") and n > 4:
        base = tok[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        return base
    elif tok.endswith("es") and n > 4 and tok[-3] in "sxzho":
        tok = tok[:-2]
    elif tok.endswith("s") and not tok.endswith("ss") and n > 3 and not tok[-2].isdigit():
        tok = tok[:-1]
    if len(tok) >= 4 and tok.endswith("e") and not tok.endswith("ee"):
        tok = tok[:-1]
    return tok


@lru_cache(maxsize=4096)
def tokenize_text(text: str, drop_stopwords: bool = True) -> Tuple[str, ...]:
    """Split identifiers (camelCase, snake_case) and prose into stemmed lexical tokens."""
    if not text:
        return ()
    raw_tokens: List[str] = []
    for chunk in _NON_ALNUM_RE.split(text):
        if not chunk:
            continue
        subparts = _SPLIT_IDENT_RE.findall(chunk)
        if len(subparts) > 1:
            lowered_parts = [sp.lower() for sp in subparts]
            raw_tokens.extend(lowered_parts)
            # Emit adjacent compound subtokens (e.g. fetchGitHubPull -> "github", "git", "hub")
            for idx in range(len(lowered_parts) - 1):
                raw_tokens.append(lowered_parts[idx] + lowered_parts[idx + 1])
            joined = chunk.lower()
            if len(joined) <= 24:
                raw_tokens.append(joined)
        else:
            raw_tokens.append(chunk.lower())

    if not raw_tokens:
        return ()

    filtered = [t for t in raw_tokens if t not in _STOPWORDS] if drop_stopwords else raw_tokens
    # Never erase a query composed exclusively of stopwords
    active = filtered if filtered else raw_tokens
    return tuple(_stem_token(t) for t in active if t)


@lru_cache(maxsize=4096)
def _char_trigrams(word: str) -> Tuple[Dict[str, int], float]:
    padded = f"^{word}$"
    counts: Dict[str, int] = {}
    for i in range(len(padded) - 2):
        tri = padded[i : i + 3]
        counts[tri] = counts.get(tri, 0) + 1
    norm = math.sqrt(sum(v * v for v in counts.values()))
    return counts, norm


def char_trigram_cosine(a: str, b: str) -> float:
    """Compute character 3-gram cosine similarity in [0.0, 1.0] between two tokens."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la < 3 or lb < 3:
        return 0.0
    ga, norm_a = _char_trigrams(a)
    gb, norm_b = _char_trigrams(b)
    if len(ga) > len(gb):
        ga, gb = gb, ga
    dot = 0
    for k, va in ga.items():
        vb = gb.get(k)
        if vb:
            dot += va * vb
    if dot == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_param_text(parameters: Any) -> str:
    if not isinstance(parameters, dict):
        return ""
    props = parameters.get("properties", parameters)
    if not isinstance(props, dict):
        return ""
    parts: List[str] = []
    for pname, pinfo in props.items():
        parts.append(str(pname))
        if isinstance(pinfo, dict):
            desc = pinfo.get("description")
            if isinstance(desc, str):
                parts.append(desc)
    return " ".join(parts)


def rank_bm25f(
    query: str,
    items: Sequence[Any],
    field_extractor: Callable[[Any], Tuple[str, str, str]],
    *,
    limit: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
    weights: Tuple[float, float, float] = (2.8, 1.4, 1.0),
    fuzzy_threshold: float = 0.55,
) -> List[Any]:
    """Rank items using Fielded Okapi BM25F + bounded character 3-gram soft-matching.

    Args:
        query: Natural-language or identifier search query.
        items: Candidate objects (e.g. deferred ``Tool`` instances).
        field_extractor: Callable returning ``(name_text, params_text, desc_text)``.
        limit: Maximum ranked items to return.
        k1: BM25 term-frequency saturation parameter (default 1.5).
        b: Field-length normalization factor in [0, 1] (default 0.75).
        weights: Field boost weights ``(w_name, w_params, w_desc)``.
        fuzzy_threshold: Minimum token-level character 3-gram cosine to admit typo matches.

    Returns:
        Top ``limit`` items with strictly positive relevance scores.
    """
    if not items or limit <= 0:
        return []

    q_terms = tokenize_text(query, drop_stopwords=True)
    if not q_terms:
        return []
    # Deduplicate query terms while preserving order (weight repeated query terms gently)
    q_counts: Dict[str, int] = {}
    unique_q: List[str] = []
    for qt in q_terms:
        if qt not in q_counts:
            unique_q.append(qt)
            q_counts[qt] = 1
        else:
            q_counts[qt] += 1

    n_docs = len(items)
    # Tokenize all 3 fields for every item
    doc_fields: List[Tuple[List[str], List[str], List[str]]] = []
    field_len_sums = [0, 0, 0]
    vocab: Set[str] = set()

    for item in items:
        f0, f1, f2 = field_extractor(item)
        t0 = tokenize_text(f0, drop_stopwords=False)
        t1 = tokenize_text(f1, drop_stopwords=True)
        t2 = tokenize_text(f2, drop_stopwords=True)
        doc_fields.append((t0, t1, t2))
        field_len_sums[0] += len(t0)
        field_len_sums[1] += len(t1)
        field_len_sums[2] += len(t2)
        vocab.update(t0)
        vocab.update(t1)
        vocab.update(t2)

    avg_lens = [max(s / n_docs, 1.0) for s in field_len_sums]

    # Map each query term to exact matches or fuzzy 3-gram matches in corpus vocabulary
    q_expansion: Dict[str, List[Tuple[str, float]]] = {}
    min_len_ratio = fuzzy_threshold * fuzzy_threshold
    for qt in unique_q:
        matches: List[Tuple[str, float]] = []
        if qt in vocab:
            matches.append((qt, 1.0))
        elif len(qt) >= 4:
            l_qt = len(qt)
            for vt in vocab:
                l_vt = len(vt)
                if l_vt < 4:
                    continue
                # Cauchy-Schwarz upper bound on padded trigram overlap
                if min(l_qt, l_vt) < min_len_ratio * max(l_qt, l_vt):
                    continue
                sim = char_trigram_cosine(qt, vt)
                if sim >= fuzzy_threshold:
                    matches.append((vt, 0.65 * sim))
        q_expansion[qt] = matches

    # Precompute document frequencies for matched vocabulary tokens
    matched_vocab = {vt for pairs in q_expansion.values() for vt, _ in pairs}
    if not matched_vocab:
        return []

    df: Dict[str, int] = {vt: 0 for vt in matched_vocab}
    doc_tf_maps: List[Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]] = []

    for t0, t1, t2 in doc_fields:
        m0: Dict[str, int] = {}
        for tok in t0:
            if tok in df:
                m0[tok] = m0.get(tok, 0) + 1
        m1: Dict[str, int] = {}
        for tok in t1:
            if tok in df:
                m1[tok] = m1.get(tok, 0) + 1
        m2: Dict[str, int] = {}
        for tok in t2:
            if tok in df:
                m2[tok] = m2.get(tok, 0) + 1

        seen_in_doc = set(m0) | set(m1) | set(m2)
        for tok in seen_in_doc:
            df[tok] += 1
        doc_tf_maps.append((m0, m1, m2))

    # Lucene/BM25+ non-negative smooth IDF (strictly > 0 even when N == 1)
    idf_qt: Dict[str, float] = {}
    for qt, pairs in q_expansion.items():
        if not pairs:
            idf_qt[qt] = 0.0
            continue
        # Effective document frequency is minimum df among exact match or best fuzzy expansion
        best_vt = max(pairs, key=lambda p: p[1])[0]
        doc_freq = df.get(best_vt, 0)
        idf_qt[qt] = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    scored: List[Tuple[float, int, Any]] = []
    for idx, item in enumerate(items):
        m0, m1, m2 = doc_tf_maps[idx]
        if not m0 and not m1 and not m2:
            continue
        t0, t1, t2 = doc_fields[idx]
        lens = (len(t0), len(t1), len(t2))
        maps = (m0, m1, m2)

        doc_score = 0.0
        for qt in unique_q:
            pairs = q_expansion[qt]
            if not pairs:
                continue
            pseudo_tf = 0.0
            for f_idx in range(3):
                f_map = maps[f_idx]
                if not f_map:
                    continue
                raw_f_tf = 0.0
                for vt, sim_weight in pairs:
                    cnt = f_map.get(vt, 0)
                    if cnt:
                        raw_f_tf += cnt * sim_weight
                if raw_f_tf > 0.0:
                    norm = 1.0 - b + b * (lens[f_idx] / avg_lens[f_idx])
                    pseudo_tf += weights[f_idx] * (raw_f_tf / norm)

            if pseudo_tf > 0.0:
                q_boost = 1.0 + 0.25 * (q_counts[qt] - 1)
                tf_sat = (pseudo_tf * (k1 + 1.0)) / (pseudo_tf + k1)
                doc_score += idf_qt[qt] * tf_sat * q_boost

        if doc_score > 0.0:
            # Tie-break deterministically by original registration order (-idx when reversed)
            scored.append((doc_score, -idx, item))

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return [item for _, _, item in scored[:limit]]


def rank_tools_bm25f(query: str, tools: Iterable[Any], limit: int = 5) -> List[Any]:
    """Rank ``Tool`` instances across ``name``, ``parameters``, and ``description``."""
    tool_list = list(tools)
    return rank_bm25f(
        query,
        tool_list,
        lambda t: (
            getattr(t, "name", "") or "",
            _extract_param_text(getattr(t, "parameters", None)),
            getattr(t, "description", "") or "",
        ),
        limit=limit,
    )
