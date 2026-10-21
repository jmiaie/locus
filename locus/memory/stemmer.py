"""
Porter stemmer: pure standard library, deterministic, no dependencies.

This is an independent implementation, written for Locus, of the algorithm in
M.F. Porter, "An algorithm for suffix stripping", *Program* 14(3):130-137, 1980
(https://tartarus.org/martin/PorterStemmer/).  It follows the published
algorithm (steps 1a-5b), including the two later revisions published by the
author (``bli -> ble``, ``logi -> log``), and its output matches SQLite
FTS5's built-in ``porter`` tokenizer (checked in tests/test_analyzer.py).
Both index backends use this stemmer, so they see the same terms.

Part of Locus; MIT licensed (see LICENSE).  The algorithm itself is
published by its author for free use.
"""

from __future__ import annotations

from functools import lru_cache

__all__ = ["stem"]

_VOWELS = frozenset("aeiou")


def _cons(w: str, i: int) -> bool:
    c = w[i]
    if c in _VOWELS:
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(stem: str) -> int:
    """Number of VC sequences in *stem* ([C](VC)^m[V])."""
    n, i, L = 0, 0, len(stem)
    while i < L and _cons(stem, i):
        i += 1
    while i < L:
        while i < L and not _cons(stem, i):
            i += 1
        if i >= L:
            break
        while i < L and _cons(stem, i):
            i += 1
        n += 1
    return n


def _has_vowel(stem: str) -> bool:
    return any(not _cons(stem, i) for i in range(len(stem)))


def _double_cons(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w: str) -> bool:
    """*o: stem ends cvc, where the final c is not w, x or y."""
    L = len(w)
    if L < 3:
        return False
    return (_cons(w, L - 3) and not _cons(w, L - 2) and _cons(w, L - 1)
            and w[-1] not in "wxy")


def _replace_m(w: str, rules, cond) -> str:
    """Apply the first rule whose suffix matches; replace only if cond(stem)."""
    for suf, rep in rules:
        if w.endswith(suf):
            stem = w[: len(w) - len(suf)]
            return stem + rep if cond(stem) else w
    return w


_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("bli", "ble"), ("logi", "log"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
)
# Longest-match order for overlapping suffixes.
_STEP2 = tuple(sorted(_STEP2, key=lambda r: -len(r[0])))
_STEP3 = tuple(sorted((
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
), key=lambda r: -len(r[0])))
_STEP4 = tuple(sorted((
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
), key=len, reverse=True))


def _porter(w: str) -> str:
    if len(w) <= 2:
        return w
    # Step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # Step 1b
    flag = False
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w, flag = w[:-2], True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w, flag = w[:-3], True
    if flag:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _double_cons(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _m(w) == 1 and _cvc(w):
            w += "e"
    # Step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    # Step 2, 3
    w = _replace_m(w, _STEP2, lambda s: _m(s) > 0)
    w = _replace_m(w, _STEP3, lambda s: _m(s) > 0)
    # Step 4
    for suf in _STEP4:
        if w.endswith(suf):
            stem = w[: len(w) - len(suf)]
            if _m(stem) > 1:
                w = stem
            break
    else:
        if w.endswith("ion"):
            stem = w[:-3]
            if _m(stem) > 1 and stem and stem[-1] in "st":
                w = stem
    # Step 5a
    if w.endswith("e"):
        stem = w[:-1]
        m = _m(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem
    # Step 5b
    if _m(w) > 1 and _double_cons(w) and w.endswith("l"):
        w = w[:-1]
    return w


@lru_cache(maxsize=1 << 17)
def stem(word: str) -> str:
    """Porter stem of a lower-case *word*.  Words with digits/underscores are kept."""
    if not word.isalpha() or not word.isascii():
        return word
    return _porter(word)
