"""Name normalization for the evaluation harness: the join keys and version qualifiers.

Station-neutral. These rules say how two spellings of an artist, album or title are
compared, and which bracketed clauses name a different recording. They read no flowsheet,
database or station config, so a station's ``plays.jsonl`` producer (WXYC's is
:mod:`evaluation.corpus`) and the scorer can share one definition instead of drifting copies.
"""

from __future__ import annotations

import re
import unicodedata

# A featuring or edition clause. The edition phrases of SAME_RECORDING that end in "version"
# go wherever they sit in the clause ("2011 Deluxe Version"); the other edition words only
# when they lead it.
_CRUFT = re.compile(
    r"\s*[\(\[](?:(?:feat|ft|featuring|with|deluxe|remaster(?:ed)?|expanded|anniversary|bonus)\b"
    r"|[^\)\]]*?\b(?:deluxe|expanded|anniversary|bonus track|special) version\b)[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
# A dotted initialism: two or more single letters, each followed by a period but the last,
# whose period is optional. A letter inside a longer word, or beside a digit, is no initial;
# an underscore is a separator, as in the keys, so it is no part of a word here.
_INITIALISM = re.compile(r"(?<![^\W_])[^\W\d_](?:\.[^\W\d_])+(?![^\W_])")
_FEATURING = frozenset({"feat", "ft", "featuring", "with"})
_BRACKETED = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}")
# A bracketed clause with one of these whole words, or its plural or past form, names a
# different recording: a version clause, which the keys keep and every tier honors.
# "version" counts only alone: "(Live Version)" names what "(Live)" names.
VERSION_QUALIFIERS = frozenset(
    "live remix mix demo edit version acoustic instrumental session rehearsal".split()
)
# Phrases naming the same recording, removed from a clause before qualifiers are looked for.
# Phrases come before the bare words, so "clean version" goes whole.
SAME_RECORDING = re.compile(
    r"\b(?:(?:radio|single|fcc|clean) edit"
    r"|(?:album|single|mono|stereo|lp|clean|explicit|radio|original|edited|remaster(?:ed)?"
    r"|deluxe|expanded|anniversary|bonus track|special) version|(?:original|mono|stereo) mix"
    r"|remaster(?:ed)?|mono|stereo|clean|explicit|edited)\b"
)


def _stem(word: str) -> str:
    """``word`` less a plural or past suffix: "remixes" and "remixed" are "remix"."""
    return re.sub(r"(?:e?s|ed)$", "", word)


def _undotted(text: str) -> str:
    """``text`` with each dotted initialism ("R.E.M.", "L.P.") as one word ("REM", "LP")."""
    return _INITIALISM.sub(lambda m: m.group().replace(".", ""), text)


def _words(text: str) -> list[str]:
    """``text``'s words, split as the key splits (dotted initialisms collapsed first), less
    same-recording phrases and a generic "version" beside another qualifier, so "(L.P.
    Version)" is the phrase "LP version"."""
    words = SAME_RECORDING.sub(" ", " ".join(re.split(r"[\W_]+", _undotted(text)))).split()
    if {_stem(w) for w in words} & (VERSION_QUALIFIERS - {"version"}):
        return [w for w in words if _stem(w) != "version"]
    return words


def _version_words(clause: str) -> list[str]:
    """A version clause's :func:`_words`; [] for any other clause, a featuring one included."""
    words = _words(clause)
    if words[:1] and words[0] in _FEATURING:
        return []
    return words if any(_stem(w) in VERSION_QUALIFIERS for w in words) else []


def fold(s: str | None) -> str:
    """NFKD, strip combining marks, casefold; punctuation kept (the exact-tier key)."""
    decomposed = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def album_key(s: str | None) -> str:
    """Lowercase, drop featuring and edition cruft, collapse whitespace.

    A featuring clause ("feat.", "ft.", "featuring", "with") always goes. An edition clause
    (one led by "Deluxe", "Remastered", ..., or holding a "Deluxe", "Expanded",
    "Anniversary", "Bonus Track" or "Special" "Version" phrase) stays only when it names
    another recording by more than the generic "version": "(Deluxe Version)", "(2011 Deluxe
    Version)" and "(Remastered 2011 Version)" go, "(Bonus Live Track)" stays. Only ASCII
    ``(...)`` and ``[...]`` clauses are seen: the fuzzy keys fold full-width brackets first.
    """

    def drop_unless_version(m: re.Match[str]) -> str:
        stems = {_stem(w) for w in _version_words(m.group())}
        return m.group() if stems & (VERSION_QUALIFIERS - {"version"}) else ""

    return " ".join(_CRUFT.sub(drop_unless_version, (s or "").lower()).split())


def fuzzy(s: str | None) -> str:
    """The fuzzy-tier key: ``album_key(fold(s))`` less bracketed clauses, non-word runs as a space.

    A dotted initialism, two or more single letters each followed by a period (the last
    period optional), is one word first: "R.E.M." and "R.E.M" key as "REM" does, and
    "A.R. Kane" as "AR Kane". A single initial ("J. Mascis") and an abbreviation with a
    longer word ("St. Vincent") are untouched, and the exact keys never join such a pair.

    ``(...)``, ``[...]`` and ``{...}`` clauses are dropped after NFKD, which folds
    full-width brackets to ASCII, so "The Worm" joins a tag "The Worm（ザ・ワーム）".
    A version clause is kept, its words less :data:`SAME_RECORDING` phrases and a generic
    "version" in the key, so "Back, Baby (Live)" never joins the studio "Back, Baby" and
    "(Live Version)" keys as "(Live)". A qualifier is keyed by its stem, so "(Remixed)"
    keys as "(Remix)". Letters and digits of every script outside
    brackets survive (a Japanese or Cyrillic name keeps a real key); a name that is all
    brackets keys to "" and never joins on this tier. Underscores and punctuation separate.
    """

    def drop_unless_version(m: re.Match[str]) -> str:
        words = (
            _stem(w) if _stem(w) in VERSION_QUALIFIERS else w for w in _version_words(m.group())
        )
        return f" {' '.join(words)} "

    undotted = _undotted(album_key(fold(s)))
    return " ".join(re.sub(r"[\W_]+", " ", _BRACKETED.sub(drop_unless_version, undotted)).split())


def qualifiers(*names: str | None) -> frozenset[str]:
    """The stemmed qualifiers in the version clauses of ``names``: the recording a play names."""
    clauses = [c for s in names for c in _BRACKETED.findall(album_key(fold(s)))]
    return frozenset(_stem(w) for c in clauses for w in _version_words(c)) & VERSION_QUALIFIERS


def named_qualifiers(album: str | None, title: str | None) -> frozenset[str]:
    """The stemmed qualifiers a pool file's album and title name: what a pool file has.

    :func:`qualifiers` plus the qualifier words outside brackets: an album's anywhere
    ("Live at KEXP"), a title's only after a spaced hyphen, en dash or em dash ("Back, Baby
    - Live"), so a title like "Live Forever" names nothing.
    """

    def outside(s: str | None) -> str:
        return _BRACKETED.sub(" ", album_key(fold(s)))

    suffix = re.split(r" [-\u2010-\u2015] ", outside(title), maxsplit=1)[1:]
    words = _words(outside(album)) + _words(" ".join(suffix))
    return qualifiers(album, title) | (frozenset(_stem(w) for w in words) & VERSION_QUALIFIERS)


def title_keys(artist: str | None, title: str | None) -> list[tuple[str, tuple[str, str]]]:
    """The title-tier join keys of one recording or play, as ``(kind, key)`` in kind order.

    ``exact`` is ``(fold(artist), album_key(title))`` and ``fuzzy`` is ``(fuzzy(artist),
    fuzzy(title))``. A key with an empty part (a missing tag, or a name that normalizes to
    nothing) is left out: it never joins.
    """
    candidates = [
        ("exact", (fold(artist), album_key(title))),
        ("fuzzy", (fuzzy(artist), fuzzy(title))),
    ]
    return [(kind, key) for kind, key in candidates if all(key)]


def title_tier(
    play_artist: str | None,
    play_album: str | None,
    play_title: str | None,
    artists: tuple[str | None, ...],
    album: str | None,
    title: str | None,
) -> str | None:
    """``exact``, ``fuzzy``, or None: whether a play's title keys meet one recording's.

    The recording is named by ``album``, ``title`` and any of ``artists`` (a pool file passes
    its artist and album artist, an emission its one artist). The play and the recording
    join when their title keys are equal and the recording names each version qualifier
    the play names in a bracketed clause of its album or title (:func:`qualifiers` against
    :func:`named_qualifiers`), so a play logged "(Live)" never meets the studio recording.

    ``exact`` and ``fuzzy`` name the title keys that met. They are not ``plays.jsonl``'s
    ``pool_match_tier``, where ``exact`` and ``fuzzy`` are album tiers and every join on the
    title keys is ``title``. ``artists`` is a tuple of names, never a bare string, which
    would be read a letter at a time: mypy rejects it and a caller that evades mypy gets ``TypeError``.
    """
    if isinstance(artists, str):
        raise TypeError("artists must be a tuple of names, not a str")
    if not qualifiers(play_album, play_title) <= named_qualifiers(album, title):
        return None
    have = {key for artist in artists for key in title_keys(artist, title)}
    return next(
        (kind for kind, key in title_keys(play_artist, play_title) if (kind, key) in have), None
    )
