"""Tests for evaluation.names: the join keys and version qualifiers."""

from __future__ import annotations

import pytest

from evaluation import names as norm


@pytest.mark.parametrize(
    ("s", "folded", "album_key", "fuzzy"),
    [
        ("Hermanos Gutiérrez", "hermanos gutierrez", "hermanos gutiérrez", "hermanos gutierrez"),
        ("Nilüfer Yanya", "nilufer yanya", "nilüfer yanya", "nilufer yanya"),
        ("DOGA (Deluxe Edition)", "doga (deluxe edition)", "doga", "doga"),
        ("Edits [Remastered]", "edits [remastered]", "edits", "edits"),
        (
            "Call  Your Name (feat. Someone)",
            "call  your name (feat. someone)",
            "call your name",
            "call your name",
        ),
        ("Back, Baby", "back, baby", "back, baby", "back baby"),
        # Cruft words match whole words only: the exact key keeps "(Feathers)", and
        # only the fuzzy key, which drops every bracketed clause, loses it.
        ("Ghost (Feathers)", "ghost (feathers)", "ghost (feathers)", "ghost"),
        ("Halo (ft. Someone)", "halo (ft. someone)", "halo", "halo"),
        (None, "", "", ""),
        # The fuzzy key drops bracketed clauses after NFKD, full-width ones included.
        ("The Worm（ザ・ワーム）", "the worm(サ・ワーム)", "the worm（ザ・ワーム）", "the worm"),
        # A clause naming a different recording is kept, as words; the others still go.
        (
            "Edits {Live} [Tape] (Demo)",
            "edits {live} [tape] (demo)",
            "edits {live} [tape] (demo)",
            "edits live demo",
        ),
        ("Back, Baby (Live)", "back, baby (live)", "back, baby (live)", "back baby live"),
        ("Back, Baby [Demo]", "back, baby [demo]", "back, baby [demo]", "back baby demo"),
        (
            "Back, Baby（Instrumental）",
            "back, baby(instrumental)",
            "back, baby（instrumental）",
            "back baby instrumental",
        ),
        ("Halo (Live at KEXP)", "halo (live at kexp)", "halo (live at kexp)", "halo live at kexp"),
        # Cruft runs first, so the "(ft. ...)" clause is gone before qualifiers are looked at.
        (
            "Halo (Live) (ft. Someone)",
            "halo (live) (ft. someone)",
            "halo (live)",
            "halo live",
        ),
        # Words split as the key splits them, underscores included.
        (
            "Back, Baby (Live_Session)",
            "back, baby (live_session)",
            "back, baby (live_session)",
            "back baby live session",
        ),
        # A version clause that opens with a cruft word is kept by both keys; the fuzzy
        # key leaves out its same-recording phrases.
        (
            "Back, Baby (Bonus Live Track)",
            "back, baby (bonus live track)",
            "back, baby (bonus live track)",
            "back baby bonus live track",
        ),
        (
            "Back, Baby (Remastered Live Version)",
            "back, baby (remastered live version)",
            "back, baby (remastered live version)",
            "back baby live",
        ),
        # An edition clause whose only qualifier is the generic "version" is still cruft,
        # and a featuring clause always is, whatever words it holds.
        *(
            (f"Back, Baby ({clause})", f"back, baby ({clause.lower()})", "back, baby", "back baby")
            for clause in (
                "Deluxe Version",
                "Bonus Track Version",
                "Expanded Version",
                "Anniversary Version",
                "Remastered 2011 Version",
                "feat. Mix Master Mike",
                "ft. Live Skull",
            )
        ),
        (
            "Back, Baby（feat. Mix Master Mike）",
            "back, baby(feat. mix master mike)",
            "back, baby（feat. mix master mike）",
            "back baby",
        ),
        # An edition phrase that does not lead the clause is cruft on the exact key too.
        *(
            (f"Back, Baby ({clause})", f"back, baby ({clause.lower()})", "back, baby", "back baby")
            for clause in (
                "2011 Deluxe Version",
                "Super Deluxe Version",
                "Special Version",
                "2011 Expanded Version",
                "25th Anniversary Version",
                "Japan Bonus Track Version",
            )
        ),
        # A "with" credit is a featuring clause on every key, whatever words it holds.
        (
            "Back, Baby (with Live Skull)",
            "back, baby (with live skull)",
            "back, baby",
            "back baby",
        ),
        # A full-width bracket is folded only by the fuzzy key, so the exact key keeps the
        # clause; the cruft rule runs after the fold there, as the fuzzy key's must.
        (
            "Back, Baby（Deluxe Version）",
            "back, baby(deluxe version)",
            "back, baby（deluxe version）",
            "back baby",
        ),
        *(
            (
                f"Back, Baby{wide[0]}{clause}{wide[1]}",
                f"back, baby{narrow[0]}{clause.lower()}{narrow[1]}",
                f"back, baby{wide[0]}{clause.lower()}{wide[1]}",
                "back baby",
            )
            for wide, narrow in (("（）", "()"), ("［］", "[]"))
            for clause in ("Remastered 2011 Version", "Deluxe Edition Version", "Bonus 2CD Version")
        ),
        # A qualifier is keyed by its stem, so its plural and past forms join it.
        *(
            (
                f"Back, Baby ({word})",
                f"back, baby ({word.lower()})",
                f"back, baby ({word.lower()})",
                key,
            )
            for word, key in (
                ("Remixed", "back baby remix"),
                ("Peel Sessions", "back baby peel session"),
            )
        ),
        # "Version" beside another qualifier adds nothing: "(Live Version)" keys as "(Live)".
        (
            "Back, Baby (Live Version)",
            "back, baby (live version)",
            "back, baby (live version)",
            "back baby live",
        ),
        (
            "Back, Baby (Alternate Version)",
            "back, baby (alternate version)",
            "back, baby (alternate version)",
            "back baby alternate version",
        ),
        # A clause naming the same recording is no version clause.
        (
            "Back, Baby (Radio Edit)",
            "back, baby (radio edit)",
            "back, baby (radio edit)",
            "back baby",
        ),
        # The qualifier is a whole word, and clauses that merely contain its letters go.
        ("Halo (Delivered)", "halo (delivered)", "halo (delivered)", "halo"),
        ("Halo (Editor's Note)", "halo (editor's note)", "halo (editor's note)", "halo"),
        # An entirely bracketed name has no fuzzy key, so it never joins on that tier.
        ("（ザ・ワーム）", "(サ・ワーム)", "（ザ・ワーム）", ""),
        # Letters and digits of any script survive the fuzzy key; diacritics still fold.
        ("ザ・ワーム", "サ・ワーム", "ザ・ワーム", "サ ワーム"),
        ("Хуана Молина", "хуана молина", "хуана молина", "хуана молина"),
        ("Csillagrablók_2", "csillagrablok_2", "csillagrablók_2", "csillagrablok 2"),
    ],
)
def test_normalizers(s: str | None, folded: str, album_key: str, fuzzy: str) -> None:
    assert norm.fold(s) == folded
    assert norm.album_key(s) == album_key
    assert norm.fuzzy(s) == fuzzy


DASHES = ["-", *(chr(c) for c in range(0x2010, 0x2016))]
ASCII_AND_WIDE = [
    tuple(p) for p in ("()", "[]", "{}", "\uff08\uff09", "\uff3b\uff3d", "\uff5b\uff5d")
]
CJK = [tuple(p) for p in ("\u3010\u3011", "\u300c\u300d", "\u300e\u300f", "\u3014\u3015")]


@pytest.mark.parametrize(
    ("s", "album_key", "fuzzy"),
    [
        # An unbracketed featuring credit goes as the bracketed one does: the word "feat.",
        # or "featuring" (any case) and the words after it. "ft." is deliberately not handled inline.
        ("Hibiscus Feat. Bbyafricka", "hibiscus", "hibiscus"),
        ("Hibiscus (feat Bbyafricka)", "hibiscus", "hibiscus"),
        ("Hibiscus Featuring Bbyafricka", "hibiscus", "hibiscus"),
        ("Hibiscus  feat.  Bbyafricka  &  Juana Molina", "hibiscus", "hibiscus"),
        ("Hibiscus\tfeat.\nBbyafricka", "hibiscus", "hibiscus"),
        ("Hibiscus feat. R.E.M.", "hibiscus", "hibiscus"),
        ("Juana Molina feat. Jessica Pratt", "juana molina", "juana molina"),
        ("Song 2 feat. Jessica Pratt", "song 2", "song 2"),
        ("Hibiscus feat. Jay-Z", "hibiscus", "hibiscus"),
        # A credit needs a period ("feat" alone is a word), a word after it, and text before
        # it. "ft." is no inline credit at all: it is feet and Fort as often as featuring.
        ("Hibiscus ft. Bbyafricka", "hibiscus ft. bbyafricka", "hibiscus ft bbyafricka"),
        ("Hibiscus FT. Bbyafricka", "hibiscus ft. bbyafricka", "hibiscus ft bbyafricka"),
        ("Hibiscus feat Bbyafricka", "hibiscus feat bbyafricka", "hibiscus feat bbyafricka"),
        ("Hibiscus FT Bbyafricka", "hibiscus ft bbyafricka", "hibiscus ft bbyafricka"),
        ("Hibiscus feat", "hibiscus feat", "hibiscus feat"),
        ("Hibiscus feat.", "hibiscus feat.", "hibiscus feat"),
        ("Hibiscus featuring", "hibiscus featuring", "hibiscus featuring"),
        ("Hibiscus feat. (Live)", "hibiscus feat. (live)", "hibiscus feat live"),
        ("Little Feat", "little feat", "little feat"),
        ("No Mean Feat", "no mean feat", "no mean feat"),
        ("A Feat of Clay", "a feat of clay", "a feat of clay"),
        ("Six Ft. Under", "six ft. under", "six ft under"),
        ("50 Ft. Queenie", "50 ft. queenie", "50 ft queenie"),
        ("10 Ft. Ganja Plant", "10 ft. ganja plant", "10 ft ganja plant"),
        ("Live at Ft. Worth", "live at ft. worth", "live at ft worth"),
        ("Dancing with Myself", "dancing with myself", "dancing with myself"),
        ("Waltz with Bashir", "waltz with bashir", "waltz with bashir"),
        ("Featuring Ourselves", "featuring ourselves", "featuring ourselves"),
        ("Feat. Ourselves", "feat. ourselves", "feat ourselves"),
        ("Theft", "theft", "theft"),
        ("Feather", "feather", "feather"),
        ("Lift", "lift", "lift"),
        ("Ghost Feather Boa", "ghost feather boa", "ghost feather boa"),
        # The credit never swallows a version qualifier that follows it: a bracketed clause
        # of any kind or a spaced-dash suffix is kept (and read as one), the other clauses go
        # as before. So does anything after a comma, semicolon, colon or slash.
        ("Hibiscus feat. Bbyafricka -Live", "hibiscus -live", "hibiscus live"),
        ("Hibiscus feat. Bbyafricka - Y", "hibiscus - y", "hibiscus y"),
        ("Hibiscus feat. Bbyafricka- Y", "hibiscus", "hibiscus"),
        ("Hibiscus feat. Jay-Z - Y", "hibiscus - y", "hibiscus y"),
        ("Hibiscus feat. Bbyafricka, Y", "hibiscus, y", "hibiscus y"),
        ("Hibiscus feat. Bbyafricka; Y", "hibiscus; y", "hibiscus y"),
        ("Hibiscus feat. Bbyafricka: Y", "hibiscus: y", "hibiscus y"),
        ("Hibiscus feat. Bbyafricka / Y", "hibiscus / y", "hibiscus y"),
        # ...and so does a version word, plural and past forms included, wherever it sits.
        ("Hibiscus feat. Bbyafricka Remix", "hibiscus remix", "hibiscus remix"),
        ("Hibiscus Featuring Bbyafricka Remixes", "hibiscus remixes", "hibiscus remixes"),
        ("Hibiscus feat. Bbyafricka Remixed", "hibiscus remixed", "hibiscus remixed"),
        (
            "Hibiscus feat. Bbyafricka Live Version",
            "hibiscus live version",
            "hibiscus live version",
        ),
        (
            "Hibiscus feat. Bbyafricka, Live at KEXP",
            "hibiscus, live at kexp",
            "hibiscus live at kexp",
        ),
        (
            "Hibiscus feat. Bbyafricka: Live in Tokyo",
            "hibiscus: live in tokyo",
            "hibiscus live in tokyo",
        ),
        ("Hibiscus feat. Bbyafricka / Demo", "hibiscus / demo", "hibiscus demo"),
        ("Hibiscus feat. Bbyafricka \u2212 Remix", "hibiscus \u2212 remix", "hibiscus remix"),
        ("Hibiscus feat. Live Skull", "hibiscus feat. live skull", "hibiscus feat live skull"),
        (
            "Hibiscus feat. Mix Master Mike",
            "hibiscus feat. mix master mike",
            "hibiscus feat mix master mike",
        ),
        # A leading space is no text before the word, so a leading credit word stays a word.
        (" Featuring Ourselves", "featuring ourselves", "featuring ourselves"),
        ("Hibiscus [Live feat. Bbyafricka]", "hibiscus [live]", "hibiscus live"),
        # Every bracket kind stops a credit and closes one, spaced from it or not: the exact
        # keys see full-width and CJK brackets unfolded.
        *(
            (f"Hibiscus feat. Bbyafricka{o}Y{c}", f"hibiscus {o}y{c}", "hibiscus")
            for o, c in ASCII_AND_WIDE
        ),
        *((f"Hibiscus feat. Bbyafricka{o}Y{c}", f"hibiscus {o}y{c}", "hibiscus y") for o, c in CJK),
        *(
            (f"Hibiscus {o}Y feat. Bbyafricka{c}", f"hibiscus {o}y{c}", "hibiscus")
            for o, c in ASCII_AND_WIDE
        ),
        *(
            (f"Hibiscus {o}Y feat. Bbyafricka{c}", f"hibiscus {o}y{c}", "hibiscus y")
            for o, c in CJK
        ),
        *((f"Hibiscus feat. Bbyafricka {d} Y", f"hibiscus {d} y", "hibiscus y") for d in DASHES),
        ("Hibiscus feat. Bbyafricka (Live)", "hibiscus (live)", "hibiscus live"),
        ("Hibiscus feat. Bbyafricka [Demo]", "hibiscus [demo]", "hibiscus demo"),
        ("Hibiscus feat. Bbyafricka {Live}", "hibiscus {live}", "hibiscus live"),
        ("Hibiscus feat. Bbyafricka（Live）", "hibiscus （live）", "hibiscus live"),
        ("Hibiscus feat. Bbyafricka［Demo］", "hibiscus ［demo］", "hibiscus demo"),
        ("Hibiscus feat. Bbyafricka｛Live｝", "hibiscus ｛live｝", "hibiscus live"),
        *(
            (f"Hibiscus feat. Bbyafricka {d} Live", f"hibiscus {d} live", "hibiscus live")
            for d in DASHES
        ),
        (
            "Hibiscus feat. Bbyafricka (Live) (Bonus Track Version)",
            "hibiscus (live)",
            "hibiscus live",
        ),
        ("Hibiscus feat. Bbyafricka (Deluxe Version)", "hibiscus", "hibiscus"),
        ("Hibiscus (Live) feat. Bbyafricka", "hibiscus (live)", "hibiscus live"),
        ("Hibiscus feat. Bbyafricka (Brazil)", "hibiscus (brazil)", "hibiscus"),
        # A credit inside a bracketed clause stops at its closing bracket, so the clause
        # is still one: main keys these the same on the fuzzy key.
        ("Hibiscus (Radio Edit feat. Bbyafricka)", "hibiscus (radio edit)", "hibiscus"),
        ("Hibiscus (Live feat. Bbyafricka)", "hibiscus (live)", "hibiscus live"),
        (
            "Hibiscus [Live ft. Bbyafricka]",
            "hibiscus [live ft. bbyafricka]",
            "hibiscus live ft bbyafricka",
        ),
        ("Hibiscus {Live featuring Bbyafricka}", "hibiscus {live}", "hibiscus live"),
        # The edition rule judges a clause before the credit is cut from it: "special version"
        # is no edition phrase until the credit is gone, so the clause is kept whole and then
        # loses the credit, on the exact key.
        ("Hibiscus (Special feat. Bbyafricka Version)", "hibiscus (special version)", "hibiscus"),
        ("Hibiscus (Live feat. Bbyafricka) (Demo)", "hibiscus (live) (demo)", "hibiscus live demo"),
        ("Hibiscus（Live feat. Bbyafricka）", "hibiscus（live）", "hibiscus live"),
    ],
)
def test_an_unbracketed_featuring_credit_is_dropped_as_the_bracketed_one_is(
    s: str, album_key: str, fuzzy: str
) -> None:
    assert norm.album_key(s) == album_key
    assert norm.fuzzy(s) == fuzzy


@pytest.mark.parametrize(
    ("play_title", "pool_title", "tier"),
    [
        # The Carré case: the flowsheet's bracketed credit meets the tag's inline one.
        ("Hibiscus (feat Bbyafricka)", "Hibiscus Feat. Bbyafricka", "exact"),
        ("Hibiscus Feat. Bbyafricka", "Hibiscus (feat Bbyafricka)", "exact"),
        ("Hibiscus", "Hibiscus ft. Bbyafricka", None),
        ("Hibiscus featuring Bbyafricka", "Hibiscus", "exact"),
        ("Hibiscus Feat. Bbyafricka!", "Hibiscus", "exact"),
        ("Dancing with Myself", "Dancing", None),
        ("Six Ft. Under", "Six Ft. Deep", None),
        ("Little Feat", "Little", None),
        # A version the credit precedes is still named, on the play side and the pool side.
        ("Hibiscus feat. Bbyafricka (Live)", "Hibiscus (Live)", "exact"),
        ("Hibiscus feat. Bbyafricka (Live)", "Hibiscus", None),
        ("Hibiscus feat. Bbyafricka (Live)", "Hibiscus feat. Bbyafricka", None),
        ("Hibiscus feat. Bbyafricka (Live)", "Hibiscus featuring Someone Else (Live)", "exact"),
        ("Hibiscus (Live)", "Hibiscus feat. Bbyafricka (Live)", "exact"),
        ("Hibiscus (Live)", "Hibiscus feat. Bbyafricka", None),
        ("Hibiscus (Live)", "Hibiscus feat. Bbyafricka - Live", "fuzzy"),
        ("Hibiscus feat. Bbyafricka - Live", "Hibiscus (Live)", "fuzzy"),
        ("Hibiscus feat. Bbyafricka (Demo)", "Hibiscus feat. Bbyafricka (Live)", None),
        # A credit never swallows a version word: none of these joins the plain title.
        ("Hibiscus", "Hibiscus feat. Bbyafricka Remix", None),
        ("Hibiscus", "Hibiscus feat. Bbyafricka, Live at KEXP", None),
        ("Hibiscus", "Hibiscus feat. Bbyafricka\u3010Live\u3011", None),
        ("Hibiscus", "Hibiscus feat. Bbyafricka: Live in Tokyo", None),
        ("Hibiscus", "Hibiscus feat. Bbyafricka -Live", None),
        ("Hibiscus feat. Bbyafricka Remix", "Hibiscus", None),
        ("Hibiscus feat. Bbyafricka", "Hibiscus", "exact"),
        ("Hibiscus", "Hibiscus feat. Bbyafricka", "exact"),
        # ...also in full-width brackets, whose exact key is not folded.
        ("Hibiscus", "Hibiscus feat. Bbyafricka（Live）", None),
        ("Hibiscus feat. Bbyafricka（Live）", "Hibiscus", None),
        ("Hibiscus feat. Bbyafricka（Live）", "Hibiscus（Live）", "fuzzy"),
        # A credit inside a version clause leaves the clause, and the qualifier, in place.
        ("Hibiscus (Radio Edit feat. Bbyafricka)", "Hibiscus", "fuzzy"),
        ("Hibiscus", "Hibiscus (Radio Edit feat. Bbyafricka)", "fuzzy"),
        ("Hibiscus (Live feat. Bbyafricka)", "Hibiscus", None),
        ("Hibiscus", "Hibiscus (Live feat. Bbyafricka)", None),
        ("Hibiscus (Live feat. Bbyafricka)", "Hibiscus (Live)", "exact"),
    ],
)
def test_title_tier_joins_an_unbracketed_featuring_credit(
    play_title: str, pool_title: str, tier: str | None
) -> None:
    assert norm.title_tier("Carré", "", play_title, ("Carré",), "", pool_title) == tier


@pytest.mark.parametrize(
    ("s", "fuzzy"),
    [
        # A dotted initialism (two or more single letters each followed by a period, the
        # last period optional) keys as the same letters undotted.
        ("R.E.M.", "rem"),
        ("R.E.M", "rem"),
        ("REM", "rem"),
        ("A.R. Kane", "ar kane"),
        ("AR Kane", "ar kane"),
        ("U.S. Girls", "us girls"),
        ("L.A. Witch", "la witch"),
        ("The L.A. Witch Tapes", "the la witch tapes"),
        ("S.G. Goodman (Live)", "sg goodman live"),
        ("A.B.C.D.", "abcd"),
        ("Ñ.Ö.", "no"),
        # A single initial is no initialism.
        ("J. Mascis", "j mascis"),
        ("J.Mascis", "j mascis"),
        # An abbreviation with a longer word, a number, or a decimal is untouched.
        ("St. Vincent", "st vincent"),
        ("Mr. Twin Sister", "mr twin sister"),
        ("Vol. 2", "vol 2"),
        ("No. 1", "no 1"),
        ("Disc 1.5", "disc 1 5"),
        ("v1.2", "v1 2"),
        ("x.com", "x com"),
        ("a.k.a.mix", "aka mix"),
        ("a.k.a", "aka"),
        # A chain after a period that follows a digit still collapses: it is single letters.
        ("1.A.B", "1 ab"),
        # An underscore separates, so it never hides an initialism or joins one to a word.
        ("_R.E.M.", "rem"),
        ("R.E.M._", "rem"),
        ("01_R.E.M.", "01 rem"),
        # A letter that ends a longer word is no initial: the chain starts at a boundary.
        ("xa.b.c", "xa bc"),
        ("Vol.A.B", "vol ab"),
        # Letters of any script survive fold and still collapse.
        ("Д.Д.Т.", "ддт"),
        ("Α.Β.", "αβ"),
        # Initials with a space between them are separate initials, as before.
        ("J. M. Barrie", "j m barrie"),
    ],
)
def test_fuzzy_joins_a_dotted_initialism_to_the_same_letters(s: str, fuzzy: str) -> None:
    assert norm.fuzzy(s) == fuzzy


@pytest.mark.parametrize("s", ["R.E.M.", "A.R. Kane", "U.S. Girls"])
def test_the_exact_keys_keep_an_initialism_dotted(s: str) -> None:
    undotted = s.replace(".", "")
    assert norm.fold(s) != norm.fold(undotted)
    assert norm.album_key(s) != norm.album_key(undotted)
    assert norm.fuzzy(s) == norm.fuzzy(undotted)


@pytest.mark.parametrize(
    ("names", "play", "pool"),
    [
        # The play side reads bracketed clauses only; the pool side reads outside them too.
        (("Live at KEXP", "Back, Baby"), set(), {"live"}),
        (("Bootleg", "Back, Baby - Live"), set(), {"live"}),
        (("Bootleg", "Back, Baby \u2013 Live"), set(), {"live"}),
        (("Bootleg", "Back, Baby \u2014 Live"), set(), {"live"}),
        # A dash without spaces separates no suffix.
        (("Bootleg", "Back, Baby-Live"), set(), set()),
        # A title's unbracketed words count only after " - "; an album's count anywhere.
        (("Bootleg", "Live Forever"), set(), set()),
        (("Bootleg", "Session 9 - Demo"), set(), {"demo"}),
        (("Session 9", "Back, Baby"), set(), {"session"}),
        (("Bootleg", "Back, Baby (Live Version)"), {"live"}, {"live"}),
        (("Bootleg", "Back, Baby (Alternate Version)"), {"version"}, {"version"}),
        # A clause's lone "version" is judged within the clause, not beside the name's words.
        (("Live at KEXP", "Back, Baby (Alternate Version)"), {"version"}, {"live", "version"}),
        (("Bootleg", "Back, Baby (Remixes) [Demo]"), {"remix", "demo"}, {"remix", "demo"}),
        (("Bootleg", "Back, Baby (feat. Mix Master Mike)"), set(), set()),
        (("Bootleg", "Back, Baby (Edited)"), set(), set()),
        (("Bootleg", "Back, Baby - Radio Edit"), set(), set()),
        # Every edition phrase of SAME_RECORDING names no version, outside brackets as well.
        *(
            ((f"Bootleg {phrase}", "Back, Baby"), set(), set())
            for phrase in (
                "Deluxe Version",
                "2011 Expanded Version",
                "25th Anniversary Version",
                "Japan Bonus Track Version",
                "Special Version",
            )
        ),
        (("Bootleg", "Back, Baby {2011 Expanded Version}"), set(), set()),
        # A credit closed by another bracket kind goes whole, full-width or not.
        (("Bootleg", "Back, Baby (feat. Live Skull]"), set(), set()),
        (("Bootleg", "Back, Baby（feat. Live Skull]"), set(), set()),
        (("Bootleg（feat. Live Skull]", "Back, Baby"), set(), set()),
        (("Bootleg", "Back, Baby - Demo（feat. Live Skull]"), set(), {"demo"}),
    ],
)
def test_qualifiers(names: tuple[str, str], play: set[str], pool: set[str]) -> None:
    assert norm.qualifiers(*names) == play
    assert norm.named_qualifiers(*names) == pool


@pytest.mark.parametrize(
    ("artist", "title", "keys"),
    [
        (
            "Jessica Pratt",
            "Back, Baby",
            [("exact", ("jessica pratt", "back, baby")), ("fuzzy", ("jessica pratt", "back baby"))],
        ),
        # Cruft goes on the exact key too; a diacritic folds on both.
        (
            "Hermanos Gutiérrez",
            "Sol (feat. Someone)",
            [("exact", ("hermanos gutierrez", "sol")), ("fuzzy", ("hermanos gutierrez", "sol"))],
        ),
        # A key with an empty part is left out, and only that tier's.
        ("Juana Molina", "（ザ・ワーム）", [("exact", ("juana molina", "（ザ・ワーム）"))]),
        (
            "Juana Molina",
            "(Live)",
            [("exact", ("juana molina", "(live)")), ("fuzzy", ("juana molina", "live"))],
        ),
        ("Juana Molina", "", []),
        ("", "la paradoja", []),
        (None, None, []),
    ],
)
def test_title_keys(artist: str | None, title: str | None, keys: list[tuple[str, tuple[str, str]]]):
    assert norm.title_keys(artist, title) == keys


@pytest.mark.parametrize(
    ("artist", "keys"),
    [
        ("Cass McCombs", ["cass mccombs"]),
        ("", []),
        (None, []),
        # A co-credit also keys as its sorted names, so the order of the names is lost.
        (
            "Cass McCombs & Chris Cohen",
            ["cass mccombs chris cohen", "cass mccombs|chris cohen"],
        ),
        (
            "Chris Cohen, Cass McCombs",
            ["chris cohen cass mccombs", "cass mccombs|chris cohen"],
        ),
        # Every separator reads alike, and a bracketed clause is no part of a name.
        (
            "Cass McCombs and Chris Cohen",
            ["cass mccombs and chris cohen", "cass mccombs|chris cohen"],
        ),
        (
            "Cass McCombs (solo, live) & Chris Cohen",
            ["cass mccombs solo live chris cohen", "cass mccombs solo live|chris cohen"],
        ),
        # Three names, and a repeated one stays twice.
        (
            "Chuquimamani-Condori; Juana Molina / Jessica Pratt",
            [
                "chuquimamani condori juana molina jessica pratt",
                "chuquimamani condori|jessica pratt|juana molina",
            ],
        ),
        ("Juana Molina & Juana Molina", ["juana molina juana molina", "juana molina|juana molina"]),
        # A name that is no co-credit is not split: "x" inside or at the edge of a name,
        # a separator with nothing on one side, a featuring credit, "with" in a bracket.
        ("Charli XCX", ["charli xcx"]),
        ("Lil Nas X", ["lil nas x"]),
        ("X Ambassadors", ["x ambassadors"]),
        ("Cass McCombs &", ["cass mccombs"]),
        ("Jessica Pratt feat. Juana Molina", ["jessica pratt"]),
        ("Jessica Pratt (with Juana Molina)", ["jessica pratt"]),
        ("Stereolab（ステレオラブ）", ["stereolab"]),
        # The split runs after the cruft rules: a credit swallows a separator after it.
        ("Jessica Pratt feat. Juana Molina & Cat Power", ["jessica pratt"]),
        ("Jessica Pratt featuring Juana Molina and Cat Power", ["jessica pratt"]),
        # A separator is inside a clause only when the clause is balanced: a name may end in
        # closers.
        ("Sunn O))) & Boris", ["sunn o boris", "boris|sunn o"]),
        ("Boris & Sunn O)))", ["boris sunn o", "boris|sunn o"]),
        ("Hibiscus (Juana Molina & Cat Power)", ["hibiscus"]),
        # A dotted initialism is one word on each side of a separator.
        ("R.E.M. + A.R. Kane", ["rem ar kane", "ar kane|rem"]),
    ],
)
def test_artist_keys(artist: str | None, keys: list[str]) -> None:
    assert norm.artist_keys(artist) == keys


@pytest.mark.parametrize(
    ("play", "recording", "tier"),
    [
        # The Steel Reserve case (WXYC/stream-sleuth#126): the same artists in another order
        # join on the fuzzy keys only; the exact keys keep the logged order.
        ("Cass McCombs & Chris Cohen", "Chris Cohen, Cass McCombs", "fuzzy"),
        ("Chris Cohen, Cass McCombs", "Cass McCombs & Chris Cohen", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Cass McCombs & Chris Cohen", "exact"),
        ("Cass McCombs & Chris Cohen", "Cass McCombs and Chris Cohen", "fuzzy"),
        # Separators read alike.
        ("Cass McCombs & Chris Cohen", "Chris Cohen and Cass McCombs", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Chris Cohen x Cass McCombs", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Chris Cohen / Cass McCombs", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Chris Cohen + Cass McCombs", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Chris Cohen with Cass McCombs", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Chris Cohen vs. Cass McCombs", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Chris Cohen vs Cass McCombs", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Chris Cohen; Cass McCombs", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Chris Cohen&Cass McCombs", "fuzzy"),
        # Case, diacritics, and a featuring credit on either side.
        ("Hermanos Gutiérrez & Juana Molina", "JUANA MOLINA, Hermanos Gutierrez", "fuzzy"),
        ("Cass McCombs & Chris Cohen", "Chris Cohen & Cass McCombs feat. Jessica Pratt", "fuzzy"),
        # A different set of names, a subset, or a superset never joins under this rule.
        ("Cass McCombs & Chris Cohen", "Chris Cohen & Jessica Pratt", None),
        ("Cass McCombs", "Cass McCombs & Chris Cohen", None),
        ("Cass McCombs & Chris Cohen", "Cass McCombs", None),
        ("Cass McCombs & Chris Cohen", "Chris Cohen & Cass McCombs & Jessica Pratt", None),
        ("Juana Molina & Juana Molina", "Juana Molina & Jessica Pratt", None),
        # A single name is never split: it joins itself, and only the reading of its two halves.
        ("Belle and Sebastian", "Belle and Sebastian", "exact"),
        ("Belle and Sebastian", "Belle & Sebastian", "fuzzy"),
        ("Belle and Sebastian", "Belle", None),
        ("Belle and Sebastian", "Sebastian", None),
        ("Simon & Garfunkel", "Garfunkel & Simon", "fuzzy"),
        # A name with "x" as part of it is no co-credit, so it joins nothing reordered.
        ("Charli XCX", "XCX Charli", None),
        ("Charli XCX", "Charli", None),
        ("Lil Nas X", "X Nas Lil", None),
        ("Juana Molina x Lil Nas X", "Lil Nas X & Juana Molina", "fuzzy"),
        # A joined literal reading keeps working: no separator on one side.
        ("Cass McCombs Chris Cohen", "Cass McCombs & Chris Cohen", "fuzzy"),
        # A last-name-first tag is not a reordered credit.
        ("Chris Cohen", "Cohen, Chris", None),
        # An unbalanced closer is part of a name, not a clause: both orders join.
        ("Sunn O))) & Boris", "Boris & Sunn O)))", "fuzzy"),
        ("Boris & Sunn O)))", "Sunn O))) and Boris", "fuzzy"),
        # A credit swallows a separator after it, so these name one artist and join no reorder.
        ("Cass McCombs feat. Chris Cohen & Jessica Pratt", "Jessica Pratt & Cass McCombs", None),
    ],
)
def test_title_tier_joins_a_co_credit_in_another_order(
    play: str, recording: str, tier: str | None
) -> None:
    assert norm.title_tier(play, "x", "Steel Reserve", (recording,), "y", "Steel Reserve") == tier


def test_title_tier_joins_a_co_credit_through_any_of_the_artist_tags() -> None:
    artists = ("Various Artists", "Chris Cohen, Cass McCombs")
    play = ("Cass McCombs & Chris Cohen", "x", "Steel Reserve")
    assert norm.title_tier(*play, artists, "y", "Steel Reserve") == "fuzzy"


@pytest.mark.parametrize(
    ("play", "recording", "tier"),
    [
        # The same spelling joins on the exact key; punctuation and case alone, on the fuzzy.
        (("Jessica Pratt", "x", "Back, Baby"), ("Jessica Pratt", "y", "Back, Baby"), "exact"),
        (("jessica pratt", "x", "back baby!"), ("Jessica Pratt", "y", "Back, Baby"), "fuzzy"),
        (("Jessica Pratt", "x", "Back, Baby"), ("Jessica Pratt", "y", "Other"), None),
        (("Jessica Pratt", "x", "Back, Baby"), ("Juana Molina", "y", "Back, Baby"), None),
        # Featuring and edition clauses drop out on either side.
        (
            ("Jessica Pratt", "x", "Back, Baby (feat. Someone)"),
            ("Jessica Pratt", "y", "Back, Baby"),
            "exact",
        ),
        (
            ("Jessica Pratt", "x", "Back, Baby"),
            ("Jessica Pratt", "y", "Back, Baby (Remastered 2011 Version)"),
            "exact",
        ),
        (
            ("Jessica Pratt", "Edits (Deluxe Version)", "Back, Baby"),
            ("Jessica Pratt", "Edits", "Back, Baby"),
            "exact",
        ),
        # A version the play names, in its title or its album, must be named by the recording.
        (("Jessica Pratt", "x", "Back, Baby (Live)"), ("Jessica Pratt", "y", "Back, Baby"), None),
        (
            ("Jessica Pratt", "Edits [Live]", "Back, Baby"),
            ("Jessica Pratt", "y", "Back, Baby"),
            None,
        ),
        (
            ("Jessica Pratt", "x", "Back, Baby (Live)"),
            ("Jessica Pratt", "y", "Back, Baby (Live)"),
            "exact",
        ),
        (
            ("Jessica Pratt", "x", "Back, Baby (Live)"),
            ("Jessica Pratt", "Live at KEXP", "Back, Baby"),
            None,
        ),
        (
            ("Jessica Pratt", "x", "Back, Baby (Live)"),
            ("Jessica Pratt", "Live at KEXP", "Back, Baby (Live)"),
            "exact",
        ),
        # ...but the recording may name more than the play does.
        (("Jessica Pratt", "x", "Back, Baby"), ("Jessica Pratt", "y", "Back, Baby (Live)"), None),
        # The fuzzy key stems a qualifier, so "(Remixed)" meets "(Remix)".
        (
            ("Jessica Pratt", "x", "Back, Baby (Remixed)"),
            ("Jessica Pratt", "y", "Back, Baby (Remix)"),
            "fuzzy",
        ),
        # A dotted initialism joins the same letters undotted, on the fuzzy key only.
        (("US Girls", "x", "Overtime"), ("U.S. Girls", "y", "Overtime"), "fuzzy"),
        (("A.R. Kane", "x", "Baby Milk Snatcher"), ("AR Kane", "y", "Baby Milk Snatcher"), "fuzzy"),
        (("J. Mascis", "x", "Overtime"), ("JM Mascis", "y", "Overtime"), None),
        (("REM", "x", "Overtime"), ("R.E.M.", "y", "Overtime"), "fuzzy"),
        (("R.E.M.", "x", "Overtime"), ("REM", "y", "Overtime"), "fuzzy"),
        # The collapse reads version words too, so a dotted qualifier is the undotted one,
        # whichever side carries the dots.
        (
            ("Jessica Pratt", "x", "Song (LP Version)"),
            ("Jessica Pratt", "y", "Song (L.P. Version)"),
            "fuzzy",
        ),
        (
            ("Jessica Pratt", "x", "Song (L.P. Version)"),
            ("Jessica Pratt", "y", "Song (LP Version)"),
            "fuzzy",
        ),
        (
            ("Jessica Pratt", "x", "Song (FCC Edit)"),
            ("Jessica Pratt", "y", "Song (F.C.C. Edit)"),
            "fuzzy",
        ),
        (
            ("Jessica Pratt", "x", "Song (F.C.C. Edit)"),
            ("Jessica Pratt", "y", "Song (FCC Edit)"),
            "fuzzy",
        ),
        (("Jessica Pratt", "x", "Song (Live)"), ("Jessica Pratt", "y", "Song (L.I.V.E.)"), "fuzzy"),
        (("Jessica Pratt", "x", "Song (L.I.V.E.)"), ("Jessica Pratt", "y", "Song (Live)"), "fuzzy"),
        (("Jessica Pratt", "x", "Song (L.I.V.E.)"), ("Jessica Pratt", "y", "Song"), None),
        # An empty part never joins, on either side.
        (("Jessica Pratt", "x", ""), ("Jessica Pratt", "y", ""), None),
        (("", "x", "Back, Baby"), ("", "y", "Back, Baby"), None),
        (("Jessica Pratt", "x", "Back, Baby"), ("", "y", "Back, Baby"), None),
        (("Jessica Pratt", "x", "Back, Baby"), (None, "y", None), None),
    ],
)
def test_title_tier(
    play: tuple[str, str, str], recording: tuple[str | None, str, str | None], tier: str | None
) -> None:
    artist, album, title = recording
    assert norm.title_tier(*play, (artist,), album, title) == tier


def test_title_tier_joins_either_of_two_artist_names() -> None:
    play = ("Stereolab", "x", "Brakhage")
    artists = ("Stereolab（ステレオラブ）", "Various Artists")
    assert norm.title_tier(*play, artists, "Dots and Loops", "Brakhage") == "fuzzy"
    assert (
        norm.title_tier("Various Artists", "x", "Brakhage", artists, "Dots", "Brakhage") == "exact"
    )
    assert norm.title_tier("Juana Molina", "x", "Brakhage", artists, "Dots", "Brakhage") is None
    assert norm.title_tier(*play, (None, ""), "Dots and Loops", "Brakhage") is None
    assert norm.title_tier(*play, (), "Dots and Loops", "Brakhage") is None


@pytest.mark.parametrize(
    ("play_album", "play_title", "album", "title", "tier"),
    [
        # The recording names the play's version outside brackets, which qualifiers() cannot see:
        # anywhere in its album, but in its title only after a spaced dash.
        ("Edits [Live]", "Back, Baby", "Live at KEXP", "Back, Baby", "exact"),
        ("Edits (Demo)", "Back, Baby - Demo", "Bootleg", "Back, Baby - Demo", "exact"),
        ("Edits [Demo]", "Back, Baby", "Bootleg", "Back, Baby - Demo", None),
        # A title's unspaced or leading word names nothing, so these stay unjoined.
        ("Edits [Live]", "Forever", "Bootleg", "Live Forever", None),
        ("Edits [Live]", "Back, Baby", "Bootleg", "Back, Baby-Live", None),
    ],
)
def test_title_tier_reads_the_version_a_recording_names_outside_brackets(
    play_album: str, play_title: str, album: str, title: str, tier: str | None
) -> None:
    assert (
        norm.title_tier("Jessica Pratt", play_album, play_title, ("Jessica Pratt",), album, title)
        == tier
    )


def test_title_tier_refuses_a_bare_string_for_artists() -> None:
    """A str would be iterated by character: every name longer than one letter would miss."""
    with pytest.raises(TypeError, match="artists"):
        norm.title_tier("Jessica Pratt", "x", "Back, Baby", "Jessica Pratt", "y", "Back, Baby")  # type: ignore[arg-type]
