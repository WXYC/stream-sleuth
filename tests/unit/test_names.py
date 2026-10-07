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
    assert norm.title_tier(*play, [artist], album, title) == tier


def test_title_tier_joins_either_of_two_artist_names() -> None:
    play = ("Stereolab", "x", "Brakhage")
    artists = ["Stereolab（ステレオラブ）", "Various Artists"]
    assert norm.title_tier(*play, artists, "Dots and Loops", "Brakhage") == "fuzzy"
    assert (
        norm.title_tier("Various Artists", "x", "Brakhage", artists, "Dots", "Brakhage") == "exact"
    )
    assert norm.title_tier("Juana Molina", "x", "Brakhage", artists, "Dots", "Brakhage") is None
    assert norm.title_tier(*play, [None, ""], "Dots and Loops", "Brakhage") is None
    assert norm.title_tier(*play, [], "Dots and Loops", "Brakhage") is None
