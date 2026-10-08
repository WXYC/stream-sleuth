"""Synthetic ``plays.jsonl`` fixtures shared by the score and queue tests.

``hour_records`` windows an hour of plays as ``corpus.write_plays`` does; ``RECORDS`` is the
standard hour, described below. Nothing here reads research data.
"""

from __future__ import annotations

HOUR = "2026/08/12/202608121600.mp3"
EMPTY = "2026/08/12/202608121700.mp3"  # gridded, never queried
SHORT = "2026/08/12/202608121800.mp3"  # decodes to 1,800 s
OTHER = "2026/08/12/202608121900.mp3"  # outside the leg's hours
PADS = {"canonical": 180.0, "etl": 220.0}

Track = tuple[str, str, str]  # artist, song, album
MOLINA: Track = ("Juana Molina", "la paradoja", "DOGA")
PRATT: Track = ("Jessica Pratt", "Back, Baby", "On Your Own Love Again")
CHUQUI: Track = ("Chuquimamani-Condori", "Call Your Name", "Edits")
HERMANOS: Track = ("Hermanos Gutiérrez", "El Bueno y el Malo", "El Bueno y el Malo")
ELLINGTON: Track = ("Duke Ellington & John Coltrane", "In a Sentimental Mood", "Duke Ellington & John Coltrane")  # fmt: skip
UNLOGGED: Track = ("Stereolab", "French Disko", "Jenny Ondioline")
COCREDIT = "Juana Molina & Chuquimamani-Condori"
STAGE = "a" * 40  # an Olaf reference's pool stage id


def hour_records(hour_key: str, era: str, *rows: tuple[int, float, Track, dict]) -> list[dict]:
    """``plays.jsonl`` records for one hour, windowed as ``corpus.write_plays`` windows them.

    The first row is the carryover when its offset is negative.
    """
    records = []
    for i, (play_id, t, (artist, title, album), extra) in enumerate(rows):
        t_next = rows[i + 1][1] if i + 1 < len(rows) else 3600.0
        records.append(
            {
                "hour_key": hour_key,
                "play_id": play_id,
                "t_offset_s": t,
                "window_start_s": max(0.0, t - PADS[era]),
                "window_end_s": min(3600.0, t_next + PADS[era]),
                "artist": artist,
                "title": title,
                "album": album,
                "era": era,
                "pad_s": PADS[era],
                "reorder_flag": False,
                "play_order_status": "single_writer",
                "carryover": t < 0,
                "in_pool": True,
                "pool_match_tier": "exact",
                "pool_format": "mp3",
                "rotation": True,
                "talk_rows": 0,
                "group": "canonical-high",
                "band": "daytime",
                "subset": False,
                **extra,
            }  # fmt: skip
        )
    return records


# Windows: carryover [0, 300], 1 [0, 780], 2 [420, 1080], 3 [720, 1680], 4 [1320, 2580],
# 5 [2220, 3600]. Play 4's show is reorder-flagged and play 5's play order is unreliable: neither
# changes attribution, which reads windows and logged intervals only. Play 4's interval runs
# through a talk break (2,100 s to 2,400 s) in which nothing is logged. Play 3 is out of the pool.
RECORDS = hour_records(
    HOUR,
    "canonical",
    (100, -200.0, HERMANOS, {}),
    (1, 120.0, MOLINA, {}),
    (2, 600.0, PRATT, {}),
    (3, 900.0, PRATT, {"in_pool": False, "pool_match_tier": None}),  # the same song twice
    (4, 1500.0, CHUQUI, {"reorder_flag": True}),
    (5, 2400.0, ELLINGTON, {"play_order_status": "unreliable", "reorder_flag": None}),
)
