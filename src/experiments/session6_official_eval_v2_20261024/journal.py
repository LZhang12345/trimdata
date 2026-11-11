"""Per-semantic-QA durability; raw native IDs need not be unique."""
import os
from pathlib import Path
import sqlite3

from . import common as c


def semantic_key(bank, source_sha256, original_row_index, native_id, round_id=0):
    c.require(bank in c.METRICS and isinstance(source_sha256, str)
              and len(source_sha256) == 64
              and all(x in "0123456789abcdef" for x in source_sha256),
              "Exact bank and original source SHA required")
    c.require(type(original_row_index) is int and original_row_index >= 0
              and type(native_id) in (str, int) and type(round_id) is int and round_id >= 0,
              "Original row/native-ID/round identity required")
    return c.key([bank, source_sha256, original_row_index,
                  type(native_id).__name__, native_id, round_id])


class Journal:
    def __init__(self, directory, function, roster):
        c.require(roster and len({r["semantic_key"] for r in roster}) == len(roster),
                  "Complete unique semantic QA roster required")
        for row in roster:
            c.require(row["semantic_key"] == semantic_key(
                row["bank"], row["source_sha256"], row["original_row_index"],
                row["native_id"], row.get("round_id", 0)), "Forged semantic row key")
        self.function = function
        self.order = [row["semantic_key"] for row in roster]
        self.roster = {row["semantic_key"]: row for row in roster}
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "raw_answers.sqlite3"
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata(k TEXT PRIMARY KEY,v TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS answers(k TEXT PRIMARY KEY,raw TEXT NOT NULL,audit TEXT NOT NULL)")
        expected = dict(function=c.key(function), ordered_roster=c.key(roster))
        previous = dict(self.db.execute("SELECT k,v FROM metadata"))
        c.require(not previous or previous == expected, "Foreign model/runtime/QA journal identity")
        if not previous:
            with self.db:
                self.db.executemany("INSERT INTO metadata VALUES(?,?)", expected.items())
        self.present = {row[0] for row in self.db.execute("SELECT k FROM answers")}
        c.require(self.present <= set(self.roster), "Foreign semantic QA in journal")

    def append(self, semantic_id, raw, audit):
        c.require(semantic_id in self.roster and semantic_id not in self.present,
                  "Unknown or duplicate semantic QA")
        expected = self.roster[semantic_id]
        c.require(type(raw.get("question_id")) is type(expected["native_id"])
                  and raw["question_id"] == expected["native_id"]
                  and isinstance(raw.get("text"), str), "Native answer identity/type differs")
        c.require(audit["function_sha256"] == c.key(self.function)
                  and audit["semantic_key"] == semantic_id, "Actual generation audit identity differs")
        with self.db:
            self.db.execute("INSERT INTO answers VALUES(?,?,?)",
                            (semantic_id, c.canonical(raw).decode(), c.canonical(audit).decode()))
        self.present.add(semantic_id)

    def missing(self):
        return [self.roster[k] for k in self.order if k not in self.present]

    def export(self, path):
        c.require(self.present == set(self.roster), "Partial bank cannot be exported as complete")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = b"".join(self.db.execute("SELECT raw FROM answers WHERE k=?", (key,)).fetchone()[0].encode()
                        for key in self.order)
        if path.exists():
            c.require(path.read_bytes() == body, "Existing native answer export changed")
        else:
            with path.open("xb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        return c.desc(path)

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()
