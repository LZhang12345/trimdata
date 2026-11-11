#!/usr/bin/env python3
"""Verify downloaded files against a provenance.json (checkpoint) or selection_meta.json (selection)."""
import hashlib, json, os, sys
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
d = sys.argv[1]
if os.path.exists(os.path.join(d, "provenance.json")):
    prov = json.load(open(os.path.join(d, "provenance.json")))
    for f, info in prov["files"].items():
        want = info.get("published_sha256", info["sha256"])
        got = sha(os.path.join(d, f)); print(("OK  " if got == want else "BAD ") + f)
else:
    meta = json.load(open(os.path.join(d, "selection_meta.json")))
    got = sha(os.path.join(d, "records.json"))
    print(("OK  " if got == meta["records_json_sha256"] else "BAD ") + "records.json")
