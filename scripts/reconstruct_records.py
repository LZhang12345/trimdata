#!/usr/bin/env python3
"""Rebuild the exact training input (records.json) of a selection from the public LLaVA-1.5 mixture.

usage: python reconstruct_records.py --source llava_v1_5_mix665k.json --selection selections/<method>/<budget>/<seed> [--out records.json]

The source file is `llava_v1_5_mix665k.json` from the Hugging Face dataset liuhaotian/LLaVA-Instruct-150K
(revision 9d451dc7629cfe0469f6ae4432b765cd603d5fcb). Record ids have the form "<index>:<LLaVA id>" where <index> is the
0-based position in that file. The output is byte-identical to the file we trained on; its sha256 is checked against
selection_meta.json.
"""
import argparse, hashlib, json, os, sys
SOURCE_SHA256 = "ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8"  # sha256 of llava_v1_5_mix665k.json (1,029,887,963 bytes)
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True); ap.add_argument("--selection", required=True); ap.add_argument("--out")
a = ap.parse_args()
meta = json.load(open(os.path.join(a.selection, "selection_meta.json")))
ids = json.load(open(os.path.join(a.selection, "record_ids.json")))
h = hashlib.sha256(open(a.source, "rb").read()).hexdigest()
if h != SOURCE_SHA256: print("warning: source sha256 %s differs from the expected upstream file" % h)
src = json.load(open(a.source, encoding="utf-8"))
recs = []
for rid in ids:
    i, lid = rid.split(":", 1); r = src[int(i)]
    if str(r["id"]) != lid: sys.exit(f"id mismatch at {rid}: source has {r['id']} (wrong source file?)")
    recs.append(r)
fmt = meta["records_json_reconstruction"]["serialization"]
kw = dict(fmt["dumps_kwargs"])
if kw.get("separators"): kw["separators"] = tuple(kw["separators"])
data = (json.dumps(recs, **kw) + ("\n" if fmt["trailing_newline"] else "")).encode("utf-8")
got = hashlib.sha256(data).hexdigest()
out = a.out or os.path.join(a.selection, "records.json")
open(out, "wb").write(data)
print(f"wrote {out}: {len(recs)} records, sha256 {got}")
if got != meta["records_json_sha256"]: sys.exit("SHA256 MISMATCH vs selection_meta.json (expected %s)" % meta["records_json_sha256"])
print("sha256 matches the training input")
