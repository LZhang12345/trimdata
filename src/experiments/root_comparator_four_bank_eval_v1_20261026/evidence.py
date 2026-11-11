"""Authenticate existing exported HOME metadata, never synthesize results."""
import hashlib
import json
from pathlib import Path

from . import contract as c

EXPORTS = (
    "collaboration/root-control10-closure-phase3-20261026/ACTUAL_THIRTY_DECISION_EXPORT_V1.json",
    "collaboration/root-control10-closure-phase3-20261026/ACTUAL_HOME_GATE_ANNEX_V1.json",
    "collaboration/root-babel-followup-queue-20261026/ACTUAL_QUALIFIER_V3_CUSTODY_V1.json",
)


def exported_documents(root=c.ROOT):
    result = {}
    for relative in EXPORTS:
        for path, entry in json.loads((root / relative).read_bytes())["documents"].items():
            ref, raw = entry["reference"], entry["text"].encode()
            c.require(ref["path"] == path and len(raw) == ref["bytes"]
                      and hashlib.sha256(raw).hexdigest() == ref["sha256"],
                      "Exported original metadata bytes changed")
            c.require(path not in result or result[path] == entry,
                      "Conflicting exported original metadata")
            result[path] = entry
    return result


def read_exported(ref, documents, root=c.ROOT):
    c.require(set(ref) == {"path", "bytes", "sha256"}, "Exact metadata descriptor")
    path = ref["path"]
    if path in documents:
        entry = documents[path]
        c.require(entry["reference"] == ref, "Exported reference differs")
        raw = entry["text"].encode()
    else:
        original = Path(path)
        source_prefix = c.HOME / "native_campaign_v3_20261023/src"
        c.require(original.is_relative_to(c.HOME / "collaboration") or original.is_relative_to(source_prefix),
                  "No local substitute for unexported DATA or checkpoint evidence")
        relative = original.relative_to(source_prefix) if original.is_relative_to(source_prefix) else original.relative_to(c.HOME)
        raw = (root / relative).read_bytes()
    c.require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"],
              "Original HOME metadata hash differs")
    return json.loads(raw)
