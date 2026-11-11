"""Root-only finite storage-claim resealing before each separate admission.

This reconciles HOME metadata, NOT an HF free-space measurement. Root must pass
the returned descriptor to a fresh registered owned-GPU storage observation.
"""
from __future__ import annotations

import argparse
import re

from . import admission as a, contract as c


def reconcile_claims(runtime, configs, base_ref):
    base = runtime.checked(base_ref)
    a.require(base["owner"] == "Root" and base["covers_all_project_obligations"] is True
              and 0 <= runtime.now() - base["epoch"] <= 300,
              "Fresh Root collective accounting required; this helper does not refresh its evidence age")
    entries, seen = [], set()
    for row in base["entries"]:
        ref = a.descriptor(row["reservation"])
        grant = runtime.checked(ref)
        a.require(ref["path"] not in seen and type(row["unwritten_bytes"]) is int
                  and 0 <= row["unwritten_bytes"] <= grant["reserved_bytes"]
                  and type(row["unwritten_inodes"]) is int and row["unwritten_inodes"] >= 0,
                  "Invalid or duplicate project-wide collective storage obligation")
        seen.add(ref["path"])
        entries.append(dict(row))
    added, conservatively_recounted = [], []
    for sid, cfg in configs.items():
        path = a.ADMISSIONS / sid / "STORAGE.json"
        if not runtime.files.exists(path):
            continue
        ref, grant = runtime.document(path)
        a.require(grant["configuration"]["path"] == str(c.CONTROL / "configs" / ("CONFIG_" + sid + ".json"))
                  and grant["output"] == cfg["output"] and grant["reserved_bytes"] == cfg["output_limit_bytes"]
                  and grant["reserved_inodes"] == 128
                  and grant["checkpoint_retention"] == cfg["checkpoint_retention"],
                  "Existing actual current12 STORAGE grant does not belong to this reviewed function")
        prior = [row for row in entries if row["reservation"]["path"] == str(path)]
        if prior:
            a.require(len(prior) == 1 and prior[0]["reservation"] == ref,
                      "Base collective accounting uses changed bytes for an existing STORAGE grant")
            row = prior[0]
            row.update(unwritten_bytes=grant["reserved_bytes"], unwritten_inodes=grant["reserved_inodes"])
            conservatively_recounted.append(ref)
        else:
            entries.append(dict(reservation=ref, unwritten_bytes=grant["reserved_bytes"],
                                unwritten_inodes=grant["reserved_inodes"]))
            seen.add(ref["path"])
            added.append(ref)
    return dict(status="ROOT_RECONCILED_COLLECTIVE_CLAIMS_FOR_NEXT_CURRENT12_ADMISSION_V1",
        owner="Root", covers_all_project_obligations=True, epoch=base["epoch"],
        metadata_reconciled_epoch=runtime.now(), source_collective_claims=base_ref, entries=entries,
        newly_included_current12_STORAGE_grants=added,
        conservatively_recounted_current12_STORAGE_grants=conservatively_recounted,
        prior_result_reservations_counted_in_full_no_unverified_written_credit=True,
        actual_free_space_measured_here=False, no_qualification_or_evaluation_PASS_claimed=True,
        required_next_step="Bind this exact claim descriptor into a NEW actual registered-GPU HF free-space observation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--source-review", required=True)
    parser.add_argument("--root-approval", required=True)
    parser.add_argument("--base-collective-claims", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute-root-reviewed", action="store_true")
    args = parser.parse_args()
    a.require(args.execute_root_reviewed, "No operation: --execute-root-reviewed is required")
    runtime = a.Runtime()
    a.require(runtime.uid() == a.UID, "Only actual Root-owned Babel metadata actor")
    with runtime.files.lock() as recheck:
        _, configs, _ = a.reviewed(runtime, args.configuration, args.packet,
                                   args.source_review, args.root_approval)
        base_ref, _ = runtime.document(args.base_collective_claims)
        result = reconcile_claims(runtime, configs, base_ref)
        path = a.literal(args.output)
        a.require(path.parent == a.ADMISSIONS / "storage-reconciliation"
                  and re.fullmatch(r"CLAIMS_[A-Za-z0-9_-]+\.json", path.name),
                  "New exact Root HOME reconciliation output required")
        recheck()
        runtime.files.mkdir(path.parent)
        ref = a.seal_same(runtime, path, result)
    print(c.canonical(dict(collective_claims=ref, next_step=result["required_next_step"])).decode())


if __name__ == "__main__":
    main()
