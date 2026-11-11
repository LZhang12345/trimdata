"""Read-only recovery planning for an interrupted, explicitly intended retirement."""
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .checkpoint import (
    checked_json, descriptor, integer, require, safe_member, validate_checkpoint_directory,
)


def plan_retirement_recovery(
    *, checkpoint: Path, training_root: Path, report_root: Path, identity: str,
    intent_descriptor: Mapping[str, Any], archive_descriptor: Mapping[str, Any],
    newer_checkpoint: Path, cuda_state_sizes: Sequence[int],
) -> dict[str, Any]:
    """Authorize no mutation: return an exact reconciliation plan for Root review."""
    training = Path(training_root).absolute()
    old, newer = Path(checkpoint).absolute(), Path(newer_checkpoint).absolute()
    require(not training.is_symlink() and old.parent == training and newer.parent == training
            and not old.is_symlink() and not newer.is_symlink(), "Recovery outside exact owned checkpoint tree")
    relative = old.name
    require(relative.startswith("checkpoint-") and relative[len("checkpoint-"):].isdigit(),
            "Unknown checkpoint directory identity")
    receipt_root = Path(report_root).absolute() / "retired-new-intermediates" / relative
    require(Path(intent_descriptor["path"]) == receipt_root / "DELETION_INTENT.json"
            and Path(archive_descriptor["path"]) == receipt_root / "ACCEPTED_ARCHIVE.json",
            "Recovery metadata is not the exact retirement transaction")
    for path in (receipt_root, receipt_root.parent, Path(report_root).absolute(),
                 Path(intent_descriptor["path"]), Path(archive_descriptor["path"])):
        require(not path.is_symlink(), "Symlink in recovery evidence")
    intent, archive = checked_json(intent_descriptor), checked_json(archive_descriptor)
    require(intent["identity"] == archive["identity"] == identity
            and archive["status"] == "ACCEPTED_FULL_NATIVE_STATE" and archive["world_size"] == 1,
            "Foreign retirement/archive identity")
    step = integer(archive["global_step"], "retired global_step")
    require(old.name == "checkpoint-" + str(step) and Path(intent["path"]) == old,
            "Retirement path/cursor differs")
    require(isinstance(archive["files"], dict) and archive["files"]
            and intent["files"] == archive["files"], "Retirement intent/archive file inventories differ")
    require(sum(integer(ref["bytes"], "archived member bytes") for ref in archive["files"].values())
            == archive["logical_bytes"], "Archived byte accounting differs")
    validated = validate_checkpoint_directory(newer, identity, cuda_state_sizes=cuda_state_sizes)
    require(validated["global_steps"] > step, "No newer semantically valid accepted checkpoint")
    missing, remaining = [], []
    for name, expected in archive["files"].items():
        require(name != "ACCEPTED.json", "Archive cannot inventory its own receipt")
        member = safe_member(old, name)
        if not member.exists():
            missing.append(name)
            continue
        actual = descriptor(member)
        require(actual["sha256"] == expected["sha256"] and actual["bytes"] == expected["bytes"],
                "Remaining retired member changed: " + name)
        remaining.append(actual)
    accepted = old / "ACCEPTED.json"
    if accepted.exists():
        require(not accepted.is_symlink(), "Retired accepted receipt is a symlink")
        import json
        require(json.loads(accepted.read_text()) == archive, "Retired receipt differs from accepted archive")
    for path in old.rglob("*"):
        require(not path.is_symlink(), "Symlink in retired checkpoint")
        if path.is_file():
            name = str(path.relative_to(old))
            require(name in archive["files"] or name == "ACCEPTED.json",
                    "Unrecorded file in retirement target: " + name)
    return {
        "status": "PASS_EXACT_RETIREMENT_RECOVERY_PLAN_ONLY",
        "identity": identity, "checkpoint": str(old), "intent": dict(intent_descriptor),
        "archive": dict(archive_descriptor), "newer_accepted": validated["accepted_descriptor"],
        "newer_semantic_state": validated["status"], "missing_intended_members": sorted(missing),
        "remaining_verified_members": remaining, "retired_ACCEPTED_present": accepted.exists(),
        "mutation_performed": False, "arbitrary_dangling_checkpoint_ignoring_authorized": False,
        "next": "Root separately reviews/rechecks this exact transaction and current consumers before mutation.",
    }
