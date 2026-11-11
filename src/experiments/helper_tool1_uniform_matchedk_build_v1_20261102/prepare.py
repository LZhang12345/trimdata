"""Tool1 matched-K Uniform (random) controls, selection seed 1235, at DataTailor's realized 5/15/20% budgets.

Frozen design (BUDGET_EXPANSION_PROTOCOL_V1): two stages per budget.
  1. A validated nominal-K seed-1235 Uniform input produced by the unchanged native selector
     (root_native_campaign_v7 selection.choose, hash rank of ['native-campaign-v1', seed, record_id]).
     5% and 20%: the manager-native-wave PASS_SELECTION sets are reused read-only after exact
     reproduction by the pinned native selector.  15%: no stored seed-1235 set exists, so the same
     pinned selector is run at the registered nominal K inside this allocation.
  2. The reviewed derive.py (exact bytes 3b5aafdc) drops input positions with an independent fixed
     derivation seed to exactly DataTailor's realized K, rejecting any split selection unit.

The registered Uniform selection unit is one whole pool conversation (budget unit
whole_conversation_count; native Uniform passes groups=None), so every unit is a singleton group and
group-conditioning is vacuous.  Alias groups by original LLaVA id are audited and disclosed, not enforced.
Outputs are written once to the shared HF NAS in the exact format the Tool1 v5 trainer verifies; the
pinned v5 validate_selection/validate_roster are executed on the real outputs inside this allocation.
"""

import argparse
import ast
from contextlib import ExitStack, contextmanager
from decimal import Decimal, ROUND_FLOOR
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import time
from types import ModuleType


HOME = Path("/home/luyangz/data_selection_multimodal")
SOURCE = HOME / "experiments/helper_tool1_uniform_matchedk_build_v1_20261102"
CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/uniform-matchedk-build-v1"
LOCK = HOME / "collaboration/babel-two-track-20261019/ADMISSION.lock"
OUTPUT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-uniform-matchedk-v1")
OWNER = "Tool1-owned-uniform-matchedk-selection"
REVIEW_STATUS = "SELF_REVIEWED_TOOL1_UNIFORM_MATCHEDK_BUILD_V1_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_UNIFORM_MATCHEDK_BUILD_V1"
SELECTION_SCHEMA = "tool1-uniform-matchedk-selection-v1"
SELECTION_STATUS = "TOOL1_UNIFORM_MATCHEDK_BUILT_AND_VALIDATED_IN_ALLOCATION"
UID = 2708899
N = 660763
SEED = 1235
TEN_PERCENT_K = 66076
SPEC = {"N": N, "seed": SEED, "ten_K": TEN_PERCENT_K, "budgets": {
    "0.05": {"nominal_K": 33038, "K": 33032, "input": "manager", "derivation_seed": 202611020005},
    "0.15": {"nominal_K": 99114, "K": 99108, "input": "native", "derivation_seed": 202611020015},
    "0.20": {"nominal_K": 132152, "K": 132147, "input": "manager", "derivation_seed": 202611020020},
}}
POOL = {
    "path": "/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/POOL.json",
    "bytes": 18754787,
    "sha256": "c76040a17ca0dc1e99fea14514618edade3171a9b05f6ccc3e4751a5939e4964",
}
POOL_RECORDS = {
    "path": "/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/records.json",
    "bytes": 760064389,
    "sha256": "1c53e08b940af27483201efc4c2fa27e3f5c4f7862484e078736c718673c1d36",
}
NATIVE = HOME / "coincide_native_exec_src_v2_20261028/experiments/root_native_campaign_v7_20261024"
NATIVE_COMMON = {"path": str(NATIVE / "common.py"), "bytes": 3798,
                 "sha256": "e21cf1ef0cfafdf4e295bbabfc052f0638a18e0e36ea43a532e440572ba77a2d"}
NATIVE_SELECTION = {"path": str(NATIVE / "selection.py"), "bytes": 11948,
                    "sha256": "e805cbb4dabef429ed9c94caaa2cc46a1271587a18c71e4fc8b03bce9e9e4423"}
DERIVE = {"path": str(SOURCE / "derive.py"), "bytes": 2547,
          "sha256": "3b5aafdc5756afd4ebf3c8ae53de5423bafb094f4d94d32fe482a76db5c27dad"}
V5_PREPARE = {"path": str(HOME / "experiments/helper_tool1_baseline_train_v5_20261102/prepare.py"), "bytes": 59012,
              "sha256": "d965f35937cf0a6c84a447346760366184fe65f1ece97a9dd96c602a46207c2c"}
ROSTER_V4 = {"path": str(HOME / "collaboration/helper-tool1-baseline-independent-20261101/baseline-train-v5/ROSTER_V4.json"),
             "bytes": 10909, "sha256": "7ba2c27afa9ccf29f9f23acdcacf0877531c82680481e0fcd6b7b8389342d557"}
SESSION4_1235 = {"path": str(HOME / "collaboration/session4-uniform-native-v4-20261031/qualified/1235/SELECTION.json"),
                 "bytes": 1425672, "sha256": "277d0345274c16ba452c43d7e0582ab4c249f3c8d85d99f59117a847ba1b28cc"}
MANAGER = {
    "0.05": {"path": "/data/hf_cache/project_artifacts/DS-MultiModal/manager-native-wave-v1/native5-uniform-sel1235/SELECTION.json",
             "bytes": 727749, "sha256": "2d36a06a0913cddf7ea1460694810a9e45c7045618bd6b1abc9d315155e9e293"},
    "0.20": {"path": "/data/hf_cache/project_artifacts/DS-MultiModal/manager-native-wave-v1/native20-uniform-sel1235/SELECTION.json",
             "bytes": 2864222, "sha256": "3f7231563b45290aa429e6eaef6a95e8c2903d67d67730faf1d55971d4bf7a66"},
}
DATATAILOR = {
    "0.05": {"path": "/data/hf_cache/project_features/DS-MultiModal/dt_formal_selection_20261026/v1/budget-0.05/SELECTION_CANDIDATE.json",
             "bytes": 946597, "sha256": "b8d8e4cb32a8c385fe7b12a326fd20c97f877f4eb4b35ec5370c7a3b70212d38"},
    "0.15": {"path": "/data/hf_cache/project_features/DS-MultiModal/dt_formal_selection_20261026/v2/budget-0.15/SELECTION_CANDIDATE.json",
             "bytes": 2704132, "sha256": "727785224aa08e602d20d8f8fb099535e770e4386b08de92a03112c80ae38f58"},
    "0.20": {"path": "/data/hf_cache/project_features/DS-MultiModal/dt_formal_selection_20261026/v2/budget-0.20/SELECTION_CANDIDATE.json",
             "bytes": 3582463, "sha256": "47b95c2f5db5653df1bfbc43d957733e2657c77b6ebf49a83739b3d7d751c5ca"},
}
DT_CELLS = {"0.05": "bx-datatailor-b05-k33032-train", "0.15": "bx-datatailor-b15-k99108-train",
            "0.20": "bx-datatailor-b20-k132147-train"}
SOURCES = ("coco", "gqa", "ocr_vqa", "textvqa", "vg", "text_only")
JOB_NAME = "dsmm-t1-unimk-build-v1"
TIME = "00:30:00"
EXCLUDED = "babel-p9-20"
SLURM_ENV = {"HOME": str(HOME), "PATH": "/usr/bin:/bin", "USER": "luyangz", "LOGNAME": "luyangz", "LC_ALL": "C"}
ALGORITHM = ("stage1: native root_native_campaign_v7 selection.choose(ids,'uniform',nominal_K,1235) = pool-order "
             "first nominal_K of ranking by sha256(canonical(['native-campaign-v1',1235,record_id])); 5%/20% reuse the "
             "byte-pinned manager-native-wave PASS_SELECTION sets after exact reproduction, 15% is regenerated; "
             "stage2: reviewed derive.py random.Random(derivation_seed).sample of input positions to drop to "
             "DataTailor realized K, singleton whole-conversation units, retained pool order")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def floor_budget(n, fraction):
    return int((Decimal(n) * Decimal(fraction)).to_integral_value(rounding=ROUND_FLOOR))


# ---------------------------------------------------------------- pure selection logic (unit tested)

def load_module(raw, path, name):
    module = ModuleType(name)
    module.__file__ = path
    exec(compile(raw, path, "exec", dont_inherit=True), module.__dict__)
    return module


def native_choose(common_raw, common_path, selection_raw, selection_path):
    """Exact native choose() body from pinned bytes; only its Uniform branch is reachable here."""
    common = load_module(common_raw, common_path, "tool1_pinned_native_common")
    tree = ast.parse(selection_raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "choose"]
    require(len(nodes) == 1, "Exactly one native choose() required")
    namespace = {"c": common, "METHODS": ("uniform",)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), selection_path, "exec", dont_inherit=True), namespace)
    return namespace["choose"]


def derive_function(raw, path):
    return load_module(raw, path, "tool1_pinned_matched_k_derive").derive


def source_of(row):
    image = row.get("image")
    if image is None:
        return "text_only"
    require(isinstance(image, str) and image and not image.startswith("/") and ".." not in image.split("/"),
            "Canonical relative image locator required")
    top = image.split("/", 1)[0]
    require(top in SOURCES[:-1] and "/" in image, "Unknown original image source")
    return top


def composition(rows):
    counts = dict.fromkeys(SOURCES, 0)
    for row in rows:
        counts[source_of(row)] += 1
    total = sum(counts.values())
    return {"total": total, "counts": counts,
            "shares": {k: (round(v / total, 6) if total else 0.0) for k, v in counts.items()}}


def chi_square(observed, reference):
    total, ref_total = observed["total"], reference["total"]
    value = 0.0
    for name in SOURCES:
        expected = total * reference["counts"][name] / ref_total
        if expected > 0:
            value += (observed["counts"][name] - expected) ** 2 / expected
    return round(value, 4)


def validate_pool(pool, rows, n):
    ids = pool.get("record_ids")
    require(pool.get("schema") == "published-llava-pool-v1" and pool.get("status") == "PASS_POOL"
            and pool.get("pool_kind") == "all_source_protected_bridge" and pool.get("count") == n
            and isinstance(ids, list) and isinstance(rows, list) and len(ids) == len(rows) == n
            and len(set(ids)) == n, "Complete unique frozen protected pool required")
    indices = pool.get("source_indices")
    require(isinstance(indices, list) and len(indices) == n, "Pool source indices required")
    for ident, row, index in zip(ids, rows, indices):
        require(isinstance(ident, str) and type(row) is dict and type(index) is int
                and ident == "%06d:%s" % (index, row.get("id", "")), "Pool record ID/payload alignment differs")
    return ids


def alias_groups(ids):
    """Original-LLaVA-id alias groups (informational audit only)."""
    groups = {}
    for ident in ids:
        groups.setdefault(ident.split(":", 1)[1], []).append(ident)
    return {key: members for key, members in groups.items() if len(members) > 1}


def split_alias_groups(selected, aliases):
    chosen = set(selected)
    split = 0
    for members in aliases.values():
        inside = sum(member in chosen for member in members)
        if 0 < inside < len(members):
            split += 1
    return split


def check_native_selection(doc, expected_ids, *, fraction, k, pool_ref, seed, n):
    require(doc.get("schema") == "native-record-selection-v1" and doc.get("status") == "PASS_SELECTION"
            and doc.get("method") == "uniform" and doc.get("selection_seed") == seed
            and doc.get("fraction") == fraction and doc.get("count") == k and doc.get("pool") == pool_ref
            and doc.get("groups") is None and doc.get("features") is None
            and doc.get("budget") == {"unit": "whole_conversation_count", "value": k, "fraction": fraction,
                                      "eligible_pool_count": n},
            "Reused native Uniform selection metadata differs: " + fraction)
    require(doc.get("record_ids") == expected_ids,
            "Pinned native selector does not reproduce the reused Uniform selection: " + fraction)


def build_all(pool, rows, choose, derive, session4, manager, manager_rows, datatailor, spec, pool_ref):
    """All validation and selection before any output is written; returns per-budget outputs and reports."""
    n, seed = spec["N"], spec["seed"]
    ids = validate_pool(pool, rows, n)
    position = {ident: index for index, ident in enumerate(ids)}
    fractions = sorted(spec["budgets"])
    nominal = {"0.10": choose(ids, "uniform", spec["ten_K"], seed)}
    for fraction in fractions:
        budget = spec["budgets"][fraction]
        require(floor_budget(n, fraction) == budget["nominal_K"] and 0 < budget["K"] <= budget["nominal_K"],
                "Registered nominal/realized budget differs: " + fraction)
        nominal[fraction] = choose(ids, "uniform", budget["nominal_K"], seed)
    require(floor_budget(n, "0.10") == spec["ten_K"], "Registered 10% budget differs")
    check_native_selection(session4, nominal["0.10"], fraction="0.10", k=spec["ten_K"], pool_ref=pool_ref,
                           seed=seed, n=n)
    reuse = {}
    for fraction in fractions:
        budget = spec["budgets"][fraction]
        if budget["input"] != "manager":
            require(budget["input"] == "native" and fraction not in manager, "Exact input route required")
            continue
        doc = manager[fraction]
        check_native_selection(doc, nominal[fraction], fraction=fraction, k=budget["nominal_K"],
                               pool_ref=pool_ref, seed=seed, n=n)
        require(manager_rows[fraction] == [rows[position[i]] for i in doc["record_ids"]],
                "Reused manager records differ from the pool's whole conversations")
        reuse[fraction] = True
    chain = [set(nominal[f]) for f in sorted(nominal)]
    require(all(a <= b for a, b in zip(chain, chain[1:])), "Native seed-1235 nominal sets must be nested")
    sizes = {ident: 1 for ident in ids}
    aliases = alias_groups(ids)
    pool_mix = composition(rows)
    outputs, reports = {}, {}
    for fraction in fractions:
        budget = spec["budgets"][fraction]
        source_ids = nominal[fraction]
        indices = [position[i] for i in source_ids]
        require(indices == sorted(indices), "Nominal input must be in pool order")
        derived = derive(indices, list(source_ids), sizes, population_n=n, input_k=len(source_ids),
                         output_k=budget["K"], seed=budget["derivation_seed"])
        chosen = [ids[p] for p in derived["selected_indices"]]
        require(len(chosen) == len(set(chosen)) == budget["K"] and set(chosen) <= set(source_ids)
                and all(i in position for i in chosen) and [position[i] for i in chosen] == sorted(position[i] for i in chosen),
                "Derived Uniform selection is not an exact unique pool-ordered subset: " + fraction)
        selected_rows = [rows[position[i]] for i in chosen]
        dt_doc = datatailor[fraction]
        dt_ids = dt_doc.get("record_ids")
        require(isinstance(dt_ids, list) and len(dt_ids) == len(set(dt_ids)) == budget["K"]
                and all(i in position for i in dt_ids), "Pinned DataTailor realized-K selection differs: " + fraction)
        out_mix = composition(selected_rows)
        in_mix = composition(rows[p] for p in indices)
        dt_mix = composition(rows[position[i]] for i in dt_ids)
        dropped = sorted(set(source_ids) - set(chosen), key=position.__getitem__)
        reports[fraction] = {
            "K": budget["K"], "nominal_K": budget["nominal_K"], "derivation_seed": budget["derivation_seed"],
            "input_route": "reused-manager-native-wave" if fraction in reuse else "regenerated-native-in-allocation",
            "input_ordered_ID_sha256": sha_value(source_ids), "dropped_record_ids": dropped,
            "output_ordered_ID_sha256": sha_value(chosen),
            "checks": {"exact_K": True, "unique": True, "subset_of_pool": True, "subset_of_nominal_input": True,
                       "pool_order": True, "singleton_units_complete": True,
                       "nested_native_chain_0.05_0.10_0.15_0.20": True},
            "alias_audit": {"pool_alias_groups_by_original_id": len(aliases),
                            "pool_records_in_alias_groups": sum(len(v) for v in aliases.values()),
                            "input_split_alias_groups": split_alias_groups(source_ids, aliases),
                            "output_split_alias_groups": split_alias_groups(chosen, aliases),
                            "enforced": False},
            "composition": {"pool": pool_mix, "nominal_input": in_mix, "output": out_mix, "datatailor": dt_mix,
                            "chi_square_df5_output_vs_pool": chi_square(out_mix, pool_mix),
                            "chi_square_df5_datatailor_vs_pool": chi_square(dt_mix, pool_mix)},
            "datatailor_metadata": {"method": dt_doc.get("method"), "count": dt_doc.get("count"),
                                    "pool_equal": dt_doc.get("pool") == pool_ref,
                                    "requested_K": dt_doc.get("requested_K")},
            "overlap": {"output_and_datatailor": len(set(chosen) & set(dt_ids)),
                        "expected_if_independent_uniform": round(budget["K"] * budget["K"] / n, 2),
                        "output_and_nominal_input": len(chosen)},
            "image_rows": sum("image" in row for row in selected_rows),
            "text_only_rows": sum("image" not in row for row in selected_rows),
        }
        outputs[fraction] = {"record_ids": chosen, "rows": selected_rows}
    nominal_15 = nominal["0.15"]
    return outputs, reports, {"0.15": nominal_15}


def selection_document(fraction, chosen, records_ref, report, plan, job, inputs):
    budget = SPEC["budgets"][fraction]
    return {"schema": SELECTION_SCHEMA, "status": SELECTION_STATUS, "method": "uniform", "selection_seed": SEED,
            "derivation_seed": budget["derivation_seed"], "fraction": fraction, "count": budget["K"],
            "record_ids": chosen, "records_json": records_ref, "pool": POOL,
            "budget": {"unit": "whole_conversation_count", "value": budget["K"], "fraction": fraction,
                       "nominal_K": budget["nominal_K"], "eligible_pool_count": N,
                       "matched_to": "DataTailor realized K (selection job 10942655, no top-up)"},
            "matched_datatailor_selection": DATATAILOR[fraction], "nominal_input": inputs[fraction],
            "groups": {"unit": "whole_conversation_record", "all_singleton": True,
                       "alias_by_original_id_enforced": False},
            "algorithm": ALGORITHM, "algorithm_sources": {"native_common": NATIVE_COMMON,
                                                          "native_selection": NATIVE_SELECTION, "derive": DERIVE},
            "input_ordered_ID_sha256": report["input_ordered_ID_sha256"],
            "producer_job": job, "function_identity": plan["function_identity"],
            "not_a_training_or_scientific_result": True}


def candidate_cells(roster, fraction, selection_ref, records_ref):
    """Candidate v5 roster cells mirroring the DataTailor budget cells (recipe/minutes/save_steps unchanged)."""
    cells = {}
    for training_seed in (17, 29, 43):
        base = dict(roster["cells"][DT_CELLS[fraction] + str(training_seed)])
        tag = fraction.replace("0.", "")
        base.update({"method": "uniform", "selection_seed": SEED, "selection_kind": "group",
                     "selection_status": None, "selection": selection_ref, "records": records_ref,
                     "job_name": "dsmm-t1p-bx-un%s-t%d" % (tag, training_seed)})
        cells["bx-uniform-b%s-k%d-sel1235-train%d" % (tag, base["K"], training_seed)] = base
    return cells


# ---------------------------------------------------------------- literal, owned, pinned file I/O

def parent_fd(path):
    text = os.fspath(path)
    require(text.startswith("/") and not text.startswith("//") and not text.endswith("/") and "\0" not in text,
            "Literal absolute file path required")
    parts = text[1:].split("/")
    require(parts and all(part not in ("", ".", "..") for part in parts), "Literal path components required")
    with ExitStack() as stack:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        stack.callback(os.close, fd)
        for component in parts[:-1]:
            fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            stack.callback(os.close, fd)
        return os.dup(fd), parts[-1]


def open_literal(path, flags, mode=0o600):
    directory, name = parent_fd(path)
    try:
        return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, mode, dir_fd=directory)
    finally:
        os.close(directory)


def read_fd(fd, limit, label):
    before = os.fstat(fd)
    require(stat.S_ISREG(before.st_mode) and before.st_uid == UID and before.st_size <= limit,
            "Bounded regular owned file required: " + label)
    chunks, remaining = [], before.st_size + 1
    while remaining:
        block = os.read(fd, min(8 << 20, remaining))
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    raw, after = b"".join(chunks), os.fstat(fd)
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(len(raw) == before.st_size and all(getattr(before, k) == getattr(after, k) for k in keys),
            "File changed while open: " + label)
    return raw, after


def read_exact(path, limit):
    fd = open_literal(path, os.O_RDONLY)
    try:
        return read_fd(fd, limit, str(path))[0]
    finally:
        os.close(fd)


def descriptor(path, limit=4 << 20):
    raw = read_exact(path, limit)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked(ref, limit=4 << 20):
    require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}, "Exact descriptor required")
    raw = read_exact(ref["path"], max(limit, ref["bytes"]))
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"],
            "Pinned descriptor changed: " + ref["path"])
    return raw


def seal(path, value):
    raw = value if isinstance(value, bytes) else canonical(value) + b"\n"
    directory, name = parent_fd(path)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                     dir_fd=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def private_directory(path):
    directory, name = parent_fd(path)
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=directory)
        except FileExistsError:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
                    "Owned private directory required: " + str(path))
        os.fsync(directory)
    finally:
        os.close(directory)


def parse_mount(fdinfo, mountinfo, device):
    """Same acceptance rule as the v5 trainer's 'group' input kind: the exact shared HF NAS."""
    ids = re.findall(r"^mnt_id:\s*([0-9]+)\s*$", fdinfo, re.MULTILINE)
    require(len(ids) == 1, "Exact open FD mount ID required")
    found = []
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        a, b = left.split(), right.split()
        if a and a[0] == ids[0]:
            require(separator and len(a) >= 6 and len(b) >= 3, "Malformed FD mount metadata")
            found.append((a, b))
    require(len(found) == 1, "One mount entry must match actual FD")
    a, b = found[0]
    require(a[2] == "%d:%d" % (os.major(device), os.minor(device)) and a[4] == "/data/hf_cache"
            and b[0] == "nfs4" and b[1] == "babel-nas-2:/srv/nfs/data/hf_cache", "Exact shared HF NAS required")
    return {"mount_id": int(ids[0]), "device": a[2], "target": a[4], "filesystem": b[0], "source": b[1]}


def nas_readback(ref, limit):
    fd = open_literal(ref["path"], os.O_RDONLY)
    try:
        raw, info = read_fd(fd, limit, ref["path"])
        proc = Path("/proc") / str(os.getpid())
        mount = parse_mount((proc / "fdinfo" / str(fd)).read_text(), (proc / "mountinfo").read_text(), info.st_dev)
    finally:
        os.close(fd)
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"], "NAS readback differs")
    return raw, mount


# ---------------------------------------------------------------- plan, reviews, admission

def make_plan(source_ref, batch_ref, derive_ref):
    function = {"owner": OWNER, "purpose": "matched-K Uniform controls at DataTailor realized 5/15/20% budgets",
                "pool": POOL, "pool_records": POOL_RECORDS, "spec": SPEC, "algorithm": ALGORITHM,
                "native_common": NATIVE_COMMON, "native_selection": NATIVE_SELECTION, "derive": DERIVE,
                "reused_inputs": {"session4_10pct_sel1235_identity_check": SESSION4_1235, "manager": MANAGER},
                "datatailor": DATATAILOR, "v5_prepare": V5_PREPARE, "roster_v4": ROSTER_V4,
                "output": str(OUTPUT), "control": str(CONTROL), "benchmark_QA_reads": 0}
    function_id = hashlib.sha256(canonical(function)).hexdigest()
    return {"schema": "tool1-uniform-matchedk-build-v1", "owner": OWNER, "function": function,
            "function_identity": function_id,
            "sources": {"prepare.py": source_ref, "run.sbatch": batch_ref, "derive.py": derive_ref},
            "job_name": JOB_NAME, "comment": "dsmm-t1-unimk-build-v1:" + function_id,
            "resources": {"account": "cx", "partition": "preempt", "qos": "preempt_qos", "gres": "gpu:L40S:1",
                          "cpus": 8, "memory": "64G", "time": TIME, "exclude": EXCLUDED, "requeue": False}}


def current_plan():
    return make_plan(descriptor(SOURCE / "prepare.py"), descriptor(SOURCE / "run.sbatch"), descriptor(SOURCE / "derive.py"))


def contract(independent=False):
    require(Path(__file__).absolute() == SOURCE / "prepare.py" and os.getuid() == UID, "Exact staged source and UID")
    require(descriptor(SOURCE / "derive.py") == DERIVE, "Reviewed derive.py bytes required")
    plan_ref = descriptor(CONTROL / "PLAN.json")
    plan = json.loads(checked(plan_ref))
    require(plan == current_plan(), "Exact current plan required")
    review_ref = descriptor(CONTROL / "OWNER_REVIEW.json")
    review = json.loads(checked(review_ref))
    require(review.get("status") == REVIEW_STATUS and review.get("independent") is False
            and review.get("plan") == plan_ref and review.get("sources") == plan["sources"]
            and review.get("blocking_findings") == [], "Owner self-review bound to the plan required")
    independent_ref = None
    if independent:
        independent_ref = descriptor(CONTROL / "INDEPENDENT_REVIEW.json")
        value = json.loads(checked(independent_ref))
        require(value.get("status") == INDEPENDENT_STATUS and value.get("independent") is True
                and value.get("reviewer") not in (None, "", OWNER) and value.get("blocking_findings") == []
                and value.get("sources") == plan["sources"] and value.get("plan") == plan_ref,
                "Independent exact source/config PASS bound to the plan required")
    return plan_ref, plan, review_ref, independent_ref


def fields(raw):
    rows = [line for line in raw.splitlines() if line.strip()]
    require(len(rows) == 1, "One exact Slurm object required")
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", rows[0]))
    require(matches and matches[0].start() == 0, "Malformed Slurm metadata")
    result = {}
    for i, match in enumerate(matches):
        require(match[1] not in result, "Duplicate Slurm key")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(rows[0])
        result[match[1]] = rows[0][match.end():end].strip()
    return result


def query(args):
    return subprocess.check_output(args, text=True, timeout=30, env=SLURM_ENV, stdin=subprocess.DEVNULL)


@contextmanager
def admission_lock():
    fd = open_literal(LOCK, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def argv(plan):
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL", "--partition=preempt",
            "--account=cx", "--qos=preempt_qos", "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--mem=64G", "--time=" + TIME, "--exclude=" + EXCLUDED,
            "--job-name=" + JOB_NAME, "--comment=" + plan["comment"], "--chdir=" + str(HOME),
            "--input=/dev/null", "--output=/dev/null", "--error=/dev/null", str(SOURCE / "run.sbatch")]


def expected(plan, job):
    return {"JobId": job, "UserId": "luyangz(2708899)", "Account": "cx", "Partition": "preempt",
            "QOS": "preempt_qos", "JobName": JOB_NAME, "Comment": plan["comment"], "NumCPUs": "8",
            "NumTasks": "1", "TresPerNode": "gres/gpu:L40S:1", "TimeLimit": TIME, "ExcNodeList": EXCLUDED,
            "Requeue": "0", "Restarts": "0", "Command": str(SOURCE / "run.sbatch"), "WorkDir": str(HOME),
            "Dependency": "(null)", "StdIn": "/dev/null", "StdOut": "/dev/null", "StdErr": "/dev/null"}


def tres_ok(row, key):
    tres = dict(x.split("=", 1) for x in row.get(key, "").split(",") if "=" in x)
    return tres.get("cpu") == "8" and tres.get("mem") in ("64G", "65536M") and tres.get("gres/gpu") == "1"


def readback(job, plan, state):
    raw = query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = fields(raw)
    require(all(row.get(k) == v for k, v in expected(plan, job).items())
            and row.get("NumNodes") in ("1", "1-1") and shlex.split(row.get("SubmitLine", "")) == argv(plan),
            "Request identity/resources differ")
    if state == "held":
        require(row.get("JobState") == "PENDING" and row.get("Reason") == "JobHeldUser"
                and row.get("RunTime") == "00:00:00" and tres_ok(row, "ReqTRES"), "Verified held request required")
    elif state == "running":
        node = socket.gethostname().split(".")[0]
        require(row.get("JobState") == "RUNNING" and row.get("NodeList") == node and node != EXCLUDED
                and tres_ok(row, "AllocTRES"), "Actual running allocation required")
    else:
        require(row.get("JobState") in ("PENDING", "CONFIGURING", "RUNNING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin"), "Released request must be eligible")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"], timeout=30,
                                    env=SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == checked(plan["sources"]["run.sbatch"]), "Spool differs from reviewed source")
    return raw


def make_plan_file():
    private_directory(CONTROL)
    require(not (CONTROL / "PLAN.json").exists(), "Existing plan is immutable")
    require(descriptor(SOURCE / "derive.py") == DERIVE, "Reviewed derive.py bytes required")
    return seal(CONTROL / "PLAN.json", current_plan())


def owner_review(notes):
    require(notes and all(isinstance(x, str) and x for x in notes), "Concrete self-review notes required")
    plan_ref = descriptor(CONTROL / "PLAN.json")
    plan = json.loads(checked(plan_ref))
    require(plan == current_plan(), "Self-review must bind the current exact plan")
    return seal(CONTROL / "OWNER_REVIEW.json", {"status": REVIEW_STATUS, "independent": False, "reviewer": OWNER,
                                                "plan": plan_ref, "sources": plan["sources"],
                                                "blocking_findings": [], "notes": notes})


def submit():
    plan_ref, plan, review_ref, _ = contract()
    with admission_lock():
        require(contract()[:3] == (plan_ref, plan, review_ref), "Contract changed under the lock")
        require(not any((CONTROL / n).exists() for n in ("INTENT.json", "ACK.json", "AMBIGUOUS.json")),
                "Existing attempt: reconcile, never retry")
        queue = query(["/usr/bin/squeue", "-u", "luyangz", "-r", "-h", "-o", "%i|%j|%T|%q|%k"])
        rows = [line.split("|") for line in queue.splitlines() if line.strip()]
        require(all(len(r) == 5 for r in rows) and not any(r[1] == JOB_NAME or plan["function_identity"] in r[4]
                                                           for r in rows), "Live duplicate or bad census")
        limit = query(["/usr/bin/sacctmgr", "-nP", "show", "qos", "preempt_qos", "format=Name,MaxSubmitPU"])
        name, maximum = limit.strip().split("|")
        preempt = sum(r[3] == "preempt_qos" for r in rows)
        require(name == "preempt_qos" and maximum.isdecimal() and preempt < int(maximum),
                "Preempt submission limit full")
        intent = seal(CONTROL / "INTENT.json", {"owner": OWNER, "plan": plan_ref, "review": review_ref,
                                                "argv": argv(plan), "expanded_requests": len(rows),
                                                "preempt_qos_requests": preempt, "preempt_max_submit": int(maximum),
                                                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            response = subprocess.run(argv(plan), capture_output=True, text=True, timeout=45, env=SLURM_ENV,
                                      stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as error:
            seal(CONTROL / "AMBIGUOUS.json", {"intent": intent, "error": repr(error)})
            raise
        good = response.returncode == 0 and re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?\n?", response.stdout)
        seal(CONTROL / "SUBMISSION_RESULT.json", {"intent": intent, "returncode": response.returncode,
                                                   "stdout": response.stdout, "stderr": response.stderr})
        require(good, "No unambiguous ACK; reconcile")
        job = response.stdout.strip().split(";")[0]
        ack = seal(CONTROL / "ACK.json", {"owner": OWNER, "job_id": job, "intent": intent, "plan": plan_ref})
        raw = readback(job, plan, "held")
        seal(CONTROL / "HELD_READBACK.json", {"ACK": ack, "scheduler": raw})
        return {"job_id": job, "state": "HELD_VERIFIED"}


def release():
    plan_ref, plan, review_ref, independent_ref = contract(independent=True)
    ack_ref = descriptor(CONTROL / "ACK.json")
    ack = json.loads(checked(ack_ref))
    with admission_lock():
        require(contract(independent=True) == (plan_ref, plan, review_ref, independent_ref),
                "Contract changed under the lock")
        require(not (CONTROL / "RELEASE_INTENT.json").exists(), "Existing release requires reconciliation")
        readback(ack["job_id"], plan, "held")
        intent = seal(CONTROL / "RELEASE_INTENT.json", {"ACK": ack_ref, "job_id": ack["job_id"],
                                                        "independent_review": independent_ref})
        response = subprocess.run(["/usr/bin/scontrol", "release", ack["job_id"]], capture_output=True, text=True,
                                  timeout=30, env=SLURM_ENV, stdin=subprocess.DEVNULL)
        seal(CONTROL / "RELEASE_RESULT.json", {"intent": intent, "returncode": response.returncode,
                                               "stdout": response.stdout, "stderr": response.stderr})
        require(response.returncode == 0, "Release failed; no retry")
        raw = readback(ack["job_id"], plan, "released")
        seal(CONTROL / "RELEASE_READBACK.json", {"ACK": ack_ref, "scheduler": raw})
        return {"job_id": ack["job_id"], "state": fields(raw).get("JobState")}


# ---------------------------------------------------------------- in-allocation worker

def work():
    started = time.monotonic()
    plan_ref, plan, review_ref, independent_ref = contract(independent=True)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack_ref = descriptor(CONTROL / "ACK.json")
    ack = json.loads(checked(ack_ref))
    release_intent = json.loads(checked(descriptor(CONTROL / "RELEASE_INTENT.json")))
    require(ack["job_id"] == job and release_intent == {"ACK": ack_ref, "job_id": job,
                                                         "independent_review": independent_ref}
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "Actual reviewed unrestarted allocation required")
    actor = {"job_id": job, "node": socket.gethostname().split(".")[0], "pid": os.getpid(),
             "scheduler": readback(job, plan, "running")}
    seal(CONTROL / ("ACTOR_BEFORE-" + job + ".json"), actor)
    choose = native_choose(checked(NATIVE_COMMON), NATIVE_COMMON["path"],
                           checked(NATIVE_SELECTION), NATIVE_SELECTION["path"])
    derive = derive_function(checked(DERIVE), DERIVE["path"])
    v5 = load_module(checked(V5_PREPARE, 1 << 20), V5_PREPARE["path"], "tool1_pinned_v5_prepare")
    roster_v4 = json.loads(checked(ROSTER_V4))
    require(v5.POOL == POOL and v5.N == N, "v5 trainer pool constants differ")
    session4 = json.loads(checked(SESSION4_1235))
    manager, manager_rows, manager_records = {}, {}, {}
    for fraction, ref in sorted(MANAGER.items()):
        manager[fraction] = json.loads(checked(ref))
        manager_records[fraction] = manager[fraction]["records_json"]
        manager_rows[fraction] = json.loads(checked(manager_records[fraction], 1 << 30))
    datatailor = {f: json.loads(checked(ref)) for f, ref in sorted(DATATAILOR.items())}
    pool = json.loads(checked(POOL, 32 << 20))
    require(pool.get("records_json") == POOL_RECORDS, "Pool/records binding differs")
    rows = json.loads(checked(POOL_RECORDS, 1 << 30))
    loaded = time.monotonic() - started
    outputs, reports, nominal = build_all(pool, rows, choose, derive, session4, manager, manager_rows,
                                          datatailor, SPEC, POOL)
    del manager_rows
    selected_at = time.monotonic() - started
    inputs = {f: {"route": "reused-manager-native-wave", "selection": MANAGER[f], "records_json": manager_records[f]}
              for f in MANAGER}
    private_directory(OUTPUT)
    nominal_dir = OUTPUT / "budget-0.15"
    private_directory(nominal_dir)
    nominal_ref = seal(nominal_dir / "NOMINAL_INPUT_SELECTION.json", {
        "schema": "tool1-native-uniform-nominal-input-v1", "method": "uniform", "selection_seed": SEED,
        "fraction": "0.15", "count": SPEC["budgets"]["0.15"]["nominal_K"], "record_ids": nominal["0.15"],
        "pool": POOL, "algorithm_sources": {"native_common": NATIVE_COMMON, "native_selection": NATIVE_SELECTION},
        "producer_job": job, "records_not_materialized": True, "not_a_training_input": True})
    inputs["0.15"] = {"route": "regenerated-native-in-allocation", "selection": nominal_ref, "records_json": None}
    summary = {}
    for fraction in sorted(outputs):
        value, report = outputs[fraction], reports[fraction]
        folder = OUTPUT / ("budget-" + fraction)
        private_directory(folder)
        records = seal(folder / "records.json", canonical(value["rows"]) + b"\n")
        raw_records, mount = nas_readback(records, 1 << 30)
        reread = json.loads(raw_records)
        del raw_records
        require(reread == value["rows"] and len(reread) == SPEC["budgets"][fraction]["K"], "Records readback differs")
        document = selection_document(fraction, value["record_ids"], records, report, plan, job, inputs)
        selection = seal(folder / "SELECTION.json", document)
        raw_selection, selection_mount = nas_readback(selection, 32 << 20)
        require(json.loads(raw_selection) == document, "Selection readback differs")
        cells = candidate_cells(roster_v4, fraction, selection, records)
        v5.validate_roster({"schema": v5.ROSTER_SCHEMA, "cells": cells})
        v5.SPEC = cells[sorted(cells)[0]]
        v5_check = v5.validate_selection(pool, rows, json.loads(raw_selection), reread)
        v5.SPEC = None
        require(v5_check["canonical_selected_rows_sha256"] == records["sha256"] and v5_check["K"] == report["K"],
                "v5 canonical selected rows differ from the sealed records")
        del reread
        result = {"status": "PASS_TOOL1_UNIFORM_MATCHEDK_BUDGET_BUILT", "fraction": fraction, "job_id": job,
                  "records": records, "selection": selection, "mount": mount, "selection_mount": selection_mount,
                  "report": report, "nominal_input": inputs[fraction], "v5_validate_selection": v5_check,
                  "v5_validate_roster_candidate_cells": sorted(cells), "plan": plan_ref}
        result_ref = seal(folder / "RESULT.json", result)
        summary[fraction] = {"records": records, "selection": selection, "result": result_ref, "K": report["K"],
                             "roster_fields": {"K": report["K"], "method": "uniform", "selection_seed": SEED,
                                               "selection_kind": "group", "selection_status": None,
                                               "records": records, "selection": selection},
                             "candidate_cells": cells, "v5_validate_selection": v5_check, "report": report}
    actor_after = {"job_id": job, "scheduler": readback(job, plan, "running")}
    seal(CONTROL / ("RESULT-" + job + ".json"), {
        "status": "PASS_TOOL1_UNIFORM_MATCHEDK_SELECTIONS_BUILT_ALL_THREE_BUDGETS", "plan": plan_ref,
        "review": review_ref, "independent_review": independent_ref, "budgets": summary, "actor": actor,
        "actor_after": actor_after, "seconds": {"load": round(loaded, 1), "select": round(selected_at, 1),
                                                "total": round(time.monotonic() - started, 1)},
        "benchmark_QA_reads": 0, "model_forwards": 0, "downloads": 0,
        "caveats": ["The allocated L40S is idle (Babel CPU-inside-GPU rule); count the full allocation in all-in cost.",
                    "5%/20% outputs are the manager-native-wave sel1235 sets minus 6/5 records; they nearly duplicate "
                    "the manager's sel1235 training inputs (different trainer route).",
                    "One selection draw per budget; seed-1235 nominal sets are nested across 5/10/15/20%.",
                    "Alias groups by original LLaVA id are audited, not enforced (native Uniform is per conversation)."]})
    print(json.dumps({"job_id": job, "selections": {k: v["selection"]["sha256"] for k, v in summary.items()},
                      "records": {k: v["records"]["sha256"] for k, v in summary.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "worker"))
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    if args.command == "worker":
        work()
    elif args.command == "plan":
        print(json.dumps(make_plan_file()))
    elif args.command == "self-review":
        print(json.dumps(owner_review(args.note)))
    else:
        print(json.dumps({"submit": submit, "release": release}[args.command]()))


if __name__ == "__main__":
    main()
