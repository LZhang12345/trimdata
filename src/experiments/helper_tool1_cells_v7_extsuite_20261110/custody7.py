"""CELLS_V7 custody: custody6.py (CELLS_V6, independently reviewed 2026-11-09) with the version identity changed
(roster schema, inventory control/output roots, owner, schemas) and one allow-list extension: kind tool1_v10_original
may name ROSTER_V7.json or ROSTER_V8.json (both Tool1 trainer v10 rosters; ROSTER_V8 registers Tool2's 5% cells).
Each cell still binds exactly one pinned roster whose entry must equal its producer PLAN; every other kind rule is
unchanged. CELLS_V7 registers the two trainer-v10 original producers t2-m1ur-b05 and t2-univn-b05 (ROSTER_V8; Tool2
request, 2026-11-10). CELLS_V6 text follows.

CELLS_V6 custody: custody5.py (CELLS_V5, independently reviewed 2026-11-08) with only the version identity changed
(roster schema, inventory control/output roots, owner, schemas); every kind rule is unchanged. CELLS_V6 registers the
single trainer-v10 original producer t2-m1ur-b20 (Tool2 request, 2026-11-09). CELLS_V5 text follows.

CELLS_V5 custody (custody-only successor of custody3.py/custody4.py) for the extended-benchmark view of Tool1's
budget-curve producers: one module binding four producer kinds by their own sealed records.

Kinds (each cell names exactly one; nothing is borrowed across kinds):
  * tool1_v6_original   Tool1 trainer v6 original job sealed the complete epoch (custody3 logic; PLAN.json schema
                        tool1-baseline-native-preempt-v6, ROSTER_V6).
  * tool1_v8_successor  Tool1 trainer v8 General successor-<n> sealed the epoch (custody4 logic: PLAN-v8.json,
                        successor-<n>/ACK/FUNCTION_CLAIM/HELD_READBACK, trajectory chain original -> withdrawn
                        predecessor -> producer).
  * tool1_v10_original  Tool1 trainer v10 original job (ROSTER_V7/ROSTER_V8, Tool2 selection trained by Tool1's trainer;
                        custody3 logic with PLAN.json schema tool1-baseline-native-general-original-v10).
  * tool3_chain_v3      Tool3 chain-v3 RESULT; the accept rule is t3coin v2's
                        (helper_tool1_baseline_eval_t3coin_v2_20261106: prepare.py tool3_receipts / frozen_registry /
                        make_plan link-config checks / validate_attestation, model.py validate_producer), conditions
                        copied verbatim onto already-read documents, plus the pinned, independently reviewed and
                        executed t3coin v2 PLAN whose producer custody must equal this one.
For every kind CELLS_V7 pins the producer RESULT.json (size+sha256, hashed on a GPU node before registration) and the
sealed completion record (Tool1 TRAINING_RESULT / Tool3 RESULT_SUCCESS LINK_EXIT). The cells-v7 inventory re-hashes
the RESULT and the four inference files on-node and loads the model for its parameter digest; evaluators late-bind
that PASS. The function names and signatures used by the byte-identical v3 workers and inventory child (identity3.py,
model.py, model6.py, model4.py: worker_custody, producer_files, identity_path, model_identity, INFERENCE_FILES,
INVENTORY_OUTPUT) equal custody3.py; plan-time reading goes through plan_custody(), which receives the caller's own
exact readers (read(ref)->bytes, describe(path)->ref, listdir(path)->names). Pure stdlib otherwise.
"""

import hashlib
import json
import math
from pathlib import PurePosixPath
import re


HOME = "/home/luyangz/data_selection_multimodal"
CELLS_SCHEMA = "tool1-baseline-cells-v7-extsuite-custody-v1"
TRAIN_CONTROL = HOME + "/collaboration/helper-tool1-baseline-independent-20261101/baseline-train-v5"
CHECKPOINT_ROOT = "/data/hf_cache/project_checkpoints/DS-MultiModal/helper-tool1-baseline-train-v4-20261102"
INVENTORY_CONTROL = HOME + "/collaboration/helper-tool1-baseline-independent-20261101/cells-v7-inventory-v1"
INVENTORY_OUTPUT = "/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-cells-v7-inventory-v1-20261110"
INVENTORY_OWNER = "Tool1-owned-CELLS-V7-custody-and-parameter-identity-inventory-v1"
INVENTORY_SCHEMA = "tool1-cells-v7-custody-inventory-result-v1"
INVENTORY_STATUS = "COMPLETE_CELLS_V7_CUSTODY_INVENTORY_V1"
IDENTITY_SCHEMA = "tool1-cells-v7-model-identity-v1"
KINDS = ("tool1_v6_original", "tool1_v8_successor", "tool1_v10_original", "tool3_chain_v3")
TOOL1_KINDS = KINDS[:3]
PLAN_SCHEMAS = {"tool1_v6_original": "tool1-baseline-native-preempt-v6",
                "tool1_v8_successor": "tool1-baseline-native-general-successor-v8",
                "tool1_v10_original": "tool1-baseline-native-general-original-v10"}
KIND_ROSTERS = {"tool1_v6_original": ("ROSTER_V6.json",), "tool1_v8_successor": ("ROSTER_V6.json",),
                "tool1_v10_original": ("ROSTER_V7.json", "ROSTER_V8.json")}
ROSTER_NAMES = ("ROSTER_V6.json", "ROSTER_V7.json", "ROSTER_V8.json")
SUCCESSOR_PLAN = "PLAN-v8.json"
ORIGINAL_RECORDS = ("PLAN", "ACK")
PRODUCER_OWNER = "Tool1-owned-published-baseline-native-training"
PRODUCER_STATUS = "ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED"
PRODUCER_ROLE = "TOOL1_BASELINE_FULL_EPOCH_NOT_EVALUATED"
TRAINING_RESULT_SCHEMA = "tool1-baseline-native-training-result-v5"
PRODUCER_MODEL_SHA = "b53089293b49a5a2d9fb8d433999105d06c70629f3be788bbf35d3089a728b11"
INFERENCE_FILES = ("adapter_config.json", "adapter_model.safetensors", "config.json", "non_lora_trainables.bin")
RECORD_NAMES = ("PLAN", "ACK", "FUNCTION_CLAIM", "HELD_READBACK")
# Tool3 chain-v3 producers: constants of the t3coin v2 accept rule (prepare.py/model.py), unchanged.
TOOL3_CONTROL = HOME + "/collaboration/helper-tool3-baselines-20261104"
TOOL3_TRAINING = TOOL3_CONTROL + "/training"
TOOL3_REGISTRY = TOOL3_CONTROL + "/CELLS.json"
TOOL3_CHECKPOINT_ROOT = "/data/hf_cache/project_checkpoints/DS-MultiModal/tool3-baselines-20261104"
T3COIN_CONTROL = HOME + "/collaboration/helper-tool1-baseline-independent-20261101/t3-coincide-budget-eval-v2"
T3COIN_PREPARE = {"path": HOME + "/experiments/helper_tool1_baseline_eval_t3coin_v2_20261106/prepare.py",
                  "bytes": 48064, "sha256": "314b91ec36cbc92a9d67e41e317af85d328c46f7196b110a9a09a814adefcc6f"}
T3COIN_INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_T3COIN_EVAL_V2"
T3_PRODUCER_OWNER = "Tool3-owned-baseline-native-training"
T3_PRODUCER_MODEL_SHA = "b0c099075001ee60a2547e7c3a72e917fddf0f657d42fbb9b2dd7731ff14814a"
T3_PRODUCER_ROLE = "TOOL3_BASELINE_RESUMABLE_FULL_EPOCH_NOT_EVALUATED"
T3_ATTESTATION_SCHEMA = "tool1-t3coin-producer-custody-attestation-v2"
T3_AUDIT_CHECKS = ("chunks_tile_all_steps", "deterministic_attention_executed", "exposure_exact_once", "final_step",
                   "microbatches")
T3_REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors", "config.json", "non_lora_trainables.bin")
T3_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
T3_ROUTE_AMENDMENT = {"path": TOOL3_CONTROL + "/CHAIN_AMENDMENT_V3_2.json", "bytes": 1224,
                      "sha256": "be38d2a21cf3b5d0b7d80b4ee5f78c6a9cb2d55113addd32f878f6f0121319b7"}
T3_TRAINABLE_REFERENCE = {
    "path": HOME + "/collaboration/helper-tool1-baseline-independent-20261101/baseline-eval-v4/"
                   "b6-uniform-mk1235-b05-k33032-train43/attempt-0/MODEL_CONFIG-11068026.json",
    "bytes": 734587, "sha256": "6ce0144f4b8bd44fcd87ae1cf907e1f53ffbaa5782d948d11bb1d2705b6b6054"}
T3_RECORD_NAMES = ("REGISTRY", "LINK_EXIT", "MODEL_CONFIG", "SUBMIT_ACK", "ROUTE_AMENDMENT", "ATTESTATION",
                   "TRAINABLE_REFERENCE", "T3COIN_PLAN", "T3COIN_INDEPENDENT_REVIEW")
CELL = re.compile(r"[a-z0-9][a-z0-9-]{2,60}")
SHORT = re.compile(r"[a-z0-9]{3,10}")
SHA = re.compile(r"[0-9a-f]{64}")
JOB = re.compile(r"[1-9][0-9]*")


class CustodyError(ValueError):
    pass


def require(value, message):
    if not value:
        raise CustodyError(message)


def is_ref(value):
    return (isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}
            and isinstance(value["path"], str) and value["path"].startswith("/")
            and type(value["bytes"]) is int and value["bytes"] >= 0
            and isinstance(value["sha256"], str) and SHA.fullmatch(value["sha256"]) is not None)


def plain(ref):
    return {"path": ref["path"], "bytes": ref["bytes"], "sha256": ref["sha256"]}


def canonical(value):
    """Identical to the bound metadata helper's canonical() (t3coin v2 config_sha256)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def valid_singleton_node(node):
    """Identical to the bound metadata helper's valid_singleton_node()."""
    if not isinstance(node, str) or len(node) > 64:
        return False
    match = re.fullmatch(r"babel-([a-z])([0-9]+)-([0-9]+)", node)
    return (match is not None and not (match[1] == "p"
            and match[2].lstrip("0") == "9" and match[3].lstrip("0") == "20"))


def steps_for(k):
    require(type(k) is int and 1024 <= k <= 660763, "Bounded selected-conversation budget K required")
    steps = (k + 3) // 4 // 32
    require(steps >= 2, "At least two optimizer updates required")
    return steps


# ------------------------------------------------------------------------------------------------ registration
def tool1_record_path(cell, spec, name):
    if spec["producer_kind"] == "tool1_v8_successor":
        if name == "PLAN":
            return "%s/%s/%s" % (TRAIN_CONTROL, cell, SUCCESSOR_PLAN)
        return "%s/%s/successor-%d/%s.json" % (TRAIN_CONTROL, cell, spec["successor"], name)
    return "%s/%s/%s.json" % (TRAIN_CONTROL, cell, name)


def tool3_record_paths(cell, spec):
    job, link = spec["producer_job"], spec["link_id"]
    folder = TOOL3_TRAINING + "/" + cell
    return {"REGISTRY": TOOL3_REGISTRY, "LINK_EXIT": "%s/LINK_EXIT-%s.json" % (folder, link),
            "MODEL_CONFIG": "%s/MODEL_CONFIG-%s.json" % (folder, link),
            "SUBMIT_ACK": "%s/SUBMIT_ACK-%s.json" % (folder, job), "ROUTE_AMENDMENT": T3_ROUTE_AMENDMENT["path"],
            "ATTESTATION": None, "TRAINABLE_REFERENCE": T3_TRAINABLE_REFERENCE["path"],
            "T3COIN_PLAN": "%s/%s/PLAN.json" % (T3COIN_CONTROL, cell),
            "T3COIN_INDEPENDENT_REVIEW": "%s/INDEPENDENT_REVIEW-%s.json" % (T3COIN_CONTROL, cell)}


def check_trajectory(cell, spec):
    """custody4: successor index, accepted-trajectory chain (original owner .. withdrawn predecessor .. producer)."""
    successor, trajectory = spec.get("successor"), spec.get("trajectory")
    require(type(successor) is int and 1 <= successor <= 8 and isinstance(trajectory, dict)
            and set(trajectory) == {"owner_job", "chain", "original_records"}, "Successor trajectory required: " + cell)
    chain, original = trajectory["chain"], trajectory["original_records"]
    require(isinstance(chain, list) and len(chain) == successor + 1
            and all(isinstance(link, list) and len(link) == 2 and link[0] == index
                    and isinstance(link[1], str) and JOB.fullmatch(link[1]) is not None
                    for index, link in enumerate(chain))
            and len({link[1] for link in chain}) == len(chain)
            and chain[0][1] == trajectory["owner_job"] and chain[-1][1] == spec.get("producer_job"),
            "Exact trajectory chain ending at the producer job required: " + cell)
    require(isinstance(original, dict) and set(original) == set(ORIGINAL_RECORDS)
            and all(is_ref(original[name]) and original[name]["path"] == "%s/%s/%s.json" % (TRAIN_CONTROL, cell, name)
                    for name in ORIGINAL_RECORDS), "Pinned original trainer PLAN/ACK required: " + cell)
    return successor, trajectory


def check_cells(cells):
    """The whole registered roster: well-formed per kind, unique shorts, horizon derived from K, no eligibility."""
    require(isinstance(cells, dict) and cells.get("schema") == CELLS_SCHEMA and isinstance(cells.get("cells"), dict)
            and cells["cells"] and isinstance(cells.get("rosters"), dict), "Exact CELLS_V7 roster required")
    for name, ref in cells["rosters"].items():
        require(name in ROSTER_NAMES and is_ref(ref)
                and ref["path"] == TRAIN_CONTROL + "/" + name, "Pinned Tool1 trainer roster descriptor required")
    shorts = set()
    for cell, spec in cells["cells"].items():
        require(CELL.fullmatch(cell) is not None and isinstance(spec, dict) and spec.get("producer_kind") in KINDS,
                "Literal registered cell with a known producer kind required: " + str(cell))
        kind = spec["producer_kind"]
        require(spec.get("kind") == "student" and SHORT.fullmatch(str(spec.get("short"))) is not None
                and spec["short"] not in shorts and spec.get("training_seed") in (17, 29, 43)
                and spec.get("steps") == steps_for(spec.get("K"))
                and isinstance(spec.get("producer_job"), str) and JOB.fullmatch(spec["producer_job"]) is not None
                and isinstance(spec.get("method_key"), str) and isinstance(spec.get("method"), str)
                and "plan_status" not in spec and "eligible" not in spec
                and is_ref(spec.get("training_result")) and is_ref(spec.get("producer_result")),
                "Registered CELLS_V7 cell is malformed: " + cell)
        records = spec.get("trainer_records")
        if kind in TOOL1_KINDS:
            require(spec.get("roster_cell") == cell and spec.get("roster") in KIND_ROSTERS[kind]
                    and spec["roster"] in cells["rosters"] and type(spec.get("selection_seed")) is int
                    and SHA.fullmatch(str(spec.get("trainer_prepare_sha256"))) is not None
                    and isinstance(spec.get("producer_job_name"), str)
                    and spec["producer_result"]["path"] == CHECKPOINT_ROOT + "/" + cell + "/RESULT.json"
                    and re.fullmatch(re.escape("%s/%s/TRAINING_RESULT-%s-r" % (TRAIN_CONTROL, cell, spec["producer_job"]))
                                     + r"(0|[1-9][0-9]{0,2})\.json", spec["training_result"]["path"]) is not None,
                    "Registered Tool1-trainer cell is malformed: " + cell)
            if kind == "tool1_v8_successor":
                check_trajectory(cell, spec)
            else:
                require("successor" not in spec and "trajectory" not in spec, "Original-job cell has no successor")
            require(isinstance(records, dict) and set(records) == set(RECORD_NAMES)
                    and all(is_ref(records[name]) and records[name]["path"] == tool1_record_path(cell, spec, name)
                            for name in RECORD_NAMES), "Registered Tool1 trainer records are malformed: " + cell)
        else:
            paths = tool3_record_paths(cell, spec) if isinstance(spec.get("link_id"), str) else {}
            require(isinstance(spec.get("link_id"), str)
                    and re.fullmatch(re.escape(spec["producer_job"]) + r"-r[0-9]+", spec["link_id"]) is not None
                    and spec.get("selection_seed") is None and "roster" not in spec
                    and isinstance(spec.get("tool3_short"), str)
                    and spec["producer_result"]["path"] == TOOL3_CHECKPOINT_ROOT + "/" + cell + "/RESULT.json"
                    and spec["training_result"]["path"] == paths["LINK_EXIT"]
                    and isinstance(records, dict) and set(records) == set(T3_RECORD_NAMES)
                    and all(is_ref(records[name]) for name in T3_RECORD_NAMES)
                    and all(paths[name] is None or records[name]["path"] == paths[name] for name in T3_RECORD_NAMES)
                    and records["LINK_EXIT"] == spec["training_result"]
                    and plain(records["ROUTE_AMENDMENT"]) == T3_ROUTE_AMENDMENT
                    and plain(records["TRAINABLE_REFERENCE"]) == T3_TRAINABLE_REFERENCE
                    and PurePosixPath(records["ATTESTATION"]["path"]).parent == PurePosixPath(T3COIN_CONTROL) / cell
                    and re.fullmatch(r"ATTESTATION-[0-9TZ]+\.json", PurePosixPath(records["ATTESTATION"]["path"]).name)
                    is not None, "Registered Tool3 chain-v3 cell is malformed: " + cell)
        shorts.add(spec["short"])
    return cells


def cell_spec(cells, cell):
    check_cells(cells)
    require(isinstance(cell, str) and cell in cells["cells"], "Cell is not registered in CELLS_V7")
    return cells["cells"][cell]


def v7_custody(spec):
    """The CELLS_V7-only part of a custody record (carried into plans, the inventory gate and the worker)."""
    value = dict(producer_kind=spec["producer_kind"], producer_result=spec["producer_result"],
                 training_result=spec["training_result"], cells_schema=CELLS_SCHEMA)
    if spec["producer_kind"] == "tool1_v8_successor":
        value.update(successor=spec["successor"], trajectory=spec["trajectory"])
    if spec["producer_kind"] == "tool3_chain_v3":
        value.update(link_id=spec["link_id"])
    return value


def producer_descriptor(spec, restart_count):
    cell = spec["roster_cell"] if spec["producer_kind"] in TOOL1_KINDS else spec["producer_result"]["path"].split(
        "/")[-2]
    if spec["producer_kind"] in TOOL1_KINDS:
        return dict(cell=cell, job_id=spec["producer_job"], K=spec["K"], steps=spec["steps"],
                    training_seed=spec["training_seed"], selection_seed=spec["selection_seed"],
                    method=spec["method_key"], restart_count=restart_count, model_source_sha256=PRODUCER_MODEL_SHA,
                    model_result=CHECKPOINT_ROOT + "/" + cell + "/RESULT.json",
                    training_dir=CHECKPOINT_ROOT + "/" + cell + "/training", producer_kind=spec["producer_kind"])
    return dict(cell=cell, job_id=spec["producer_job"], K=spec["K"], steps=spec["steps"],
                training_seed=spec["training_seed"], selection_seed=None, method=spec["method_key"],
                restart_count=restart_count, model_source_sha256=T3_PRODUCER_MODEL_SHA,
                model_result=TOOL3_CHECKPOINT_ROOT + "/" + cell + "/RESULT.json",
                training_dir=TOOL3_CHECKPOINT_ROOT + "/" + cell + "/training", producer_kind=spec["producer_kind"],
                link_id=spec["link_id"])


def training_result_name(spec, restart_count):
    return "TRAINING_RESULT-%s-r%d.json" % (spec["producer_job"], restart_count)


# ------------------------------------------------------------------------------------------------ plan time
def plan_custody(cells, cell, read, describe, listdir):
    """Plan-time custody through the caller's exact readers: every pinned record is re-read by its pin, the
    completion record must be the only one of its kind, then trainer_custody() validates the documents."""
    spec = cell_spec(cells, cell)
    refs = spec["trainer_records"]
    records = {name: json.loads(read(refs[name])) for name in sorted(refs)}
    if spec["producer_kind"] in TOOL1_KINDS:
        roster = json.loads(read(cells["rosters"][spec["roster"]]))
        if spec["producer_kind"] == "tool1_v8_successor":
            for ref in spec["trajectory"]["original_records"].values():
                read(ref)
        folder = TRAIN_CONTROL + "/" + cell
        names = sorted(name for name in listdir(folder) if name.startswith("TRAINING_RESULT-"))
        require(len(names) == 1 and folder + "/" + names[0] == spec["training_result"]["path"],
                "Exactly one sealed TRAINING_RESULT (the pinned one) required: " + cell)
    else:
        roster = None
        folder = TOOL3_TRAINING + "/" + cell
        success = []
        for name in sorted(listdir(folder)):
            if re.fullmatch(r"LINK_EXIT-([1-9][0-9]*)-r([0-9]+)\.json", name) is None:
                continue
            ref = describe(folder + "/" + name)
            if json.loads(read(ref)).get("status") == "RESULT_SUCCESS":
                success.append(ref)
        require(success == [spec["training_result"]],
                "Exactly one Tool3 RESULT_SUCCESS link (the pinned one) required: " + cell)
    result_ref = describe(spec["training_result"]["path"])
    require(result_ref == spec["training_result"], "Pinned completion record changed: " + cell)
    return trainer_custody(cells, cell, records, result_ref, json.loads(read(result_ref)), roster)


def tool1_trainer_custody(cells, cell, spec, records, training_result_ref, training_result, roster):
    """custody3.trainer_custody + per-kind plan schema + custody4 successor checks, unchanged conditions."""
    require(isinstance(records, dict) and set(records) == set(RECORD_NAMES), "All four trainer records required")
    plan, ack, claim, held = (records[name] for name in RECORD_NAMES)
    refs = spec["trainer_records"]
    roster_spec = roster.get("cells", {}).get(cell) if isinstance(roster, dict) else None
    require(isinstance(roster_spec, dict) and roster.get("schema") == "tool1-baseline-train-roster-v1"
            and roster_spec.get("kind") == "production" and roster_spec.get("K") == spec["K"]
            and roster_spec.get("training_seed") == spec["training_seed"]
            and roster_spec.get("selection_seed") == spec["selection_seed"]
            and roster_spec.get("method") == spec["method_key"]
            and roster_spec.get("job_name") == spec["producer_job_name"], "Pinned trainer roster entry differs")
    require(ack.get("cell") == cell and ack.get("plan") == refs["PLAN"] and ack.get("owner") == PRODUCER_OWNER
            and ack.get("job_id") == spec["producer_job"] and held.get("job_id") == ack["job_id"]
            and held.get("ACK") == refs["ACK"]
            and claim.get("function_identity") == plan.get("function_identity") and claim.get("cell") == cell
            and claim.get("owner") == plan.get("owner") == PRODUCER_OWNER and plan.get("cell") == cell
            and plan.get("job_name") == spec["producer_job_name"]
            and plan.get("schema") == PLAN_SCHEMAS[spec["producer_kind"]]
            and isinstance(plan.get("function"), dict) and plan["function"].get("kind") == "production"
            and plan["function"].get("roster") == cells["rosters"][spec["roster"]]
            and plan["function"].get("spec") == roster_spec
            and plan.get("sources", {}).get("model.py", {}).get("sha256") == PRODUCER_MODEL_SHA
            and plan.get("sources", {}).get("prepare.py", {}).get("sha256") == spec["trainer_prepare_sha256"],
            "Exact actual roster-bound Tool1 producer records required; no borrowed custody")
    restart = training_result.get("restart_count") if isinstance(training_result, dict) else None
    require(type(restart) is int and 0 <= restart <= 200 and is_ref(training_result_ref)
            and training_result_ref == spec["training_result"]
            and training_result_ref["path"] == "%s/%s/%s" % (TRAIN_CONTROL, cell, training_result_name(spec, restart)),
            "The trainer's own pinned TRAINING_RESULT-<job>-r<N>.json is required")
    selected = training_result.get("selected", {})
    require(training_result.get("schema") == TRAINING_RESULT_SCHEMA
            and training_result.get("status") == PRODUCER_STATUS and training_result.get("kind") == "production"
            and training_result.get("cell") == cell and training_result.get("job_id") == spec["producer_job"]
            and training_result.get("plan") == refs["PLAN"]
            and training_result.get("function_identity") == plan["function_identity"]
            and training_result.get("model_result") == CHECKPOINT_ROOT + "/" + cell + "/RESULT.json"
            and isinstance(selected, dict) and selected.get("K") == spec["K"],
            "Complete production epoch TRAINING_RESULT bound to the pinned PLAN required")
    if spec["producer_kind"] == "tool1_v8_successor":
        successor, trajectory = spec["successor"], spec["trajectory"]
        original = trajectory["original_records"]
        require(plan.get("original_plan") == original["PLAN"] and plan.get("original_ack") == original["ACK"]
                and plan.get("trajectory_owner_job") == trajectory["owner_job"]
                and ack.get("successor") == successor and ack.get("predecessor_job") == trajectory["chain"][-2][1]
                and claim.get("successor") == successor and claim.get("plan") == refs["PLAN"]
                and claim.get("trajectory_owner_job") == trajectory["owner_job"]
                and training_result.get("successor") == successor
                and training_result.get("predecessor_chain") == trajectory["chain"]
                and training_result.get("trajectory_owner_job") == trajectory["owner_job"],
                "Exact CELLS_V7 successor trajectory records and pinned TRAINING_RESULT required")
    else:
        require(training_result.get("successor") is None and training_result.get("predecessor_chain") is None
                and "successor" not in ack, "Original-job custody must not be a successor trajectory")
    return restart


def tool3_accept(cell, spec, records, link_exit_ref, link_exit):
    """t3coin v2's Tool3 accept rule on already-read documents; the require() conditions are verbatim copies of
    prepare.py tool3_receipts/frozen_registry/make_plan/validate_attestation (p.* helpers -> identical local ones),
    then the executed, independently reviewed t3coin v2 PLAN must bind exactly the same producer custody."""
    refs = spec["trainer_records"]
    job, link, value = spec["producer_job"], spec["link_id"], link_exit
    chunk = value.get("chunk_record") or {}
    # tool3_receipts (unique RESULT_SUCCESS link checked by plan_custody's listing)
    require(value.get("status") == "RESULT_SUCCESS" and link_exit_ref == refs["LINK_EXIT"],
            "Pinned Tool3 RESULT_SUCCESS LINK_EXIT required")
    require(value.get("returncode") == 0 and value.get("result_exists") is True and value.get("progress") is True
            and chunk.get("status") == "FINAL_CHUNK" and chunk.get("job_id") == job and chunk.get("link_id") == link
            and chunk.get("end_step") == chunk.get("max_steps") == spec["steps"]
            and spec["steps"] in (chunk.get("saved_steps") or []),
            "Tool3 RESULT_SUCCESS link must end the full natural horizon (end_step == max_steps == steps)")
    ack = records["SUBMIT_ACK"]
    require(ack.get("job_id") == job and ack.get("returncode") == 0
            and ack.get("stdout", "").split(";")[0].strip() == job, "Tool3 SUBMIT_ACK of the RESULT job required")
    # frozen_registry
    registry = records["REGISTRY"]
    require(registry.get("schema") == "tool3-baseline-cells-v1" and cell in registry.get("cells", {}),
            "Tool3 registry entry required")
    entry = registry["cells"][cell]
    model_config = records["MODEL_CONFIG"]
    # make_plan link-config checks
    require(entry.get("K") == spec["K"] and entry.get("training_seed") == spec["training_seed"]
            and entry.get("short") == spec["tool3_short"] and entry.get("route") in ("general", "general_budget")
            and dict(model_config.get("registry_entry") or {}, route=None) == dict(entry, route=None)
            and model_config.get("cell") == cell and model_config.get("job_id") == job
            and model_config.get("link_id") == link and model_config.get("K") == spec["K"]
            and model_config.get("training_seed") == spec["training_seed"]
            and model_config.get("role") == T3_PRODUCER_ROLE and model_config.get("input") == entry["records"]
            and model_config.get("model_source", {}).get("sha256") == T3_PRODUCER_MODEL_SHA
            and model_config.get("output") == TOOL3_CHECKPOINT_ROOT + "/" + cell,
            "Tool3 link config must be the registered full-epoch cell (route is the only registry change)")
    reference = records["TRAINABLE_REFERENCE"]["qualification"]
    # validate_attestation
    attestation_ref = refs["ATTESTATION"]
    value = records["ATTESTATION"]
    listed = value.get("output_files_listed")
    adapter = value.get("adapter_config") or {}
    require(PurePosixPath(attestation_ref["path"]).parent == PurePosixPath(T3COIN_CONTROL) / cell
            and re.fullmatch(r"ATTESTATION-[0-9TZ]+\.json", PurePosixPath(attestation_ref["path"]).name) is not None
            and value.get("schema") == T3_ATTESTATION_SCHEMA and value.get("cell") == cell
            and value.get("read_only") is True and value.get("uid") == 2708899
            and value.get("controller") == T3COIN_PREPARE
            and valid_singleton_node(value.get("node"))
            and value["result"]["path"] == TOOL3_CHECKPOINT_ROOT + "/" + cell + "/RESULT.json"
            and value["result"]["mount"]["target"] == "/data/hf_cache"
            and value.get("status") == PRODUCER_STATUS and value.get("job_id") == job
            and value.get("link_id") == link and value.get("full_epoch_training_complete") is True
            and value.get("audit_checks") == {name: True for name in T3_AUDIT_CHECKS}
            and value.get("config_sha256") == hashlib.sha256(canonical(model_config)).hexdigest()
            and isinstance(listed, dict) and set(T3_REQUIRED_FILES) <= set(listed)
            and value.get("output_files_rehashed") == listed
            and value.get("training_entries") == sorted(listed),
            "Exact read-only GPU-node attestation of the Tool3 RESULT and every listed output file required")
    require(adapter.get("r") == 128 and adapter.get("lora_alpha") == 256 and adapter.get("lora_dropout") == .05
            and adapter.get("bias") == "none" and adapter.get("peft_type") == "LORA"
            and adapter.get("task_type") == "CAUSAL_LM" and set(adapter.get("target_modules") or []) == T3_TARGET_MODULES
            and len(adapter["target_modules"]) == len(T3_TARGET_MODULES)
            and adapter.get("base_model_name_or_path") == model_config["assets"]["vicuna"]["snapshot"],
            "Native LoRA configuration of the Tool1 recipe required (target-module order is not semantic)")
    require(listed["config.json"] == reference["output_files"]["config.json"]
            and all(model_config["assets"][role] == reference["config"]["assets"][role]
                    for role in ("vicuna", "clip336", "stage1_projector")),
            "Tool3 frozen base assets and saved model config.json must equal the Tool1 b6 reference")
    # CELLS_V7 pin and the executed t3coin v2 plan/review of the same producer
    require({k: value["result"][k] for k in ("path", "bytes", "sha256")} == spec["producer_result"],
            "Attested Tool3 RESULT.json differs from the CELLS_V7 pin")
    t3plan, review = records["T3COIN_PLAN"], records["T3COIN_INDEPENDENT_REVIEW"]
    custody = (t3plan.get("function") or {}).get("producer_custody") or {}
    producer = (t3plan.get("function") or {}).get("producer") or {}
    require(t3plan.get("schema") == "tool1-t3coin-budget-evaluations-v2" and t3plan.get("cell") == cell
            and t3plan.get("sources", {}).get("prepare.py") == T3COIN_PREPARE
            and producer == dict(cell=cell, job_id=job, K=spec["K"], training_seed=spec["training_seed"],
                                 steps=spec["steps"], model_source_sha256=T3_PRODUCER_MODEL_SHA)
            and custody.get("owner") == T3_PRODUCER_OWNER and custody.get("read_only") is True
            and custody.get("registry") == refs["REGISTRY"] and custody.get("registry_entry") == entry
            and custody.get("route_amendment") == T3_ROUTE_AMENDMENT and custody.get("link_exit") == refs["LINK_EXIT"]
            and custody.get("link_id") == link and custody.get("model_config") == refs["MODEL_CONFIG"]
            and custody.get("submit_ack") == refs["SUBMIT_ACK"] and custody.get("attestation") == attestation_ref
            and custody.get("result") == spec["producer_result"] and custody.get("output_files") == listed
            and review.get("status") == T3COIN_INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("blocking_findings") == [] and review.get("cell") == cell
            and review.get("plans") == {cell: refs["T3COIN_PLAN"]}
            and review.get("function_identity") == t3plan.get("function_identity"),
            "The executed, independently reviewed t3coin v2 plan must bind exactly this Tool3 custody")
    return entry


def trainer_custody(cells, cell, records, training_result_ref, training_result, roster):
    """Plan-time custody from records already read byte-exactly against CELLS_V7 pins (see plan_custody)."""
    spec = cell_spec(cells, cell)
    if spec["producer_kind"] in TOOL1_KINDS:
        restart = tool1_trainer_custody(cells, cell, spec, records, training_result_ref, training_result, roster)
    else:
        require(isinstance(records, dict) and set(records) == set(T3_RECORD_NAMES) and roster is None,
                "All Tool3 accept-rule records required")
        tool3_accept(cell, spec, records, training_result_ref, training_result)
        restart = 0
    return dict(kind="student", cell=cell, producer=producer_descriptor(spec, restart),
                trainer_records=dict(spec["trainer_records"]), training_result=training_result_ref,
                v7_custody=v7_custody(spec))


# ------------------------------------------------------------------------------------------------ on-node producer
def tool3_producer_check(result, producer):
    """t3coin v2 model.py validate_producer conditions (qualification = the Tool3 RESULT), unchanged."""
    source = result["config"]
    entry = source["registry_entry"]
    steps = producer["steps"]
    require(result["job_id"] == source["job_id"] == producer["job_id"]
            and result["status"] == "ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED"
            and result["full_epoch_training_complete"] is True
            and source["cell"] == producer["cell"]
            and source["K"] == entry["K"] == producer["K"]
            and source["training_seed"] == entry["training_seed"] == producer["training_seed"]
            and source["role"] == T3_PRODUCER_ROLE
            and source["model_source"]["sha256"] == T3_PRODUCER_MODEL_SHA
            and source["output"] == TOOL3_CHECKPOINT_ROOT + "/" + producer["cell"]
            and source["link_id"].split("-r")[0] == producer["job_id"],
            "Actual Tool3 source-bound full-epoch producer required; not a qualification variant")
    require(result["audit_checks"] == {name: True for name in T3_AUDIT_CHECKS}
            and type(result["overflow_steps"]) is int and 0 <= result["overflow_steps"] <= steps,
            "Tool3 full-horizon training audit (all steps, exact-once exposure) required")
    require(source["link_id"] == producer["link_id"] and producer["steps"] == steps_for(producer["K"]),
            "Tool3 RESULT must come from the pinned RESULT_SUCCESS link")


def producer_files(result, producer):
    """Validate the producer RESULT.json and return the final training/ inference file descriptors."""
    require(isinstance(result, dict) and isinstance(producer, dict) and producer.get("producer_kind") in KINDS,
            "Producer RESULT and kind-bound descriptor required")
    if producer["producer_kind"] == "tool3_chain_v3":
        try:
            tool3_producer_check(result, producer)
        except (KeyError, TypeError, AttributeError) as error:
            raise CustodyError("Malformed Tool3 RESULT: " + repr(error)[:300])
    else:
        steps = producer["steps"]
        require(result.get("status") == PRODUCER_STATUS
                and result.get("job_id") == producer["job_id"] and result.get("cell") == producer["cell"]
                and result.get("K") == producer["K"] and result.get("steps") == steps == steps_for(producer["K"])
                and type(result.get("restart_count")) is int
                and 0 <= result["restart_count"] <= producer["restart_count"]
                and result.get("full_epoch_training_complete") is True, "Actual complete producer epoch required")
        config = result.get("config", {})
        require(config.get("role") == PRODUCER_ROLE and config.get("cell") == producer["cell"]
                and config.get("K") == producer["K"] and config.get("training_seed") == producer["training_seed"]
                and config.get("job_id") == producer["job_id"]
                and config.get("model_source", {}).get("sha256") == PRODUCER_MODEL_SHA,
                "Producer RESULT config differs from the plan-bound custody")
        observation = result.get("observation", {})
        losses = observation.get("losses")
        events = observation.get("step_events")
        require(observation.get("optimizer_boundaries") == steps and observation.get("microbatches") == 32 * steps
                and isinstance(losses, list) and len(losses) == 32 * steps
                and all(type(value) in (float, int) and math.isfinite(value) for value in losses)
                and isinstance(events, list) and [event.get("step") for event in events] == list(range(1, steps + 1))
                and all(event.get("microbatches_seen") == event["step"] * 32 for event in events),
                "Actual native horizon, optimizer boundaries and finite microbatch losses required")
    outputs = result.get("output_files")
    require(isinstance(outputs, dict) and all(name in outputs for name in INFERENCE_FILES),
            "Producer RESULT must name every final inference file")
    files = {}
    for name in INFERENCE_FILES:
        value = outputs[name]
        require(isinstance(value, dict) and set(value) == {"bytes", "sha256"} and type(value["bytes"]) is int
                and value["bytes"] > 0 and SHA.fullmatch(str(value["sha256"])) is not None,
                "Producer inference file descriptor invalid: " + name)
        files[name] = dict(path=producer["training_dir"] + "/" + name, bytes=value["bytes"], sha256=value["sha256"])
    return files


# ------------------------------------------------------------------------------------------------ inventory/worker
def identity_path(run, cell):
    require(type(run) is int and run >= 0 and CELL.fullmatch(cell) is not None, "Literal inventory run/cell required")
    return "%s/run-%d/%s/MODEL_IDENTITY.json" % (INVENTORY_OUTPUT, run, cell)


def inventory_binding(ack_ref, ack):
    """Plan-time binding of one cells-v7 inventory run by its exact ACK; its RESULT is late-bound by job identity."""
    require(is_ref(ack_ref) and isinstance(ack, dict) and ack.get("owner") == INVENTORY_OWNER
            and type(ack.get("run")) is int and JOB.fullmatch(str(ack.get("job_id"))) is not None
            and ack_ref["path"] == "%s/run-%d/ACK.json" % (INVENTORY_CONTROL, ack["run"])
            and is_ref(ack.get("plan")), "Exact cells-v7 inventory run ACK required")
    return dict(ack=ack_ref, plan=ack["plan"], run=ack["run"], job_id=ack["job_id"],
                cells=list(ack.get("cells", [])),
                result_path="%s/run-%d/RESULT-%s.json" % (INVENTORY_CONTROL, ack["run"], ack["job_id"]))


def inventory_custody(result_ref, result, binding, plan_custody):
    """Release/worker gate: the bound run sealed a COMPLETE RESULT with PASS for exactly this custody."""
    cell = plan_custody["cell"]
    require(is_ref(result_ref) and result_ref["path"] == binding["result_path"]
            and isinstance(result, dict) and result.get("schema") == INVENTORY_SCHEMA
            and result.get("status") == INVENTORY_STATUS and result.get("worker_returncode") == 0
            and result.get("actual", {}).get("job_id") == binding["job_id"] and result.get("run") == binding["run"]
            and result.get("plan") == binding["plan"] and cell in binding["cells"],
            "Completed cells-v7 inventory RESULT of the bound run required before release")
    found = result.get("cells", {}).get(cell)
    require(isinstance(found, dict) and found.get("status") == "PASS"
            and found.get("training_result") == plan_custody["training_result"]
            and found.get("trainer_records") == plan_custody["trainer_records"]
            and found.get("producer") == plan_custody["producer"]
            and found.get("producer_result") == plan_custody["v7_custody"]["producer_result"],
            "Inventory did not verify this exact custody (incl. the pinned producer RESULT.json)")
    prior = found.get("model_identity")
    files = found.get("inference_files")
    producer = plan_custody["producer"]
    require(is_ref(prior) and prior["path"] == identity_path(binding["run"], cell)
            and SHA.fullmatch(str(found.get("parameter_sha256"))) is not None
            and is_ref(found.get("producer_result")) and found["producer_result"]["path"] == producer["model_result"]
            and isinstance(files, dict) and sorted(files) == sorted(INFERENCE_FILES)
            and all(is_ref(files[name]) and files[name]["path"] == producer["training_dir"] + "/" + name
                    for name in INFERENCE_FILES), "Inventory custody descriptors malformed")
    return dict(inventory_result=result_ref, producer_result=found["producer_result"], inference_files=files,
                prior_model_identity=prior, prior_parameter_sha256=found["parameter_sha256"])


def worker_custody(cells, cell, custody):
    """GPU-worker check: merged plan + inventory custody is exactly this registered cell's own trained model."""
    spec = cell_spec(cells, cell)
    require(isinstance(custody, dict) and custody.get("kind") == "student" and custody.get("cell") == cell
            and custody.get("trainer_records") == spec["trainer_records"], "Custody is not this registered cell")
    producer = custody.get("producer", {})
    require(isinstance(producer, dict) and type(producer.get("restart_count")) is int
            and producer == producer_descriptor(spec, producer.get("restart_count")),
            "Producer descriptor differs from CELLS_V7")
    require(custody.get("v7_custody") == v7_custody(spec)
            and custody.get("training_result") == spec["training_result"]
            and custody.get("producer_result") == spec["producer_result"],
            "Custody is not this cell's pinned CELLS_V7 completion record and producer RESULT")
    require(SHA.fullmatch(str(custody.get("prior_parameter_sha256"))) is not None
            and is_ref(custody.get("prior_model_identity"))
            and PurePosixPath(custody["prior_model_identity"]["path"]).name == "MODEL_IDENTITY.json"
            and custody["prior_model_identity"]["path"].startswith(INVENTORY_OUTPUT + "/run-")
            and PurePosixPath(custody["prior_model_identity"]["path"]).parent.name == cell,
            "Inventory MODEL_IDENTITY parameter digest must be bound before inference")
    files = custody.get("inference_files")
    require(is_ref(custody.get("producer_result")) and custody["producer_result"]["path"] == producer["model_result"]
            and isinstance(files, dict) and sorted(files) == sorted(INFERENCE_FILES)
            and all(is_ref(files[name]) and files[name]["path"] == producer["training_dir"] + "/" + name
                    for name in INFERENCE_FILES), "Producer inference descriptors differ from the registered cell")
    return spec


def model_identity(cell, run, job, parameter_sha256, generation_config, producer, files, load_proof):
    """MODEL_IDENTITY.json body written by the inventory child (the evaluators' prior parameter digest)."""
    require(SHA.fullmatch(str(parameter_sha256)) is not None, "Actual parameter digest required")
    return dict(schema=IDENTITY_SCHEMA, owner=INVENTORY_OWNER, cell=cell, run=run, job_id=job,
                parameter_sha256=parameter_sha256, generation_config=generation_config, producer=producer,
                inference_files=files, model_load=load_proof, adapter_active=True, adapter_merged=False,
                training_updates=0)
