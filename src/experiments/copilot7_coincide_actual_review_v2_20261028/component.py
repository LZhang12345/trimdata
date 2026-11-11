"""Model-free actual-output review, callable only by a Root-reviewed actor.

No launcher is supplied. Configuration/source review and genuine registered
General allocation bindings must precede every DATA operation.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys

from experiments.meta_probe_v1_20261028.contracts import ScopedReader, canonical, require
from . import data_io

BASE = Path("/home/luyangz/data_selection_multimodal")
OUTPUT = Path("/data/hf_cache/project_features/DS-MultiModal/coincide_native_operator_20261027/v3")
FUNCTION = "bbde85dd4c12b6321399d553d3333ec9c2e855d90cc7ca73600b3e32413be212"
JOB = "10961650"
PYTHON = "/data/hf_cache/project_envs/DS-MultiModal/coincide-faiss180-torch220-cu121-v2/bin/python"
UID = 2708899
K = {"0.05": 33038, "0.10": 66076, "0.20": 132152}
STATUS = "ACTUAL_FULL_NATIVE_COINCIDE_OPERATOR_CANDIDATES_PENDING_INDEPENDENT_ACTUAL_REVIEW"
MODULE = "experiments.copilot7_coincide_actual_review_v2_20261028.component"


def expected_document(path: Path, value: dict) -> dict:
    """Expected bytes from the actual owner's frozen canonical seal, then read."""
    raw = canonical(value)
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read_metadata(ref: dict, roots: tuple[Path, ...]) -> dict:
    if str(ref["path"]).startswith("/data/"):
        value = data_io.load_json(ref, roots)
        require(type(value) is dict, "Expected paid DATA metadata object")
        return value
    return ScopedReader(roots, [ref])(ref)


def small_array(ref: dict):
    import numpy as np
    require(OUTPUT in Path(ref["path"]).parents, "Array escaped exact owned output")
    raw = data_io.read_exact(ref, (OUTPUT,))
    return np.load(io.BytesIO(raw), allow_pickle=False)


def hash_output(ref: dict, root: Path = OUTPUT) -> dict:
    data_io.read_exact(ref, (root,), collect=False, maximum=2 << 30)
    return dict(reference=ref, fresh_actual_byte_hash=True, pickle_deserialized=False)


def summarize_candidate(raw: dict, candidate: dict, ids: list[str], labels,
                        fraction: str, expected_K: int, owner: dict) -> dict:
    import numpy as np
    require(raw["top_up"] is False and raw["source_order_restored_for_training"] is True,
            "Native no-topup/original-order semantics differ")
    positions = raw["positions"]
    require(type(expected_K) is int and expected_K > 0
            and type(raw["count"]) is int and type(raw["remaining"]) is int
            and isinstance(positions, list) and all(type(i) is int and 0 <= i < len(ids) for i in positions)
            and len(positions) == len(set(positions)) == raw["count"] > 0
            and raw["count"] <= expected_K and raw["remaining"] == expected_K - raw["count"]
            and raw["author_selection_order"] == positions, "Invalid native positions/count/remaining graph")
    require(labels.shape == (len(ids),) and labels.dtype == np.dtype("<i8")
            and int(labels.min()) >= 0, "Invalid canonical native label array")
    quotas = raw["quotas"]
    count = int(labels.max()) + 1
    require(isinstance(quotas, list) and len(quotas) == count
            and all(isinstance(q, list) and len(q) == 2 and type(q[0]) is int and type(q[1]) is int
                    and 0 <= q[0] < count and q[1] >= 0 for q in quotas)
            and {q[0] for q in quotas} == set(range(count)), "Native quota support/uniqueness differs")
    populations = np.bincount(labels, minlength=count)
    chosen = np.bincount(labels[np.asarray(positions, dtype=np.int64)], minlength=count)
    require(all(int(chosen[k]) == min(q, int(populations[k])) for k, q in quotas),
            "Positions disagree with executed native cluster quotas")
    ordered = sorted(positions)
    record_ids = [ids[i] for i in ordered]
    key = lambda value: hashlib.sha256(canonical(value)).hexdigest()
    require(candidate["status"] == "PENDING_INDEPENDENT_AUTHOR_OPERATOR_OUTPUT_REVIEW_NOT_STUDENT_READY"
            and candidate["method"] == "coincide" and candidate["fraction"] == fraction
            and candidate["selection_seed"] == 1234 and candidate["requested_K"] == expected_K
            and candidate["actual_K"] == len(ordered) and candidate["positions"] == ordered
            and candidate["record_ids"] == record_ids
            and candidate["canonical_record_ids_sha256"] == key(ids)
            and candidate["selected_record_ids_sha256"] == key(record_ids)
            and candidate["source_configuration"] == owner["configuration"]
            and candidate["pool"] == owner["pool"]
            and candidate["records_json"] is None and candidate["student_READY"] is False
            and candidate["original_whole_records_not_yet_materialized"] is True
            and candidate["author_floor_not_topped_up"] is True
            and candidate["matched_actual_K_uniform_required"] == (len(ordered) != expected_K),
            "Candidate identity/order/source/count/materialization graph differs")
    return dict(fraction=fraction, requested_K=expected_K, actual_K=len(ordered),
        unique_record_ids=len(set(record_ids)), selected_record_ids_sha256=key(record_ids),
        selected_positions_sha256=key(ordered), underfill=expected_K - len(ordered),
        native_cluster_coverage=int(np.count_nonzero(chosen)),
        duplicate_or_foreign_positions=0, no_topup=True, records_materialized=False,
        student_READY=False, independent_method_acceptance=False)


def verify_registered(cfg: dict, actual: dict, actor: dict, configuration: dict) -> None:
    require(cfg["schema"] == "s7-actual-coincide-output-review-v2"
            and cfg["owner"] == "Session7" and cfg["function_identity"] == FUNCTION
            and cfg["operator_job"] == JOB and cfg["operator_output"] == str(OUTPUT),
            "Only exact current paid COINCIDE output")
    require(cfg["new_NN"] == 0 and cfg["operator_rerun"] is False
            and cfg["scheduler_mutations"] is False and cfg["checkpoint_writes"] is False,
            "No operator/model/scheduler execution")
    require(actor.get("Root_component_review") is not None and actor.get("Root_execution_authorization") is not None,
            "NOT_READY: Root source/config and owned execution approvals missing")
    require(os.getuid() == UID and actual["uid"] == UID and actual["pid"] == os.getpid()
            and actual["job_id"] == os.environ.get("SLURM_JOB_ID")
            and actual["numeric_step"] == os.environ.get("SLURM_STEP_ID")
            and actual["numeric_step"].isdecimal()
            and actual["node"] == socket.gethostname().split(".")[0]
            and Path(sys.executable).absolute() == Path(PYTHON),
            "Actual current owned numeric allocation process required before DATA")
    require(actor["status"] == "ACTUAL_ROOT_OWNED_COMPONENT_REGISTERED_V1"
            and actor["actual"] == actual and actor["component_module"] == MODULE
            and actor["configuration"] == configuration,
            "Root must bind the exact reviewed additive component actor; no old-route spoof")
    require(all(actual["sources"].get(rel) == digest for rel, digest in cfg["sources"].items()),
            "Actual registered component source closure differs")


def run(configuration: dict, actual: dict, registration: dict) -> dict:
    """Entry for Root's registered component adapter, not an admission function."""
    roots = (BASE / "collaboration",)
    cfg = read_metadata(configuration, roots)
    actor = read_metadata(registration, roots)
    verify_registered(cfg, actual, actor, configuration)
    review = read_metadata(actor["Root_component_review"], roots)
    approval = read_metadata(actor["Root_execution_authorization"], roots)
    require(review["status"] == "ROOT_PERSONALLY_APPROVED_S7_COINCIDE_OUTPUT_REVIEW_SOURCE_V2"
            and review["configuration"] == configuration and review["sources"] == cfg["sources"],
            "Detached exact Root source/config review required")
    require(approval["owner"] == "Root" and approval["configuration"] == configuration
            and approval["component_module"] == MODULE and approval["function_identity"] == FUNCTION,
            "Detached current Root owned-execution authority required")
    home = read_metadata(cfg["owner_HOME_result"], roots)
    require(home["status"] == STATUS and home["models"] == home["forwards"] == home["new_features"] == 0,
            "Actual sealed owner completion required")
    require(home["result"] == cfg["actual_DATA_result"], "Exact paid DATA result descriptor")
    exported = read_metadata(cfg["already_registered_result_export"], roots)
    require(exported["returncode"] == 0, "Root existing result export failed")
    acquired = json.loads(exported["stdout"])
    require(acquired["reference"] == home["result"]
            and str(acquired["actual_job"]).isdecimal()
            and str(acquired["actual_step"]).isdecimal(),
            "Reuse exact already registered DATA result provider, not a new unowned read")
    result = acquired["result"]
    expected_result_bytes = canonical(result)
    require(len(expected_result_bytes) == home["result"]["bytes"]
            and hashlib.sha256(expected_result_bytes).hexdigest() == home["result"]["sha256"],
            "Actual acquired canonical owner-result bytes differ")
    require(result["status"] == STATUS and result["job_id"] == JOB
            and result["matrix_bytes_unchanged"] is True
            and result["full_selected_records_materialized"] is False
            and result["student_or_evaluation_READY"] is False, "Actual operator remains candidate-only")
    require(result["configuration"] == cfg["operator_configuration"], "Wrong actual source configuration")
    spec = read_metadata(result["configuration"], roots)
    require(spec["output"] == str(OUTPUT) and spec["feature_measurement"] == cfg["measurement_identity"],
            "Wrong operator configuration/feature function")
    from experiments.copilot7_coincide_native_operator_20261027.protocol import function_identity
    require(function_identity(spec) == FUNCTION, "Exact current bbde source/input function differs")
    pool = read_metadata(spec["pool"], (Path(spec["pool"]["path"]).parent,))
    require(pool["status"] == "PASS_POOL"
            and pool["count"] == len(pool["record_ids"]) == len(set(pool["record_ids"])) == 660763,
            "Exact complete canonical pool IDs")
    eligibility = read_metadata(spec["eligibility"], (Path(spec["eligibility"]["path"]).parent,))
    require(eligibility["status"] == "PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY"
            and eligibility["pool"] == spec["pool"], "Exact qualified protected eligibility differs")
    owner = home["outputs"][0]["owner"]
    require(owner["function_identity"] == FUNCTION and owner["configuration"] == result["configuration"]
            and owner["pool"] == spec["pool"], "Exact bbde output lease/owner identity")
    require(result["matrix"] == spec["matrix_receipt"], "Foreign paid feature provider")
    matrix_header = read_metadata(spec["matrix_receipt"], (Path(spec["matrix_receipt"]["path"]).parent,))
    require(matrix_header["count"] == 660763 and matrix_header["method"] == "coincide"
            and matrix_header["measurement_identity"] == cfg["measurement_identity"]
            and matrix_header["owner"]["pool"] == spec["pool"]
            and matrix_header["owner"]["eligibility"] == spec["eligibility"]
            and matrix_header["hidden"] == owner["matrix"]
            and matrix_header["original_qualified_rows_reused"] == 16
            and matrix_header["production_unique_rows"] == 660747,
            "Exact paid complete feature/matrix custody differs")
    require(read_metadata(expected_document(OUTPUT / "OWNER.json", owner), (OUTPUT,)) == owner,
            "Original output OWNER changed")
    cluster = read_metadata(result["native_cluster"], (OUTPUT,))
    transfer = read_metadata(result["native_transfer"], (OUTPUT,))
    require(cluster["owner"] == transfer["owner"] == owner
            and transfer["cluster"] == result["native_cluster"], "Geometry/transfer owner graph differs")
    require(set(cluster["files"]) == {"kmeans_index.pickle", "kmeans_centroids.npy", "dist_to_cent.npy", "nearest_cent.npy"},
            "Native geometry artifact set differs")
    require(all(ref["path"] == str(OUTPUT / "native-faiss" / name) for name, ref in cluster["files"].items())
            and transfer["transfer"]["path"] == str(OUTPUT / "transfer.npy"), "Foreign output artifact paths")
    # This calls only the frozen metadata predicate, never any native operator.
    from experiments.copilot7_coincide_native_operator_20261027.worker import validate_gpu_metadata
    validate_gpu_metadata(cluster["actual_GPU_index"])
    proofs = [hash_output(ref) for ref in cluster["files"].values()]
    proofs.append(hash_output(transfer["transfer"]))
    import numpy as np
    labels = small_array(cluster["files"]["nearest_cent.npy"])
    distances = small_array(cluster["files"]["dist_to_cent.npy"])
    values = small_array(transfer["transfer"])
    require(labels.shape == distances.shape == (660763,) and labels.dtype == np.dtype("<i8")
            and distances.dtype == np.dtype("<f4") and np.isfinite(distances).all()
            and np.array_equal(np.unique(labels), np.arange(10000))
            and values.shape == (10000,) and values.dtype == np.dtype("<f8") and np.isfinite(values).all(),
            "Actual native small-array shape/dtype/finite/cluster support differs")
    populations = np.bincount(labels, minlength=10000)
    require(cluster["cluster_sizes"] == populations.tolist(), "Actual cluster populations differ")
    from experiments.s1_native_selection_controls_20261023.source_labels_v1 import interval_of, label_name
    require(len(pool["source_indices"]) == 660763, "Original source-index provenance missing")
    source_labels = [label_name(interval_of(i)) for i in pool["source_indices"]]
    outputs = []
    require(result["outputs"] == home["outputs"] and len(result["outputs"]) == 3, "HOME/DATA actual outputs differ")
    for row in result["outputs"]:
        fraction = row["fraction"]
        require(fraction in K and row["owner"] == owner, "Unknown budget/owner")
        require(row["candidate"]["path"] == str(OUTPUT / ("budget-" + fraction) / "CANDIDATE.json")
                and row["operator_result"]["path"] == str(OUTPUT / ("budget-" + fraction) / "AUTHOR_OPERATOR_RESULT.json"),
                "Foreign per-budget output paths")
        complete = read_metadata(expected_document(OUTPUT / ("budget-" + fraction) / "COMPLETE.json", row), (OUTPUT,))
        require(complete == row, "Actual per-budget completion changed")
        raw = read_metadata(row["operator_result"], (OUTPUT,))
        candidate = read_metadata(row["candidate"], (OUTPUT,))
        require(candidate["operator_result"] == row["operator_result"]
                and candidate["cluster"] == result["native_cluster"]
                and candidate["transfer"] == result["native_transfer"]
                and candidate["operator_sources"] == spec["operator_sources"], "Candidate feature/source dependency differs")
        summary = summarize_candidate(raw, candidate, pool["record_ids"], labels, fraction, K[fraction], owner)
        from collections import Counter
        summary["author_original_interval_counts_not_semantic_certification"] = dict(sorted(Counter(
            source_labels[i] for i in candidate["positions"]).items()))
        summary["candidate"] = row["candidate"]
        summary["raw_native_operator_result"] = row["operator_result"]
        summary["selected_distance_quantiles"] = np.quantile(distances[candidate["positions"]], [0, .5, 1]).tolist()
        outputs.append(summary)
    return dict(status="ACTUAL_READONLY_OUTPUT_REVIEW_CANDIDATE_NOT_INDEPENDENT_ACCEPTANCE",
        operator_job=JOB, function_identity=FUNCTION, owner_HOME_result=cfg["owner_HOME_result"],
        actual_DATA_result=home["result"], actual_registered_execution=actual,
        existing_registered_result_export=cfg["already_registered_result_export"],
        repeated_DATA_RESULT_acquisition=False,
        geometry_and_transfer_actual_output_hashes=proofs, outputs=outputs,
        paid_source_feature_provider=dict(complete=spec["matrix_receipt"],
            hidden=matrix_header["hidden"], measurement_identity=cfg["measurement_identity"],
            count=660763, qualified_original_rows=16, unique_production_rows=660747,
            new_extraction=False),
        cluster_size=dict(min=int(populations.min()), max=int(populations.max()), count=10000),
        transferability_quantiles=np.quantile(values, [0, .25, .5, .75, 1]).tolist(),
        transferability_is_author_feature_not_human_quality_or_learning_gain=True,
        feature_matrix_input_reused_from_owner_verified_before_after_hash_proof=True,
        full_input_matrix_rehash_repeated=False, operator_math_or_MMD_rerun=False,
        new_models=0, new_forwards=0, new_features=0, scheduler_mutations=False,
        centroid_values_independently_recomputed=False, pickle_deserialized=False,
        selected_whole_records_materialized=False, student_READY=False,
        independent_method_acceptance=False, scientific_gain_established=False)
