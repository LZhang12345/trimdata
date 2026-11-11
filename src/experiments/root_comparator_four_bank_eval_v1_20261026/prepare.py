"""Local code/config preparation only; no scheduler or network operations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import uuid

from . import contract as c, evidence, historical, worker

DISPATCH = c.ROOT / "collaboration/root-agent5-comparator-eval-dispatch-20261026"
ROSTER = c.ROOT / "collaboration/root-babel-followup-queue-20261026/ROOT_COMPARATOR_FORMAL_EVALUATION_NEXT_ROSTER_V1.json"
ORIGINAL = c.ROOT / "collaboration/root-native-trained-evaluation-v1-20261026"


def original_ref(path):
    raw = path.read_bytes()
    return dict(path=str(c.HOME / path.relative_to(c.ROOT)),
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        c.require(path.read_bytes() == raw, "Prepared artifact differs; use a new engineering revision")
    else:
        with path.open("xb") as stream:
            stream.write(raw)
    return raw


def configuration(cell, template, roster_ref, source_map, qualifier_binding, reviewer_binding):
    cfg = copy.deepcopy(template)
    cfg.pop("prior_scheduler_cancelled_evaluation")
    cfg.pop("actual_new_qualifier_admission")
    cfg.update(status=c.STATUS, scope=c.SCOPE, source_root=str(c.SOURCE_ROOT),
               student_id=cell["student_id"], training_configuration=cell["training_configuration"],
               student_completion_path=cell["student_completion_path"], training_cell=copy.deepcopy(cell),
               official_bank_contract_configuration=c.OFFICIAL_BANK_TEMPLATE,
               root_comparator_roster=roster_ref, sources=source_map,
               nonce=uuid.uuid5(uuid.NAMESPACE_URL, c.SCOPE + ":" + cell["student_id"]).hex,
               output=str(c.OUTPUT / cell["student_id"]),
               required_training_job_id=cell["training_job_id"],
               required_loader_qualifier_job_id=c.QUALIFIER, required_independent_reviewer_job_id=c.REVIEWER,
               additive_repair="Root264 historical physical272 adapter; bound S1 consumer and loaderV3 unchanged",
               late_binding_contract="Actual Root final516 COMPLETE/accepted and full historical physical proof at execution; no future SHA",
               checkpoint_retention=dict(training_configuration=cell["training_configuration"],
                   natural_final_step=516, retain_until="SIX_ROOT_COMPARATOR_FOUR_BANK_RESULTS_INDEPENDENTLY_ACCEPTED",
                   checkpoint_hash_late_bound=True, no_weight_copy=True))
    cfg["prerequisites"] = [
        dict(role="training", job_id=cell["training_job_id"], binding=cell["training_binding"],
             artifact_path=cell["student_completion_path"]),
        dict(role="qualifier", job_id=c.QUALIFIER, binding=qualifier_binding,
             artifact_path=cfg["required_actual_qualification_result_path"]),
        dict(role="reviewer", job_id=c.REVIEWER, binding=reviewer_binding,
             artifact_path=cfg["required_independent_qualification_review_path"]),
    ]
    return cfg


def build(output):
    output = Path(output)
    c.require(output.is_relative_to(DISPATCH), "Only this preparation dispatch directory")
    documents = evidence.exported_documents()
    read = lambda ref: evidence.read_exported(ref, documents)
    roster = json.loads(ROSTER.read_bytes())
    c.require(len(roster["entries"]) == 6 and
              {(v["method"], v["training_seed"]): v["training_job_id"] for v in roster["entries"]} == c.JOBS,
              "Genuine exact six Root roster")
    template = json.loads((ORIGINAL / "formal-configs-v3/CONFIG_s1-student-v1-10-0R-s17.json").read_bytes())
    base, phase3 = read(template["official_POPE_configuration"]), read(template["phase3_configuration"])
    source_map = worker.sources(base, phase3)
    qualifier = json.loads((ORIGINAL / "ACTUAL_QUALIFIER_ADMISSION_V3.json").read_bytes())
    reviewer = json.loads((c.ROOT / "collaboration/root-native-trained-qual-review-v3-20261026/ACTUAL_ADMISSION_V3.json").read_bytes())
    c.require(qualifier["job_id"] == c.QUALIFIER and reviewer["job_id"] == c.REVIEWER, "Actual prerequisite admissions")
    configs = {}
    for cell in roster["entries"]:
        for field in ("training_configuration", "training_binding", "training_activation", "training_outer_execution"):
            read(cell[field])
        cfg = configuration(cell, template, original_ref(ROSTER), source_map, qualifier["binding"], reviewer["binding"])
        c.validate_configuration(cfg, read)
        c.require(len(read(cell["training_configuration"])["execution_adapter"]["sources"]) == 264,
                  "Actual native config must name Root264, not S1")
        path = output / "configs" / ("CONFIG_" + cell["student_id"] + ".json")
        raw = write(path, cfg)
        configs[cell["student_id"]] = dict(path=str(c.CONTROL / "configs" / path.name),
                                           bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    packet = dict(schema="root-comparator-six-four-bank-source-packet-v1", owner="Root",
        source_root=str(c.SOURCE_ROOT), sources=source_map, configurations=configs,
        original_root_roster=original_ref(ROSTER), declared_answers_each=28505, declared_answers_six=171030,
        preparation_only=True, new_jobs_submitted=[], no_future_completion_or_qualification_PASS=True,
        logical_training_sources=historical.student.sources(), physical_training_sources=historical.route.sources(),
        reused_bank_generation_scoring_package="experiments.root_native_trained_evaluation_v3_20261026",
        result_reservation_bytes_each=512 << 20, collective_result_reservation_bytes=6 * (512 << 20),
        retained_final_checkpoints=[dict(configuration=cell["training_configuration"], final_step=516,
            hashes_late_bound=True, copy_weights=False) for cell in roster["entries"]])
    write(output / "SOURCE_PACKET_V1.json", packet)
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    result = build(parser.parse_args().output_directory)
    print(json.dumps(dict(prepared_configurations=len(result["configurations"]),
                          frozen_sources=len(result["sources"]), jobs_submitted=[])))
