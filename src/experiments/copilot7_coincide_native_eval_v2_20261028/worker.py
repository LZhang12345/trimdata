"""Root-launched final native COINCIDE checkpoint, unchanged official bank loops."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from experiments.root_current_four_bank_eval_v1_20261027 import worker as core
from . import protocol as p, parents, model, scoring

u = core.u
banks, media, generation, qualified = core.banks, core.media, core.generation, core.qualified
budget, publish_same = core.budget, core.publish_same


def documents(binding, load=p.checked):
    cfg = load(binding["configuration"])
    training = load(cfg["training_configuration"])
    basis = load(cfg["qualified_basis"])
    p.validate_science(cfg, basis, training)
    review = load(binding["independent_source_config_review"])
    p.native.independent(review)
    p.require(review["status"] == p.REVIEW and review["independent"] is True
        and review["configurations"][cfg["student_id"]] == binding["configuration"]
        and review["sources"] == cfg["sources"]
        and review["unchanged_loader_numerical_bodies"] == model.qualified_body_proof(),
        "Independent exact new COINCIDE eval source/config and scoped trained-loader review")
    for relative, expected in cfg["sources"].items():
        p.require(p.native.sha(p.ROOT / relative) == expected, "Frozen evaluation source changed: " + relative)
    qref = p.desc(basis["required_actual_qualification_result_path"])
    core.checks.qualification_result(basis, load(qref), load)
    qpeer = load(p.desc(basis["required_independent_qualification_review_path"]))
    core.checks.independent_result(basis, qpeer, qref, load)
    _, qcfg = core.c.prerequisite_roster(basis, load)
    anchor = load(qcfg["training_configuration"])
    p.require(training["recipe"] == anchor["recipe"]
        and all(training["providers"][name] == anchor["providers"][name] for name in ("environment", "assets")),
        "Exact already qualified native model/processor/runtime class")
    claim, grant = load(binding["ownership"]), load(binding["storage_reservation"])
    p.require(binding["status"] == p.BINDING and binding["owner"] == "Root"
        and claim["owner"] == "Root" and claim["job_id"] == binding["job_id"]
        and claim["configuration"] == binding["configuration"]
        and claim["nonce"] == binding["nonce"] == cfg["nonce"]
        and claim["training_configuration"] == cfg["training_configuration"]
        and claim["no_live_duplicate_checkpoint_evaluation"] is True,
        "Root exact final-checkpoint/four-bank function claim")
    p.require(grant["owner"] == "Root" and grant["configuration"] == binding["configuration"]
        and grant["output"] == cfg["output"] and grant["reserved_bytes"] == cfg["output_limit_bytes"]
        and grant["final_checkpoint_retained_for_evaluation"] is True,
        "Real separate evaluation storage and retained final state")
    return cfg, load(cfg["official_POPE_configuration"])


def actor(binding, cfg, query=subprocess.check_output):
    job = binding["job_id"]
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    p.require(os.getuid() == 2708899 and job.isdecimal() and os.environ.get("SLURM_JOB_ID") == job
        and restart.isdecimal() and int(restart) < cfg["max_physical_attempts"]
        and not os.environ.get("SLURM_ARRAY_JOB_ID") and p.ROOT == p.SOURCE_ROOT
        and Path(sys.executable).absolute() == Path(p.PYTHON), "Actual source-bound General evaluator")
    raw = query(["scontrol", "show", "job", "-o", job], text=True, timeout=30)
    fields = u.fields(raw)
    expected = dict(JobId=job, JobState="RUNNING", JobName=p.NAME, UserId="luyangz(2708899)",
        Partition="general", Account="cx", QOS="normal", NumCPUs="8", NumTasks="1", NumNodes="1",
        Restarts=restart, TimeLimit="06:00:00", TresPerNode="gres/gpu:L40S:1",
        Command=str(p.HERE / "run.sbatch"), WorkDir=str(p.ROOT),
        Comment=p.COMMENT + cfg["nonce"], NodeList=socket.gethostname().split(".")[0])
    p.require(all(fields.get(name) == value for name, value in expected.items()), "Actual evaluator job/spool/resources")
    tres = dict(value.split("=", 1) for value in fields["AllocTRES"].split(",") if "=" in value)
    p.require(tres.get("cpu") == "8" and tres.get("mem") in ("64G", "65536M")
        and tres.get("gres/gpu") == "1"
        and query(["scontrol", "write", "batch_script", job, "-"], timeout=30)
        == (p.HERE / "run.sbatch").read_bytes(), "Actual CPU8/64GiB/L40S1 and immutable script")
    p.require(os.environ.get("TZ") == "America/New_York", "Exact scheduler timezone")
    time.tzset()
    return dict(job_id=job, restart=int(restart), node=fields["NodeList"], nonce=cfg["nonce"],
        scheduler=raw, end_epoch=int(time.mktime(time.strptime(fields["EndTime"], "%Y-%m-%dT%H:%M:%S"))),
        process_nonce=uuid.uuid4().hex, pid=os.getpid())


def prior(binding, actual, output, load=p.checked, describe=p.desc):
    if not actual["restart"]:
        return None
    path = output / ("r" + str(actual["restart"] - 1)) / "RESULT.json"
    previous_ref = describe(path)
    value = load(previous_ref)
    proof = load(binding["restart_reconciliation"])
    p.native.independent(proof)
    p.require(value["status"] == "ACCEPTED_QA_PROGRESS_REQUIRES_ROOT_CONTINUATION"
        and value["new_answers"] > 0 and value["configuration"] == binding["configuration"]
        and value["actual"]["job_id"] == actual["job_id"] and value["actual"]["nonce"] == actual["nonce"]
        and value["actual"]["restart"] == actual["restart"] - 1
        and proof["status"] == "PASS_INDEPENDENT_ACTUAL_COINCIDE_EVALUATION_SUCCESSOR_V2"
        and proof["previous_result"] == previous_ref and proof["configuration"] == binding["configuration"]
        and proof["job_id"] == actual["job_id"] and proof["next_restart"] == actual["restart"]
        and proof["actual_paid_QA_prefix_verified"] is True,
        "Root independent actual successor/per-QA proof required before any resumed forward")
    return value


def function(cfg, identity, roster, complete_ref, bank):
    value = core.original.function(cfg, identity, roster, complete_ref, bank)
    return dict(value, schema="coincide-fixed10-fourbank-function-v2", role=p.SCOPE,
        adaptation_label=p.native.ADAPTATION, training_logical_protocol=p.native.PACKAGE,
        generation_API_policy=model.generation.POLICY,
        GQA_conversion=cfg["GQA_conversion"])


def run(binding_path):
    binding = p.checked(p.desc(binding_path))
    preliminary = p.checked(binding["configuration"])
    actual, began = actor(binding, preliminary), time.monotonic()
    cfg, base = documents(binding)
    output = Path(cfg["output"])
    p.require(not any(path.is_symlink() for path in (output, *output.parents)), "Literal new evaluator output")
    output.mkdir(parents=True, exist_ok=True)
    fd = os.open(output / "EXECUTION.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = output / ("r" + str(actual["restart"]))
        attempt.mkdir(exist_ok=False)
        previous = prior(binding, actual, output)
        grant = p.checked(binding["storage_reservation"])
        u.publish(attempt / "CURRENT_STORAGE.json", core.current_storage(cfg, grant, budget(cfg)))
        _, complete_ref, complete, physical = parents.complete(cfg, binding, attempt / "PARENT_PREFLIGHT")
        u.publish(attempt / "START.json", dict(actual=actual, configuration=binding["configuration"],
            student_completion=complete_ref, physical_proofs=physical))
        bankdata = {bank: banks.load(bank, cfg) for bank in cfg["banks"]}
        ready = {}
        for bank, (roster, inventory, _, physical, snapshot) in bankdata.items():
            store = media.Store(physical, roster["records"], inventory, snapshot)
            try:
                names = sorted({row["native_input"]["image"] for row in roster["records"]})
                for name in names:
                    image = store.rgb(name)
                    image.close()
                ready[bank] = dict(questions=len(roster["records"]), images=len(names),
                                   all_encoded_and_RGB_verified=True, forwards=0)
            finally:
                store.close()
        u.publish(attempt / "FULL_MEDIA_READINESS.json", ready)
        publish_same(output / "STUDENT_INPUT.json", dict(training_configuration=cfg["training_configuration"],
            student_completion=complete_ref, accepted=complete["accepted"]))
        runtime = qualified.official.runtime(base, attempt)
        u.publish(attempt / "RUNTIME.json", runtime)
        with qualified.official.offline():
            loaded3, proof = model.compose_v4(complete["accepted"], cfg["training_configuration"])
            core.strict_loaded_support(proof)
            identity = model.identity_v4(loaded3, proof, runtime)
            identity_ref = publish_same(output / "MODEL_IDENTITY.json", identity)
            u.publish(attempt / "MODEL_RECONSTRUCTION.json", proof)
            loaded = (*loaded3, getattr(loaded3[1].config, "max_sequence_length", 2048))
            reports, new = {}, 0
            for bank, (roster, inventory, targets, physical, snapshot) in bankdata.items():
                rows, path = roster["records"], output / bank
                path.mkdir(exist_ok=True)
                fn = function(cfg, identity, roster, complete_ref, bank)
                publish_same(path / "FUNCTION.json", fn)
                journal = generation.AuditedJournal(path / "journal", fn, rows)
                store = media.Store(physical, rows, inventory, snapshot)
                try:
                    for entry in qualified.official.entries(journal, rows):
                        generation.validate_entry(bank, entry, journal.roster[entry["semantic_key"]], fn)
                    core.accepted_prefix_count(previous, bank, len(journal.present))
                    before = len(journal.present)
                    if time.time() < actual["end_epoch"] - 300:
                        generation.native(bank, cfg, rows, journal, loaded, store, actual,
                                          time.monotonic() + max(0, actual["end_epoch"] - time.time() - 240))
                    new += len(journal.present) - before
                    reports[bank] = dict(questions=len(rows), accepted=len(journal.present),
                        missing=len(journal.missing()), new=len(journal.present) - before)
                    if not journal.missing():
                        raw = journal.export(path / "PREDICTIONS.jsonl")
                        value = scoring.score(bank, roster, journal, targets)
                        publish_same(path / "SCORE.json", dict(status="COMPLETE_COINCIDE_NATIVE_OFFICIAL_BANK",
                            bank=bank, function=fn, predictions=raw, score=value,
                            student_completion=complete_ref, complete_questions=len(rows),
                            scientific_improvement_demonstrated=False))
                finally:
                    journal.close()
                    store.close()
                budget(cfg)
        p.require(model.identity_v4(loaded3, proof, runtime) == identity, "Generation mutated trained state")
        bridge = model.generation.readout(loaded3[1])
        p.require(bridge["calls"] == new, "Every new QA needs the exact qualified V4 bridge")
        u.publish(attempt / "GENERATION_BRIDGE_READOUT.json", bridge)
        done = core.accepted_progress(previous, reports, new)
        value = dict(status="COMPLETE_COINCIDE_FOUR_OFFICIAL_BANKS" if done else
                     "ACCEPTED_QA_PROGRESS_REQUIRES_ROOT_CONTINUATION",
            actual=actual, configuration=binding["configuration"], student_completion=complete_ref,
            accepted_checkpoint=complete["accepted"], model_identity=identity_ref, banks=reports,
            new_answers=new, elapsed_seconds=time.monotonic() - began,
            all_prior_and_current_costs_count=True, scientific_improvement_demonstrated=False,
            scheduler_mutations=False)
        ref = u.publish(attempt / "RESULT.json", value)
        if done:
            publish_same(output / "COMPLETE.json", value)
        else:
            p.require(new > 0 and actual["restart"] + 1 < cfg["max_physical_attempts"],
                      "No accepted progress; preserve without auto retry")
            u.publish(attempt / "ROOT_CONTINUATION_REQUIRED.json", dict(previous_result=ref,
                job_id=actual["job_id"], next_restart=actual["restart"] + 1,
                per_QA_durable=True, actual_Root_transition_and_review_required=True, scheduler_action=False))
        return 0 if done else 75
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as error:
        if "attempt" in locals() and attempt.is_dir():
            u.publish(attempt / "FAILURE.json", dict(status="COINCIDE_EVALUATION_FAILED_NO_AUTO_RETRY",
                actual=actual, error=repr(error), elapsed_seconds=time.monotonic() - began))
        raise
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    raise SystemExit(run(parser.parse_args().binding))
