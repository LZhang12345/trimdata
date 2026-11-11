"""Root request/postACK construction only; never submit or release."""
import argparse
from pathlib import Path
import re
from . import protocol as p


def request(configuration, nonce, load=p.checked):
    cfg = load(configuration)
    training = load(cfg["training_configuration"])
    p.validate_science(cfg, load(cfg["qualified_basis"]), training)
    parent = load(cfg["training_binding"])
    p.require(parent["configuration"] == cfg["training_configuration"]
        and parent["job_id"] == cfg["required_training_job_id"] and parent["sources"] == p.native.sources()
        and re.fullmatch("[a-f0-9]{32}", nonce) and nonce == cfg["nonce"],
        "Actual admitted original parent ID and exact new evaluator nonce")
    control = p.native.REPORT_ROOT / cfg["student_id"] / "evaluation" / nonce
    binding = control / "BINDING.json"
    command = p.desc(p.HERE / "run.sbatch")
    argv = ["sbatch", "--parsable", "--hold", "--open-mode=append", "--account=cx",
        "--partition=general", "--qos=normal", "--gres=gpu:L40S:1", "--cpus-per-task=8",
        "--mem=65536M", "--nodes=1", "--ntasks=1", "--time=06:00:00", "--requeue",
        "--dependency=afterok:" + parent["job_id"], "--job-name=" + p.NAME,
        "--comment=" + p.COMMENT + nonce, "--chdir=" + str(p.ROOT),
        "--output=" + str(control / "slurm-%j.out"), "--error=" + str(control / "slurm-%j.err"),
        command["path"], str(binding)]
    return dict(schema="coincide-fourbank-held-request-v2", owner="Root", configuration=configuration,
        parent=cfg["training_binding"], nonce=nonce, sources=cfg["sources"], command=command,
        control=str(control), binding=str(binding), sbatch_argv=argv,
        actual_submission_count=0, afterok_not_full_parent_acceptance=True)


def post_ack(request_ref, intent_ref, ack_ref, ownership_ref, storage_ref, source_review_ref,
             parent_review_ref, load=p.checked, publish=p.native.seal):
    value, intent, ack = load(request_ref), load(intent_ref), load(ack_ref)
    cfg = load(value["configuration"])
    p.require(value["schema"] == "coincide-fourbank-held-request-v2"
        and value["sources"] == cfg["sources"]
        and intent["owner"] == "Root" and intent["request"] == request_ref
        and intent["configuration"] == value["configuration"] and intent["nonce"] == value["nonce"]
        and ack["intent"] == intent_ref and ack["response"]["returncode"] == 0
        and re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?\n?", ack["response"]["stdout"]),
        "Actual held intent and unambiguous rawACK; never mint a future ID")
    job = ack["response"]["stdout"].strip().split(";")[0]
    p.require(ack["job_id"] == job, "Exact scheduler actual ACK identity")
    for ref in (ownership_ref, storage_ref, source_review_ref, parent_review_ref):
        load(ref)
    return publish(Path(value["binding"]), dict(status=p.BINDING, owner="Root", job_id=job,
        nonce=value["nonce"], configuration=value["configuration"], intent=intent_ref, ack=ack_ref,
        request=request_ref, ownership=ownership_ref, storage_reservation=storage_ref,
        independent_source_config_review=source_review_ref, independent_completed_parent_review=parent_review_ref,
        actual_release_performed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(p.canonical(p.native.seal(args.output, request(p.desc(args.configuration), args.nonce))).decode())
