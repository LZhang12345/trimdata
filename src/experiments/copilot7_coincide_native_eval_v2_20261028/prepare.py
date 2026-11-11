"""Real current trained-loader/bank config integration; Root supplies actual refs."""
import argparse
import json
import os
from pathlib import Path
import tarfile

from . import protocol as p, model
from experiments.copilot7_coincide_actual_review_v2_20261028.prepare import source_closure

CONTROL = Path(__file__).resolve().parents[2] / "collaboration/session7-coincide-actual-output-review-20261028/v2/evaluation"


def packet(output=CONTROL):
    output = Path(output)
    initial = [path for path in p.HERE.iterdir() if path.suffix in (".py", ".sbatch")]
    initial.extend(p.ROOT / name for name in p.native.sources())
    full = source_closure(initial)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "EVALUATION_EXECUTABLE_SOURCE_V2.tar.gz"
    p.require(not archive.exists(), "Do not replace an existing frozen evaluation source")
    with tarfile.open(archive, "w:gz") as sink:
        for name in full:
            sink.add(p.ROOT / name, arcname=name, recursive=False)
    return p.native.seal(output / "EVALUATION_EXECUTABLE_HANDOFF_V2.json", dict(
        status="IMPLEMENTED_CURRENT_TRAINED_LOADER_FOUR_BANK_ENTRY_REQUIRES_REAL_FULL_PARENTS",
        package=p.PACKAGE, source_root=str(p.SOURCE_ROOT), sources=full,
        source_archive=p.desc(archive), qualified_loader_body_proof=model.qualified_body_proof(),
        exact_same_native_PEFT071_generation_bridge=True, counts=p.COUNTS,
        GQA_conversion="prediction.rstrip('.').lower(); original target exact equality",
        no_eval_targets_for_selection_or_extra_tuning=True,
        finite_cells=[dict(id=p.native.cell_id("coincide", seed) + "-fourbank",
                          native_parent=p.native.cell_id("coincide", seed),
                          full_parent_updates=516, questions=28505, seed=seed, actual_job=None)
                      for seed in p.native.SEEDS],
        actual_new_jobs=0, requested_GPUh_if_Root_admits=18, actual_GPU_seconds=0,
        actual_complete_parent_or_future_SHA=None,
        real_preparation_entry=p.PACKAGE + ".prepare",
        real_worker=p.PACKAGE + ".worker", Root_request_and_ACK_entry=p.PACKAGE + ".bind",
        missing_actual_inputs=["accepted native parent configuration/Rootbinding", "actual full516 completion and independent parent acceptance",
            "existing qualified Root bank/loader basis and current official media", "genuine new source/config/loader-scope review",
            "Root actual exclusive function ownership/storage/intent/ACK/spool/release"],
        partial_QAs_durable_missing_only=True, no_automatic_self_requeue_or_fake_ACK=True,
        Root_owns_independent_actual_successor_transition=True,
        source_tests_not_GPU_qualification_or_fullbank_result=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-only", action="store_true")
    parser.add_argument("--output", type=Path, default=CONTROL)
    for name in ("basis", "training", "training-binding", "execution-registration"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--nonce")
    args = parser.parse_args()
    if args.packet_only:
        value = packet(args.output)
    else:
        p.require(os.getuid() == 2708899 and all(getattr(args, name) for name in
                  ("basis", "training", "training_binding", "execution_registration", "nonce")),
                  "Actual existing configs, admitted parent and owned metadata registration")
        from experiments.root_production_student_adapter_v1_20261024.launch import require_metadata_registration
        require_metadata_registration(p.desc(args.execution_registration))
        value = p.native.seal(args.output, p.configuration(p.desc(args.basis), p.desc(args.training),
                              p.desc(args.training_binding), args.nonce))
    print(json.dumps(value, sort_keys=True))
