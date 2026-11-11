import argparse
from pathlib import Path

from . import common as c


def prepare(directory, model_result_path, media_provider_path, actual_provider_path, image_root):
    directory = Path(directory)
    model = c.read(model_result_path)
    c.require(model["status"] == c.MODEL_STATUS, "An actual model roundtrip result, not a fixture, is required")
    context = model["actual"]
    c.require(str(context["job_id"]).isdecimal() and type(context["restart"]) is int,
              "Actual immutable model attempt identity required")
    model_ref = dict(c.desc(model_result_path),
        path=str(c.model_common.BASE / "output/attempts" /
                 (context["job_id"] + "-r" + str(context["restart"])) / "RESULT.json"))
    provider_ref = dict(c.desc(media_provider_path), path=str(Path(actual_provider_path)))
    c.require(Path(actual_provider_path).is_absolute(), "Actual unchanged ACES provider receipt locator required")
    meta_base = c.ROOT / "collaboration/session6-official-eval-v2-20261024/metadata-qualification-v1"
    metadata = c.read(meta_base / "actual-2250425-r0/RESULT.json")
    metadata_ref = dict(c.desc(meta_base / "actual-2250425-r0/RESULT.json"),
        path="/scratch/user/u.lz341613/dsmm_s6_official_eval_20261024/metadata-qualification-v1/output/attempts/2250425-r0/RESULT.json")
    c.require(metadata["questions"] == c.QUESTIONS, "Reuse exact full5000 actual metadata")
    existing = c.read(meta_base / "CONFIG.json")
    protocol_path = c.ROOT / "collaboration/session6-official-eval-v2-20261024/EVALUATION_PROTOCOL_V2.json"
    scope_path = c.ROOT / "collaboration/root-input-contract-assembly-v1-20261024/ROOT_FROZEN_SCOPE_V1.json"
    config = dict(schema=c.SCHEMA, owner=c.OWNER, bank="TextVQA",
        questions=c.QUESTIONS, unique_images=c.IMAGES, encoded_bytes=c.ENCODED_BYTES,
        shards=c.SHARDS, python=c.model_common.PYTHON, cache_root=c.model_common.CACHE,
        model_result=model_ref, metadata_result=metadata_ref, media_provider=provider_ref,
        media=existing["destination_inputs"]["textvqa_MEDIA.json"],
        annotations=existing["destination_inputs"]["textvqa_annotations.json"],
        image_root=str(Path(image_root)), output=str(c.BASE / "output"),
        admission_lock=existing["admission_lock"], reserved_bytes=2 * 1024 ** 3,
        attempt_seconds=900, max_attempts=8,
        evaluation_protocol=dict(c.desc(protocol_path),
            path=str(c.BASE / "src" / protocol_path.relative_to(c.ROOT))),
        frozen_scope=dict(c.desc(scope_path), path=str(c.BASE / "src" / scope_path.relative_to(c.ROOT))),
        input_protection_protocol_sha256="61d13b79bf2eaff41e1e93b47ac844f4ebac3adf2781378f2fcd77c4497cd1e8",
        requested_phases=["prepare", "predict", "score"], sources=c.sources(),
        scope="Only fullauthorizedTextVQA5000/3166pixels: actualnativeGPU128token/eagerpreparewithfresh-processQAresumeandrepeat;finite32missing-onlyH100shards;pureCPUcpu-partition/noGRESfull-unionnative scoring. Other14metricsNOT_EVALUATED,primaryNULL.",
        no_dependency_on_student_optimizer_restart=True, no_QA_permission_expansion=True,
        training_updates=0, model_downloads=False, new_media_transfer=False)
    config["identity_comparison"] = "COMPLETE_TYPED_PREIMAGE_AND_STRICT_JSON_DOMAIN_ALL_FIELDS"
    config["automatic_continuation"] = "FINITE_CPU_AFTER_PREPARE_THEN32_AND_DEPENDENT_CPU_SCORE"
    c.validate_config(config)
    local = c.seal(directory / "CONFIG.json", config)
    remote = dict(local, path=str(c.BASE / "control/CONFIG.json"))
    packet = c.seal(directory / "SOURCE_PACKET_V1.json", dict(
        schema="session6-native-start-execution-v6-source-config",
        configuration=remote, local_configuration=local, sources=config["sources"],
        authored_tests=c.desc(c.HERE / "test_pipeline.py"),
        boundary_tests=c.desc(c.HERE / "test_boundaries.py"),
        identity_tests=c.desc(c.HERE / "test_identity.py"),
        pinned_native_identity_tests=c.desc(c.HERE / "test_native_identity.py"),
        preserved_legacy_JSON_test_source=c.desc(c.ROOT /
            "experiments/session6_native_start_execution_v5_20261024/identity.py"),
        workflow_tests=c.desc(c.HERE / "test_workflow.py"),
        phases=dict(prepare="1GPUjobphysicalsame-jobrequeue:1acceptedQAthenfresh-processmissing1+repeat2=4forwards/2reusableprefixQA",
                    predict="32singleGPUmissing-only tasks, same-job accepted-QA requeue, reuse2qualifiedprefixQA",
                    dispatch="1CPU2GiB20min/noGPUafterokprepare;admitsentire32GPU+1dependentCPUscorebundleonceunderlock",
                    score="1pureCPUallocationcpu-partition4CPU16GiB20minNOgpuGRESafteranyarray,complete-unionnative scoring"),
        no_scheduler_called=True, independent_review_required=True,
        actual_array_submitted=False, actual_bank_ready=False, scientific_rounds_completed=0))
    return dict(configuration=remote, packet=packet, finite_prediction_tasks=c.SHARDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-result", required=True)
    parser.add_argument("--media-provider", required=True)
    parser.add_argument("--actual-provider-path", required=True)
    parser.add_argument("--image-root", required=True)
    args = parser.parse_args()
    print(c.canonical(prepare(args.output, args.model_result, args.media_provider,
                              args.actual_provider_path, args.image_root)).decode())
