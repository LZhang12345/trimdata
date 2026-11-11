"""Prepare immutable source/config metadata, never DATA, models or submissions."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from experiments.root_native_campaign_20261023 import common as c
from . import worker


LOCAL = Path(__file__).resolve().parents[2]
REMOTE = Path("/home/luyangz/data_selection_multimodal")
CONTROL = Path("collaboration/copilot7-fullpool-published-20261024/datatailor-v2")
CODE = REMOTE / "session7_fullpool_published_src_v2"
PACKAGE = Path("experiments/copilot7_fullpool_published_20261024")


def remote_description(local, relative):
    value = c.desc(local)
    return dict(value, path=str(REMOTE / relative))


def assemble_sources(root=LOCAL):
    native = c.read(root / "collaboration/session7-native-official-execution-20261024/datatailor32-v4/SPEC_V4A.json")
    client = c.read(root / "collaboration/copilot7-same-site-baselines-20261024/SOURCE_CONFIG_PACKET_V1.json")
    sources = {}
    client_sources = dict(client["reused_source_closure"])
    for name, description in client["own_sources"].items():
        c.require((root / name).stat().st_size == description["bytes"], "Frozen client source size changed: " + name)
        c.require(name not in client_sources or client_sources[name] == description["sha256"],
                  "Conflicting client own/reused source: " + name)
        client_sources[name] = description["sha256"]
    for group in (native["dependencies"], native["sources"], client_sources):
        for name, digest in group.items():
            c.require(name not in sources or sources[name] == digest, "Conflicting frozen source: " + name)
            c.require(c.sha(root / name) == digest, "Frozen prerequisite/client source changed: " + name)
            sources[name] = digest
    authority = Path("experiments/root_native_input_authority_20261023")
    for path in list((root / authority).glob("*.py")) + list((root / PACKAGE).glob("*")):
        if path.is_file() and path.suffix in (".py", ".sbatch", ".md"):
            sources[str(path.relative_to(root))] = c.sha(path)
    supplements = {
        "experiments/root_native_campaign_v3_20261023/__init__.py": "13f7ea29e8b29aac202ec30e9ae5ae75fa1e9e1e007fd24887c616d7f4e997df",
        "experiments/root_native_campaign_v3_20261023/common.py": "e21cf1ef0cfafdf4e295bbabfc052f0638a18e0e36ea43a532e440572ba77a2d",
        "experiments/root_native_campaign_v3_20261023/selection.py": "afdae279635f8f9e6e501bb368358776004db8efb01217a49a978d717060b7ec",
    }
    c.require(all(sources.get(name) == digest for name, digest in supplements.items()),
              "Root V2 independently reviewed exact supplements missing")
    return sources, native, supplements


def staged_published_sources(stage):
    script = ("import sys,json;sys.path.insert(0,sys.argv[1]);"
              "from experiments.root_native_baselines_20261023 import pins;"
              "print(json.dumps(pins.sources(),sort_keys=True))")
    result = subprocess.check_output([sys.executable, "-I", "-B", "-c", script, str(stage)],
                                     cwd=stage, text=True, timeout=60)
    return json.loads(result)


def prepare(stage):
    directory = LOCAL / CONTROL
    directory.mkdir(parents=True, exist_ok=True)
    sources, native, supplements = assemble_sources()
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    for name, digest in sources.items():
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        c.require(not target.exists() or c.sha(target) == digest, "Existing source stage differs")
        shutil.copyfile(LOCAL / name, target)
        c.require(c.sha(target) == digest, "Staged source bytes differ")
    published_sources = staged_published_sources(stage)
    c.require(all(sources.get(name) == digest for name, digest in published_sources.items()),
              "Native dynamic published verifier discovered an unregistered source")
    handoff = c.read(LOCAL / "collaboration/root-background-research-cycles-20261024/1d6344462caa4938acd98e7001e6824f/POOL_DENOMINATOR_HANDOFF_V1.json")
    c.require(handoff["N"] == worker.N and handoff["K"] == worker.K, "Genuine Root denominator changed")
    output = "/data/user_data/luyangz/native_campaign_20261023/session7-fullpool-published-v2"
    policy_rel = Path("collaboration/root-queue-recovery-and-array-20261024/AUTHOR_GENERAL_EIGHT_SLOTS_V1.md")
    author_path = directory / "AUTHOR_INFERENCE_AUTHORIZATION.json"
    c.seal(author_path, dict(owner="Session7", fullpool_feature_acquisition=True, student_release=False,
        provenance="Author's urgent execution instruction in existing Copilot Session7; not a review/runtime PASS",
        instruction="Execute finalpool DATA consumer and fullpool published COINCIDE/DataTailor feature/selection acquisition as a reviewed large missing-only array NOW; inference-only work does not wait student optimizer restart.",
        allocation_policy=remote_description(LOCAL / policy_rel, policy_rel)))
    author = remote_description(author_path, CONTROL / author_path.name)
    reviews = [
        Path("collaboration/session6-official-eval-v2-20261024/independent-prerequisite-payload-reviews-20261024/DT32_10916022_PAYLOAD_REVIEW_V1.json"),
        Path("collaboration/session6-official-eval-v2-20261024/independent-prerequisite-payload-reviews-20261024/DT32_10916022_WARNING_REVIEW_ADDENDUM_V2.json"),
    ]
    constructor = ("experiments/root_native_datatailor_20261023/features.py",
        "experiments/root_native_baselines_20261023/operators.py",
        "experiments/root_native_baselines_20261023/store.py",
        "experiments/root_native_campaign_v3_20261023/config.py",
        "experiments/root_native_campaign_v3_20261023/resume_adapter.py",
        "experiments/root_native_qualification_20261023/masking.py")
    proof_path = directory / "SCOPED_CONSTRUCTOR_REUSE.json"
    c.seal(proof_path, dict(method="datatailor", pool=handoff["pool"], feature_contract=worker.tailor.CONTRACT,
        fullpool_previously_qualified=False, author_acquisition_instruction=author,
        closed_reviews=[remote_description(LOCAL / path, path) for path in reviews],
        expected_review_statuses=["S6_V1_PRESERVED_WITH_B1_CLOSED_BY_V2", "PASS_SCOPED_DT32_PAYLOAD_PREFIX"],
        constructor_sources={name: sources[name] for name in constructor},
        engineering_only_checks="Production uses unchanged native compute(engineering=False): skips oracle/full-tree rehashes, not numerical operations",
        no_generation_student_accuracy_or_scientific_round_PASS=True))
    profile_path = directory / "FEATURE_PROFILE.json"
    c.seal(profile_path, dict(python="/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python",
        code_root=str(CODE), cache_root="/data/hf_cache", output_root=output, site="babel",
        account="cx", partition="general", qos="normal", gpu_type="L40S",
        scope="Frozen constructor CLI profile only; not a training-site runtime qualification"))
    measurement_sources = {**native["dependencies"], **native["published_sources"]}
    claim_path = directory / "OWNERSHIP.json"
    claim = dict(owner="Session7", method="datatailor", pool=handoff["pool"], shard_count=16,
        measurement_sources=measurement_sources, environment=native["inputs"]["environment"],
        feature_assets=native["inputs"]["assets"], output=output + "/datatailor",
        status="ACTIVE_EXCLUSIVE_S7_FINALPOOL_FEATURE_PLAN_CLAIM",
        model_updates=0, evaluation_or_Root_pool_producer_owned=False)
    c.seal(claim_path, claim)
    runtime = c.read(LOCAL / "collaboration/session7-native-official-execution-20261024/datatailor32-v4/ACTUAL_RUNTIME_ADDENDUM_V1.json")
    spec = dict(schema="s7-finalpool-published-feature-array-v1", owner="Session7", method="datatailor",
        expected_N=worker.N, expected_K=worker.K, selected_data_budget_unit="whole_original_conversation",
        pool=handoff["pool"], eligibility=handoff["eligibility"], input_authority=handoff["input_authority"],
        feature_assets=claim["feature_assets"], environment=claim["environment"],
        runtime_component=runtime["actual_component_exported"], runtime_versions={},
        python=c.read(profile_path)["python"], source_root=str(CODE), sources=sources,
        published_sources=published_sources, measurement_sources=measurement_sources,
        native_sources=native["native_sources"], feature_profile=remote_description(profile_path, CONTROL / profile_path.name),
        scoped_fidelity=remote_description(proof_path, CONTROL / proof_path.name),
        ownership=remote_description(claim_path, CONTROL / claim_path.name),
        site=dict(name="babel", user="luyangz", account="cx", partition="general", qos="normal", gpu_type="L40S"),
        allocation=dict(cpus=8, gpus=1, memory_mb=65536, minutes=120, reserve_seconds=300),
        shard_count=16, global_shard_indices=list(range(16)), array_concurrency=8,
        output=claim["output"], campaign_output=output, required_storage_bytes=20 << 30,
        storage_registry=str(REMOTE / "collaboration/copilot7-fullpool-published-20261024/storage-claims"),
        job_name="dsmm-s7-fp-dt-v2", live_function_prefix="dsmm-s7-fp-dt-",
        wrapper=str(PACKAGE / "run_datatailor.sbatch"),
        author_policy=remote_description(LOCAL / policy_rel, policy_rel),
        admission_lock=str(REMOTE / "collaboration/babel-two-track-20261019/ADMISSION.lock"),
        function_registration=str(REMOTE / "collaboration/copilot7-fullpool-published-20261024/function-intents/datatailor.json"),
        dependency="", dependency_rationale="Finalpool and independent eligibility already genuine/ready; no dependency on student optimizer or Root's separate uniform/source_sqrt manifest preparation",
        no_old_refill_or_Qwen_restart=True, scientific_rounds_completed=0)
    spec_path = directory / "CONFIG.json"
    c.seal(spec_path, spec)
    archive = directory / "SOURCE.tar.gz"
    c.require(not archive.exists(), "Preserve existing exact source archive; use a new version for repairs")
    with tarfile.open(archive, "w:gz") as stream:
        for name in sorted(sources):
            stream.add(stage / name, arcname=name, recursive=False)
    packet = dict(configuration=remote_description(spec_path, CONTROL / spec_path.name),
        local_configuration=c.desc(spec_path), sources=sources, Root_V2_supplements=supplements,
        source_archive=c.desc(archive), staged_local_source=str(stage), staged_remote_source=str(CODE),
        actual_staged_dynamic_published_verifier_count=len(published_sources),
        data_or_models_read_or_submissions=0, scoped_constructor_reuse=remote_description(proof_path, CONTROL / proof_path.name))
    c.seal(directory / "SOURCE_CONFIG_PACKET.json", packet)
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    print(prepare(args.stage))
