"""Explicit H100/transport review boundary, independent of the frozen Babel gate."""
import importlib
from pathlib import Path

from experiments.root_native_qualification_v10_20261024 import common as base
from experiments.root_native_qualification_v10_20261024.common import (
    ROOT, canonical, checked, desc, key, now, read, require, seal, sha, within,
)

HERE = Path(__file__).resolve().parent
TARGET_PACKET = "a704d9716719a9ad2a0cd0890c24fec35c15bd085b7c2c98fe38ea0c98961830"
TOLERANCE = {"absolute": 1e-6, "relative": 1e-5}


def sources():
    inherited = base.sources()
    own = {str(path.relative_to(ROOT)): sha(path)
           for path in list(HERE.glob("*.py")) + list(HERE.glob("*.sbatch"))
           if not path.name.startswith("test_")}
    return dict(sorted({**inherited, **own}.items()))


def identity(config, modules):
    return modules["common"].run_identity(config)


def design_contract(study, pair):
    arms = {}
    for arm, config in pair.items():
        transport = checked(config["providers"]["media_transport"])
        arms[arm] = {
            "configuration": {name: config[name] for name in (
                "id", "role", "method", "fraction", "training_seed", "selection_seed", "allocation",
                "recipe", "sources", "site_profile", "output", "report", "checkpoint_seconds",
                "qualification_max_steps")},
            "providers": {name: config["providers"][name] for name in
                          ("assets", "environment", "pool", "selection", "eligibility")},
            "transport_contract": {name: value for name, value in transport.items() if name != "ownership"},
        }
    return {
        "schema": "h100-held-engineering-design-v10", "id": study["id"], "owner": "Root",
        "purpose": study["purpose"], "native_target": study["native_target"],
        "driver_sources": study["driver_sources"], "input_rebind_provenance": study["input_rebind_provenance"],
        "source_pool_metadata": study["source_pool_metadata"],
        "actual_prefix_proof": study["actual_prefix_proof"],
        "raw_prefix_proof_copy": study["raw_prefix_proof_copy"],
        "total_steps": 8, "interrupt_step": 4, "numeric_tolerance": TOLERANCE,
        "report": study["report"], "arms": arms,
        "late_only": ["actual_job_bound_transport_ownership", "arm_ownership", "storage", "readiness"],
    }


def validate_design(value):
    require(value["schema"] == "h100-held-engineering-design-v10" and value["owner"] == "Root"
            and value["purpose"] == "H100_TRANSPORT_ENGINEERING_ONLY"
            and value["native_target"]["source_packet"]["sha256"] == TARGET_PACKET,
            "Wrong pre-submission H100 design scope")
    review = checked(value["review"])
    require(review["status"] == "PASS_INDEPENDENT_H100_HELD_DESIGN"
            and review["design_identity"] == key({k: v for k, v in value.items() if k != "review"})
            and review["driver_sources"] == value["driver_sources"] == sources()
            and review["reviewer"] not in ("Session6", "session6_h100_native_qualification"),
            "Exact independent held H100 design review required")
    return value


def validate_rebind(provenance, pool, selected, source_pool_metadata):
    content = provenance["record_content_identity"]
    require(provenance["source_pool"] == pool["source_Babel_pool"]
            and content["target"] == pool["records_json"],
            "Engineering rebind source/target descriptors contradict the bound pool")
    require(source_pool_metadata["sha256"] == provenance["source_pool"]["sha256"]
            and source_pool_metadata["bytes"] == provenance["source_pool"]["bytes"],
            "Staged source-pool metadata differs from its authenticated original")
    original = checked(source_pool_metadata)
    require(content["source"] == original["records_json"],
            "Source record proof is not the authenticated source pool's records descriptor")
    require(pool["count"] == original["count"] == 1024
            and pool["record_ids"] == original["record_ids"]
            and pool["source_indices"] == original["source_indices"],
            "Source/target engineering record IDs or original indices differ")
    for name in ("image_manifest", "protected_exclusions"):
        require(pool[name]["sha256"] == original[name]["sha256"]
                and pool[name]["bytes"] == original[name]["bytes"],
                "Source/target engineering metadata content differs: " + name)
    require(content["source"]["sha256"] == content["target"]["sha256"]
            and content["source"]["bytes"] == content["target"]["bytes"],
            "Engineering rebind changed original record bytes")
    rows = checked(pool["records_json"])
    require(len(rows) == 1024 and checked(selected["records_json"]) == rows,
            "H100 full selection differs from unchanged1024 source conversations")
    assets = checked(pool["image_manifest"])["assets"]
    names = {row["image"] for row in rows if row.get("image")}
    require(set(assets) == names and len(assets) == len(names) == 935,
            "Actual engineering canonical/selected image cardinality differs from935")
    return {"record_sha256": content["target"]["sha256"], "records": len(rows), "images": len(names)}


def configs(study, *, reviewed=True, with_design=True):
    require(study["schema"] == "h100-transport-engineering-study-v10"
            and study["owner"] == "Root", "Wrong H100 study/owner")
    require(study["total_steps"] == 8 and study["interrupt_step"] == 4
            and study["numeric_tolerance"] == TOLERANCE, "Frozen H100 8/cut4/tolerance differs")
    require(study["driver_sources"] == sources(), "H100 driver/dependency source changed")
    require(study["purpose"] == "H100_TRANSPORT_ENGINEERING_ONLY"
            and study["native_target"]["source_packet"]["sha256"] == TARGET_PACKET,
            "H100 study cannot borrow a Babel result or replace the bound native target")
    if reviewed:
        review = checked(study["review"])
        require(review["status"] == "PASS_INDEPENDENT_H100_TRANSPORT_DRIVER_AND_STUDY"
                and review["driver_sources"] == study["driver_sources"]
                and review["study_identity"] == key({k: v for k, v in study.items() if k != "review"})
                and review["reviewer"] not in ("Session6", "session6_h100_native_qualification"),
                "Exact independent H100 review required")
    # This generic source loader is reused, not the frozen Babel study validator.
    modules, packet = base.target_modules(study["native_target"])
    require(len(packet["sources"]) == 91, "Wrong reviewed nativeV7/91 source closure")
    proof = checked(study["actual_prefix_proof"])
    from experiments.root_native_qualification_v10_20261024.preflight import source_equivalence_bridge,V5_PROOF_SHA
    require(study['actual_prefix_proof']['sha256']==V5_PROOF_SHA,'Exact actual measured V5 prefix summary required')
    source_equivalence_bridge(proof,packet['sources'])
    require(proof["status"] == "PASS_ACTUAL_TOKENIZER_PARITY_ONLY"
            and proof["records_checked"] == 1024 and proof["violations"] == []
            and proof["changed_labels"] == ["642651:djWVD94_0"]
            and proof["all_input_ids_identical"] is True
            and proof["all_unaffected_labels_identical"] is True
            and proof["all_independent_prefix_oracle_identical"] is True,
            "Exact qualified V5/tokenizer prefix proof required; never train the known-failing old fixture")
    raw_copy = study["raw_prefix_proof_copy"]
    require(all(raw_copy[name] == proof["raw_remote_only"][name] for name in ("bytes", "sha256")),
            "H100 staged prefix proof differs from the authenticated Babel raw evidence")
    raw = checked(raw_copy)
    from experiments.root_native_qualification_v10_20261024.masking import PREFIX_SOURCE
    require(raw["status"] == proof["status"] and raw["repair_sources"] == proof['repair_sources']
            and raw["oracle_sources"][PREFIX_SOURCE] == sha(ROOT / PREFIX_SOURCE)
            and raw["violations"] == [] and raw["repaired_bad_record_ids"] == []
            and raw["records_checked"] == len(raw["records"]) == 1024
            and raw["native_bad_record_ids"] == raw["changed_label_record_ids"] == ["642651:djWVD94_0"]
            and len({row["record_id"] for row in raw["records"]}) == 1024
            and all(row["input_ids_identical"] and row["ordinary_labels_identical"]
                    and row["independent_prefix_oracle_identical"]
                    and row["repaired_unexpanded_targets"] > 0 and row["repaired_expanded_targets"] > 0
                    for row in raw["records"]),
            "Relocated actual prefix/source evidence differs")
    result = {arm: checked(study["native_configurations"][arm]) for arm in ("reference", "resume")}
    native_plan = importlib.import_module(study["native_target"]["package"] + ".plan")
    for arm, config in result.items():
        require(config["sources"] == packet["sources"] and config["recipe"] == modules["config"].RECIPE,
                "H100 configuration operator differs")
        require(config["role"] == "qualify" and config["method"] == "full"
                and config["fraction"] == "1" and config["qualification_max_steps"] == 8,
                "Only complete engineering8-update arms supported")
        allocation = config["allocation"]
        require(allocation["gpus"] == 1 and allocation["microbatch"] == 4
                and modules["config"].accumulation(1, 4) == 32,
                "H100 qualification is world1/micro4/GA32")
        profile = modules["config"].validate_profile(checked(config["site_profile"]))
        require(profile["site"] == "aces" and profile["gpu_type"] == "h100"
                and profile["python"] == "/scratch/user/u.lz341613/dsmm_native_campaign_20261023/"
                "envs/native-py310-cu121-v1/bin/python", "Exact completed H100 native route required")
        require(profile["code_root"] == str(ROOT), "Execute installed H100 code root")
        require("runtime_images" not in config["providers"] and "media_transport" in config["providers"],
                "Reviewed logical H100 config must require fresh physical aliases, not contain one")
        provider = {name: checked(ref) for name, ref in config["providers"].items()}
        source_equivalence_bridge(proof,packet['sources'],raw=raw,assets=provider['assets'],environment=provider['environment'])
        pool = provider["pool"]
        require(pool["schema"] == "native-engineering-pool-v1" and pool["engineering_only"] is True
                and pool["count"] == 1024 and pool["runtime_transport_required"] is True,
                "Exact logical1024 engineering pool required, never a scientific authority")
        require(provider["media_transport"]["scope"] == "ENGINEERING_ONLY"
                and provider["media_transport"]["canonical_image_manifest"] == pool["image_manifest"],
                "Foreign/broad transport descriptor")
        claim = provider["ownership"]
        require(claim["qualification_study"] == study["id"] and claim["qualification_arm"] == arm,
                "Root must register both exact engineering arm claims")
        require(claim["owner"] == "Root" and claim["site_profile"] == config["site_profile"]
                and claim["status"] == "ACTIVE_EXCLUSIVE_NATIVE_PLAN_CELL_CLAIM"
                and claim["plan"] == config["plan"] and claim["cell_id"] == config["id"]
                and claim["pool"] == config["providers"]["pool"] and claim["replicate_site"] == "aces",
                "Exact owned H100 plan-cell acquisition claim required")
        require(config["checkpoint_seconds"] > allocation["minutes"] * 60,
                "Periodic checkpoints would change this bounded parity study")
        storage = checked(config["storage_reservation"])
        require(storage["status"] == "PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION"
                and storage["output"] == config["output"]
                and storage.get("allow_new_intermediate_retirement", False) is False
                and storage.get("retained_accepted_states", 0) >= (2 if arm == "resume" else 1),
                "Retain actual cut4 and final states; no retirement in this study")
        # Runtime aliases are intentionally late. Only the reviewed plan-cell
        # native path permits this without forging a config review afterward.
        review = checked(config["review"])
        plan = checked(config["plan"])
        cell = native_plan.get_cell(plan, config["id"])
        require(review["status"] == "PASS_INDEPENDENT_EXACT_PLAN_AND_SOURCE"
                and review["plan_identity_sha256"] == key(plan)
                and review["sources"] == plan["sources"] == config["sources"]
                and plan["recipe"] == config["recipe"],
                "Exact native engineering plan review required before physical staging")
        for field in ("method", "fraction", "training_seed", "selection_seed", "role"):
            require(config[field] == cell[field], "H100 logical cell differs: " + field)
        require(cell["site"] == "aces" and cell["gpus"] == 1 and cell["microbatch"] == 4,
                "Wrong native engineering plan cell site/allocation")
    reference, resume = result["reference"], result["resume"]
    for name in ("recipe", "sources", "site_profile", "training_seed", "selection_seed", "allocation"):
        require(reference[name] == resume[name], "Unpaired H100 " + name)
    for name in ("assets", "environment", "pool", "selection", "eligibility", "media_transport"):
        require(reference["providers"][name] == resume["providers"][name], "Unpaired H100 provider " + name)
    require(reference["output"] != resume["output"], "Engineering arms need distinct output ownership")
    require(isinstance(study["minimum_launch_seconds"], int) and study["minimum_launch_seconds"] >= 600,
            "Explicit staging/training reserve required")
    provenance = checked(study["input_rebind_provenance"])
    require(provenance["schema"] == "engineering-logical-site-rebind-provenance-v1"
            and provenance["scope"] == "ENGINEERING_ONLY"
            and provenance["target_pool"] == reference["providers"]["pool"]
            and provenance["actual_source_records"] == 1024 and provenance["actual_canonical_images"] == 935
            and provenance["all_original_conversation_turns_preserved"] is True
            and provenance["record_ids_identical"] is True and provenance["source_indices_identical"] is True,
            "S5 exact engineering record/index/conversation rebind proof required")
    validate_rebind(provenance, checked(reference["providers"]["pool"]),
                    checked(reference["providers"]["selection"]), study["source_pool_metadata"])
    require(raw["pool"] == provenance["source_pool"]
            and {row["record_id"] for row in raw["records"]} ==
            set(checked(reference["providers"]["pool"])["record_ids"]),
            "Actual prefix proof belongs to another source construction")
    if with_design:
        design = validate_design(checked(study["design"]))
        require({k: v for k, v in design.items() if k != "review"} == design_contract(study, result),
                "Final H100 study drifted from the independently reviewed pre-submission design")
    return result, modules


def physical_config(logical, alias_descriptor, *, validate_native, run_identity):
    """Attach only the validated physical alias; no review or logical identity is rewritten."""
    import copy
    from experiments.root_native_media_transport_20261023 import transport
    selected = checked(logical["providers"]["selection"])
    alias = transport.validate(alias_descriptor, logical["providers"]["pool"], selected["records_json"])
    require(alias["scope"] == "ENGINEERING_ONLY"
            and alias["transport"] == logical["providers"]["media_transport"],
            "Physical transport differs from exact logical configuration")
    result = copy.deepcopy(logical)
    result["providers"]["runtime_images"] = alias_descriptor
    require(run_identity(result) == run_identity(logical), "Physical alias changed semantic native identity")
    validate_native(result)
    return result
