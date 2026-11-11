"""Metadata-only V7 config drafts; no forged ownership, review or runtime PASS."""
import argparse
import copy
from pathlib import Path

from . import common as c
from .masking import RECORDS_SHA256


def drafts(spec):
    c.require(spec["schema"] == "root-native-v10-config-materialization-v1"
              and spec["owner"] == "Root", "Wrong V7 materialization request")
    target = spec["native_target"]
    modules, packet = c.target_modules(target)
    profile = c.checked(spec["site_profile"])
    c.require(profile["site"] == "babel" and profile["gpu_type"] == "L40S",
              "This materializer is Babel-only; H100 requires its own reviewed transport study")
    configs = {}
    for arm in ("reference", "resume"):
        base = c.checked(spec["base_configurations"][arm])
        config = copy.deepcopy(base)
        config.pop("review", None)
        config.update(
            id=spec["configuration_ids"][arm], sources=packet["sources"],
            site_profile=spec["site_profile"], output=spec["outputs"][arm],
            storage_reservation=spec["storage_reservations"][arm],
            recipe=copy.deepcopy(modules["config"].RECIPE),
            report=spec["reports"][arm],
        )
        c.require({k:v for k,v in base["recipe"].items() if k!="masking"} ==
                  {k:v for k,v in config["recipe"].items() if k!="masking"},
                  "Only the declared masking identity may change from the original recipe")
        config["providers"].update(spec.get("provider_overrides", {}))
        config["providers"]["ownership"] = spec["ownership"][arm]
        c.require(config["role"] == "qualify" and config["method"] == "full"
                  and config["fraction"] == "1" and config["qualification_max_steps"] == 8,
                  "Only the unchanged full engineering eight-step trajectory may be drafted")
        c.require(config["output"] != base["output"]
                  and c.within(config["output"], profile["output_root"]),
                  "Fresh approved arm output required; never resume the failed V5 study")
        c.require(config["providers"]["ownership"] != base["providers"]["ownership"]
                  and config["storage_reservation"] != base["storage_reservation"],
                  "Root must register new V7 ownership and collective storage")
        pool = c.checked(spec["pool_metadata_copy"])
        c.require(all(spec["pool_metadata_copy"][name]==config["providers"]["pool"][name]
                      for name in ("bytes","sha256")), "Staged metadata is not the exact original pool")
        c.require(pool["count"] == 1024 and pool["records_json"]["sha256"] == RECORDS_SHA256,
                  "Original exact1024 fixture must be preserved")
        claim = c.checked(config["providers"]["ownership"])
        selection = c.checked(config["providers"]["selection"])
        function = c.key(dict(recipe=config["recipe"],pool_records=pool["records_json"]["sha256"],
            selected_records=selection["records_json"]["sha256"],training_seed=config["training_seed"],
            role=config["role"],qualification_max_steps=config["qualification_max_steps"],
            replicate_site=config["replication_block"]["replicate_site"]))
        c.require(claim["qualification_study"] == spec["study_id"]
                  and claim["qualification_arm"] == arm and claim["function_identity"]==function,
                  "New arm claim must name the exact V7 study and recomputed V5 function")
        storage = c.checked(config["storage_reservation"])
        c.require(storage["output"] == config["output"]
                  and storage["function_identity"]==function
                  and storage.get("allow_new_intermediate_retirement", False) is False
                  and storage["retained_accepted_states"] >= (2 if arm == "resume" else 1),
                  "Three accepted states and actual cut4 retention are required")
        c.require("media_transport" not in config["providers"]
                  and "runtime_images" not in config["providers"],
                  "Do not silently adapt canonical Babel inputs into transported inputs")
        configs[arm] = config
    c.require(configs["reference"]["output"] != configs["resume"]["output"],
              "Reference and restart outputs must be distinct")
    for name in ("recipe", "training_seed", "selection_seed", "allocation"):
        c.require(configs["reference"][name] == configs["resume"][name],
                  "Unpaired V7 configuration: " + name)
    for name in ("assets", "environment", "pool", "selection", "eligibility"):
        c.require(configs["reference"]["providers"][name] == configs["resume"]["providers"][name],
                  "Unpaired V7 provider: " + name)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configs = drafts(c.read(args.spec))
    output = Path(args.output)
    result = {arm: c.seal(output / arm / "CONFIG_DRAFT.json", config)
              for arm, config in configs.items()}
    print(c.canonical(dict(
        status="METADATA_DRAFTS_REQUIRE_EXACT_REVIEW_AND_COMPUTE_MOUNT_VALIDATION",
        configurations=result, source_target=c.read(args.spec)["native_target"],
        model_or_dataset_payloads_loaded=False, job_submitted=False,
    )).decode())


if __name__ == "__main__":
    main()
