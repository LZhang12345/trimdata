import importlib
from pathlib import Path

from experiments.root_native_campaign_v7_20261024 import common as c
from . import parents
from . import numerics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sources():
    return {str(path.relative_to(ROOT)): c.sha(path) for path in HERE.glob("*.py")
            if not path.name.startswith("test_")}


def protocol(family):
    name = ("experiments.root_s1_shared_student_v1_20261025"
            if family == "S1" else "experiments.root_fulltrajectory_student_v4_20261025")
    return importlib.import_module(name + ".protocol"), importlib.import_module(name + ".collect")


def qualify_parent(evaluation_ref, output, actual, registration):
    evaluation = c.checked(evaluation_ref)
    contract_ref, contract, training = parents.declaration(evaluation)
    complete_ref = c.desc(contract["completion_path"])
    complete = c.checked(complete_ref)
    p, collector = protocol(contract["family"])
    parents.complete_fields(contract, training, complete_ref, complete, p)
    target = output / evaluation["student_id"]
    target.mkdir(parents=True, exist_ok=False)
    recomputed_ref = collector.collect(
        contract["original_configuration"], complete["attempts"], target / "RECOMPUTED_COMPLETE.json")
    c.require(parents.equal(c.checked(recomputed_ref), complete),
              "Unchanged full-epoch exposure/LR/objective collector must reproduce canonical completion")
    if contract["origin_kind"] == "completed_collector":
        from experiments.root_current_four_bank_eval_v1_20261027 import historical
        _, historical_ref, historical_complete, physical = historical.complete_student(evaluation)
        c.require(historical_ref == complete_ref and parents.equal(historical_complete, complete),
                  "Unchanged qualified historical Root physical-segment validation")
        origin = parents.collector_origin(contract, complete_ref, complete)
    else:
        origin = parents.recovery_origin(contract, complete_ref, complete)
        physical = [origin["physical_execution_proof"]]
    accepted = c.checked(complete["accepted"])
    c.require(accepted["identity"] == c.run_identity(training) and accepted["global_step"] == 516,
              "Actual current compute-visible final accepted state")
    custody_path = Path(contract["accepted_metadata_custody_path"])
    result_path = Path(contract["qualification_result_path"])
    approved_control = Path("/home/luyangz/data_selection_multimodal/collaboration")
    c.require(approved_control in custody_path.parents and approved_control in result_path.parents
              and ".." not in custody_path.parts and ".." not in result_path.parts,
              "Only small explicitly declared HOME metadata custody, never checkpoint payloads")
    export = c.seal(custody_path,
        dict(status="ACTUAL_READONLY_FINAL_ACCEPTED_METADATA_CUSTODY_NOT_A_WEIGHT_OR_QA_COPY",
             accepted=complete["accepted"], accepted_body=accepted,
             actual=actual, registration=registration))
    return c.seal(result_path,
        dict(status=parents.QUALIFIED, evaluation_configuration=evaluation_ref,
             parent_contract=contract_ref, training_configuration=contract["original_configuration"],
             original_native_identity=contract["original_native_identity"],
             student_completion=complete_ref, accepted=complete["accepted"],
             recomputed_unchanged_full_collector=recomputed_ref,
             historical_proofs=physical, origin_proof=origin, accepted_metadata_custody=export,
             bridge_sources=sources(), actual=actual, registration=registration,
             old_failed_accounting_preserved=True, full_epoch_exposure_LR_objective_verified=True,
             new_NN=0, checkpoint_writes=False,
             changed_loader_or_generator_math=False, new_scientific_gain_claimed=False))


def run(configuration, actual, registration):
    cfg = c.checked(configuration)
    c.require(cfg["schema"] == "s1-audited-native-parent-interface-qualification-v1"
              and cfg["owner"] == "Session1-author-delegated"
              and cfg["new_NN"] == 0 and cfg["checkpoint_writes"] is False
              and cfg["source_root"] == str(ROOT) and cfg["evaluation_configurations"],
              "Scoped readonly actual parent-interface qualification")
    output = Path(cfg["output"])
    c.require(output.is_absolute()
              and Path("/data/hf_cache/project_artifacts/DS-MultiModal") in output.parents,
              "Parent-interface evidence remains in approved project artifact storage")
    first = c.checked(cfg["evaluation_configurations"][0])
    equivalence = numerics.source_equivalence(cfg)
    conversion = numerics.native_conversion_evidence(first)
    prefix = numerics.cold_prefix(first)
    numerical_path = Path(cfg["numerical_reuse_result_path"])
    c.require(Path("/home/luyangz/data_selection_multimodal/collaboration") in numerical_path.parents
              and ".." not in numerical_path.parts, "Small declared numerical-custody result namespace")
    numerical = c.seal(numerical_path, dict(
        status=numerics.QUALIFIED, configuration=configuration,
        target_source_root=str(ROOT), target_sources=cfg["target_sources"],
        actual_bound19_asset_readiness=first["actual_bound19_asset_readiness"],
        equivalence=equivalence, original_qualified_cold_prefix=prefix,
        actual=actual, registration=registration,
        model_loading_and_generation_unchanged=True, non_GQA_scoring_unchanged=True,
        GQA_score_changed_to_exact_author_native_conversion=True, native_GQA_conversion=conversion,
        new_NN=0, checkpoint_writes=False,
        inherited_source7_model_qualification_not_relabelled_as_new_model_run=True))
    results = [qualify_parent(ref, output, actual, registration)
               for ref in cfg["evaluation_configurations"]]
    return c.seal(output / "RESULT.json",
        dict(status="ACTUAL_AUDITED_PARENT_INTERFACES_REQUIRE_INDEPENDENT_ACCEPTANCE",
             configuration=configuration, actual=actual, registration=registration,
             parents=results, numerical_reuse=numerical,
             bridge_sources=sources(), new_NN=0, checkpoint_writes=False))
