"""Actually accepted one-set/three-seed registry; never a source-only selection."""
from . import protocol as p
from experiments.copilot7_coincide_actual_review_v2_20261028 import bridge

STATUS = "ROOT_BOUND_EXACT_COINCIDE_FIXED10_SELECTION_REGISTRY_V2"


def registered_plan(ref, load=p.checked):
    plan = load(ref)
    p.require(plan["schema"] == "coincide-fixed10-accepted-registry-v2"
              and plan["status"] == STATUS and plan["owner"] == "Root"
              and plan["selected_K"] == p.K and plan["selection_seed"] == 1234
              and plan["pool"] == p.POOL and plan["recipe"] == p.native_config.RECIPE
              and plan["native_math_sources"] == p.n.sources()
              and plan["adapter_sources"] == p.sources() and plan["release_authorized"] is False,
              "Genuine current accepted COINCIDE registry, not a prepared source template")
    selected = load(plan["selection"])
    materialized = load(plan["materialized_candidate"])
    method = load(selected["method_review"])
    p.independent(method)
    p.require(selected["schema"] == "native-record-selection-v1" and selected["status"] == "PASS_SELECTION"
              and selected["method"] == "coincide" and selected["fraction"] == "0.10"
              and selected["selection_seed"] == 1234 and selected["pool"] == p.POOL
              and selected["count"] == len(selected["record_ids"]) == len(set(selected["record_ids"])) == p.K
              and selected["adaptation_label"] == p.ADAPTATION
              and materialized["schema"] == bridge.SCHEMA and materialized["status"] == bridge.PENDING
              and materialized["operator_candidate"] == p.CANDIDATE
              and materialized["record_ids"] == selected["record_ids"]
              and materialized["records_json"] == selected["records_json"]
              and method["status"] == "PASS_EXACT_SELECTOR_IMPLEMENTATION_AND_OUTPUT"
              and method["materialized_candidate_identity"] == bridge.key(materialized)
              and method["operator_candidate"] == p.CANDIDATE
              and method["records_json"] == selected["records_json"]
              and method["requested_K"] == method["actual_K"] == p.K
              and method["selected_record_ids_sha256"] == p.key(selected["record_ids"]),
              "Actual independently accepted original-record selection required before training")
    actual = load(plan["actual_output_verification"])
    p.require(actual["status"] == "ACTUAL_READONLY_OUTPUT_REVIEW_CANDIDATE_NOT_INDEPENDENT_ACCEPTANCE"
              and actual["operator_job"] == "10961650" and actual["function_identity"]
              == "bbde85dd4c12b6321399d553d3333ec9c2e855d90cc7ca73600b3e32413be212"
              and actual["new_models"] == actual["new_forwards"] == actual["new_features"] == 0
              and actual["operator_math_or_MMD_rerun"] is False
              and any(row["candidate"] == p.CANDIDATE and row["actual_K"] == p.K
                      and row["selected_record_ids_sha256"] == p.key(selected["record_ids"])
                      for row in actual["outputs"]),
              "Registered actual operator output/feature acceptance evidence, never reacquisition")
    registration = load(plan["Root_registration"])
    p.require(registration["owner"] == "Root" and registration["accepted"] is True
              and registration["selection"] == plan["selection"]
              and registration["materialized_candidate"] == plan["materialized_candidate"]
              and registration["actual_output_verification"] == plan["actual_output_verification"]
              and registration["adapter_sources"] == p.sources()
              and registration["no_live_duplicate_selected_seed_functions"] is True,
              "Root actual registry/selection/source and missing-only ownership reconciliation")
    p.require([row["id"] for row in plan["cells"]] == [p.cell_id("coincide", s) for s in p.SEEDS]
              and all(row["method"] == "coincide" and row["training_seed"] == seed
                      and row["selection"] == plan["selection"]
                      and row["new_training_required"] is True and row["duplicate_of"] is None
                      for row, seed in zip(plan["cells"], p.SEEDS)),
              "Exactly three distinct missing seed trajectories, not a budget sweep")
    p.require(set(plan["initialization_providers"]) == {"environment", "assets"},
              "Exact existing native initialization providers")
    for value in plan["initialization_providers"].values():
        load(value)
    eligibility = load(plan["eligibility"])
    p.require(eligibility["status"] == "PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY"
              and eligibility["pool"] == p.POOL, "Existing exact protected pool eligibility")
    return plan


def prepare(selection, materialized, actual_review, registration, environment, assets, eligibility,
            load=p.checked):
    value = dict(schema="coincide-fixed10-accepted-registry-v2", status=STATUS, owner="Root",
        selection=selection, materialized_candidate=materialized, actual_output_verification=actual_review,
        Root_registration=registration, pool=dict(p.POOL), eligibility=eligibility,
        initialization_providers=dict(environment=environment, assets=assets),
        selected_K=p.K, selection_seed=1234, recipe=p.native_config.RECIPE,
        native_math_sources=p.n.sources(), adapter_sources=p.sources(), release_authorized=False,
        cells=[dict(id=p.cell_id("coincide", seed), method="coincide", training_seed=seed,
                    selection=selection, new_training_required=True, duplicate_of=None) for seed in p.SEEDS])
    sentinel = object()
    registered_plan(sentinel, lambda ref: value if ref is sentinel else load(ref))
    return value
