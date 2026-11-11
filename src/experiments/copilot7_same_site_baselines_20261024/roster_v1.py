"""Finite scientific intents and exact-budget handoff; never a job controller."""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_campaign_v8_20261024 import selection as native
from experiments.root_native_input_authority_20261023 import pool_authority_v1 as authority
from experiments.root_native_baselines_20261023 import pins


SCHEMA = "s7-same-site-baseline-plan-v1"
TRAINING_SEEDS = (17, 29, 43)
SELECTION_SEED = 1234
PERCENTAGES = (5, 10, 20)
PROTECTED_INPUT = {
    "bytes": 86961849,
    "sha256": "6c9dbfc9ee43fb2530cc0cd9409af5e6bce67ed35c5d4a873241a466f28c7a8a",
}
EVALUATION_PROTOCOL = {
    "bytes": 10641277,
    "sha256": "61d13b79bf2eaff41e1e93b47ac844f4ebac3adf2781378f2fcd77c4497cd1e8",
}
METHODS = {
    "uniform": {"kind": "simple_control", "implementation": "native.seeded_hash"},
    "coincide": {
        "kind": "published_implementation", "author_commit": pins.COIN_PIN,
        "disclosure": "Native TinyLLaVA2B features/operator; common source-order training adaptation.",
    },
    "datatailor": {
        "kind": "published_implementation", "author_commit": pins.TAILOR_PIN,
        "disclosure": "Native pre-RMSNorm/SVD/Ward/floor operator; BF16/eager feature adaptation.",
    },
    "source_sqrt": {
        "kind": "heuristic_control",
        "implementation": "native.capped_Hamilton_sqrt_original_source_counts",
    },
}
TRAINING_GATES = [
    "Root_exact_recipe_model_processor_binding",
    "Root_actual_trainer_and_missing_only_restart_qualification",
    "qualified_site_and_exact_local_pool_transport",
    "independent_exact_source_config_review",
    "exclusive_function_ownership_and_measured_storage",
    "shared_admission_lock_intent_ACK_and_actual_site_limits",
    "Root_two_General_slots_preserved",
]


def fraction(percent):
    c.require(type(percent) is int and percent in PERCENTAGES, "Only frozen 5/10/20-percent budgets")
    return f"0.{percent:02d}"


def _percentages(values):
    values = tuple(values)
    c.require(values and len(set(values)) == len(values), "Nonempty distinct budget percentages required")
    for value in values:
        fraction(value)
    return tuple(sorted(values))


def _cell(site, method, percent, seed):
    gates = TRAINING_GATES[:]
    if method in ("coincide", "datatailor"):
        gates += ["actual_native_publication_feature_runtime_qualification",
                  "complete_exact_final_pool_operator_inputs_and_independent_output_review"]
    return {
        "id": f"s7-{site}-{method}-p{percent:02d}-s{seed}-q{SELECTION_SEED}",
        "site": site, "method": method, "role": "student", "percent": percent,
        "fraction": fraction(percent), "training_seed": seed, "selection_seed": SELECTION_SEED,
        "selection_key": f"{percent}:{method}", "method_provenance": deepcopy(METHODS[method]),
        "requested_K": None, "actual_K": None, "selection": None,
        "release_authorized": False, "required_gates": gates,
    }


def draft(site="babel", percentages=(10,)):
    c.require(site in ("babel", "aces"), "Choose one complete Babel or ACES block")
    percentages = _percentages(percentages)
    cells = [_cell(site, method, percent, seed)
             for percent in percentages for method in METHODS for seed in TRAINING_SEEDS]
    references = [{
        "id": f"s7-{site}-full-s{seed}", "site": site, "method": "full", "role": "full_reference",
        "fraction": "1", "training_seed": seed, "selection_seed": SELECTION_SEED,
        "selection_key": "full", "requested_K": None, "actual_K": None, "selection": None,
        "reused_in_percentages": list(percentages), "release_authorized": False,
        "required_gates": TRAINING_GATES[:],
    } for seed in TRAINING_SEEDS]
    return {
        "schema": SCHEMA, "dispatch": "ROOT_V2_S7_20261024", "owner": "Session7",
        "status": "DRAFT_NO_FINAL_POOL_OR_RUNTIME_AUTHORITY", "site": site,
        "percentages": list(percentages), "selection_seed": SELECTION_SEED,
        "training_seeds": list(TRAINING_SEEDS), "pool": None, "eligibility": None,
        "eligible_N": None, "target_K": None, "recipe_binding": None,
        "cells": cells, "full_references": references,
        "starting_model": {
            "id": f"s7-{site}-start", "role": "untouched_model_evaluation",
            "training_runs": 0, "evaluation_owner": "Session6", "shared_across_training_seeds": True,
        },
        "matched_count_rule": "Native floors retained; compare each underfilled publisher to exact-K uniform/source_sqrt.",
        "source_order_rule": "Whole original conversations, common original source order; disclose COINCIDE order adaptation.",
        "readiness": {"pool_verified": False, "selections_bound": False, "training_authorized": False},
        "census": {"planned_selected_data_cells": len(cells), "planned_full_references": len(references),
                   "submitted": 0, "running": 0, "completed": 0, "qualified": 0},
        "research_rounds_completed": 0, "scientific_claim_authorized": False,
    }


def _pool(pool_description, eligibility_description):
    authority.require_allocation()
    eligibility = c.checked(eligibility_description)
    for field, expected in (("protected_contract", PROTECTED_INPUT),
                            ("evaluation_protocol", EVALUATION_PROTOCOL)):
        description = eligibility[field]
        c.require(all(description[key] == value for key, value in expected.items()),
                  "Stale/foreign current Root " + field)
    return authority.validate_pool(pool_description, eligibility_description)


def freeze(pool_description, eligibility_description, site="babel", percentages=(10,)):
    pool, rows, ids = _pool(pool_description, eligibility_description)
    result = draft(site, percentages)
    count = len(ids)
    targets = {str(percent): count * percent // 100 for percent in result["percentages"]}
    c.require(all(value > 0 for value in targets.values()), "Declared budget would contain no conversations")
    c.require(pool["count"] == len(rows) == count, "Central eligible denominator differs")
    result.update(
        status="FINAL_POOL_VERIFIED_PREPARATION_NOT_TRAINING_AUTHORITY",
        pool=pool_description, eligibility=eligibility_description, eligible_N=count, target_K=targets,
        input_authority=pool["input_authority"], pool_record_ids_sha256=c.key(ids),
        pool_source_indices_sha256=c.key(pool["source_indices"]),
    )
    result["readiness"]["pool_verified"] = True
    for cell in result["cells"]:
        cell["requested_K"] = targets[str(cell["percent"])]
    for cell in result["full_references"]:
        cell["requested_K"] = count
    return result


def validate_plan(description):
    plan = c.checked(description)
    c.require(plan["schema"] == SCHEMA and plan["pool"] is not None, "A genuinely pool-bound plan is required")
    expected = freeze(plan["pool"], plan["eligibility"], plan["site"], plan["percentages"])
    c.require(plan == expected, "Frozen roster, site, seed, budget, provenance or reference binding changed")
    return plan


def _published(selected, requested, pool_description):
    c.require(selected["method"] in ("coincide", "datatailor"), "Not a supported faithful published operator")
    c.require(selected["requested_K"] == requested and selected["author_floor_not_topped_up"] is True
              and selected["matched_actual_K_uniform_required"] == (selected["count"] != requested),
              "Published requested/actual-K or native floor disclosure differs")
    pins.verify(selected["operator_sources"])
    review = c.checked(selected["method_review"])
    candidate = c.checked(review["candidate"])
    expected = {**candidate, "status": "PASS_SELECTION", "method_review": selected["method_review"]}
    c.require(candidate["status"] == "PENDING_INDEPENDENT_SELECTOR_IMPLEMENTATION_AND_OUTPUT"
              and selected == expected and review["sources"] == selected["operator_sources"],
              "Published output does not reproduce its exact independently reviewed candidate")
    operator = c.checked(selected["operator_inputs"])
    c.require(operator["status"] == "COMPLETE_PUBLISHED_OPERATOR_INPUTS"
              and operator["owner"]["pool"] == pool_description and operator["method"] == selected["method"]
              and operator["sources"] == selected["operator_sources"],
              "Published operator provenance is not for this exact final pool/source")


def bind_selections(plan_description, selections):
    plan = validate_plan(plan_description)
    required = {cell["selection_key"] for cell in plan["cells"]} | {"full"}
    c.require(set(selections) == required, "Every method/budget and one reused FULL selection is required")
    pool, _, ids = _pool(plan["pool"], plan["eligibility"])
    loaded = {}
    with native.authority_eligibility(plan["eligibility"]):
        for key in sorted(required):
            method = "full" if key == "full" else key.split(":", 1)[1]
            percent = None if key == "full" else int(key.split(":", 1)[0])
            requested = len(ids) if percent is None else plan["target_K"][str(percent)]
            value = native.validate_selection(selections[key], plan["pool"], method,
                                              "1" if percent is None else fraction(percent))
            c.require(value["selection_seed"] == SELECTION_SEED, "Selection seed differs from paired common construction")
            wanted = set(value["record_ids"])
            c.require(value["record_ids"] == [ident for ident in ids if ident in wanted],
                      "Common original source order required; never drop or alter conversation turns")
            if method in ("coincide", "datatailor"):
                _published(value, requested, plan["pool"])
            loaded[key] = value
    result = deepcopy(plan)
    result["status"] = "SELECTIONS_BOUND_NOT_TRAINING_OR_EVALUATION_AUTHORITY"
    result["readiness"]["selections_bound"] = True
    trajectories = {}
    for cell in result["cells"] + result["full_references"]:
        key = cell["selection_key"]
        value = loaded[key]
        cell.update(selection=selections[key], actual_K=value["count"])
        identity = c.key({
            "site": plan["site"], "pool_records": pool["records_json"]["sha256"],
            "selected_records": value["records_json"]["sha256"], "training_seed": cell["training_seed"],
            "recipe": "UNBOUND_REQUIRED_BEFORE_ADMISSION",
        })
        cell["prepared_data_trajectory_key"] = identity
        trajectories.setdefault(identity, []).append(cell["id"])
    matched = {}
    for key, selected in loaded.items():
        if selected["method"] not in ("coincide", "datatailor") or selected["count"] == selected["requested_K"]:
            continue
        percent = int(key.split(":", 1)[0])
        for method in ("uniform", "source_sqrt"):
            control_key = f"{percent}:{method}:K{selected['count']}"
            entry = matched.setdefault(control_key, {
                "selection_key": control_key, "site": plan["site"], "base_method": method,
                "kind": "exact_actual_K_control_not_learned_utility", "percent": percent,
                "fraction": fraction(percent), "requested_K": plan["target_K"][str(percent)],
                "actual_K": selected["count"], "selection_seed": SELECTION_SEED,
                "training_seeds": list(TRAINING_SEEDS), "published_anchors": [],
                "selection_materialized": False, "release_authorized": False,
                "native_trainer_integration_required": method == "source_sqrt",
            })
            entry["published_anchors"].append(selections[key])
    result["matched_count_requests"] = list(matched.values())
    result["prepared_unique_data_trajectories"] = trajectories
    result["census"]["prepared_unique_primary_and_full_data_trajectories"] = len(trajectories)
    result["census"]["planned_matched_count_cells"] = 3 * len(matched)
    return result


def _matched_source_sqrt_payload(plan_description, published_description, groups_description):
    plan = validate_plan(plan_description)
    pool, rows, ids = _pool(plan["pool"], plan["eligibility"])
    with native.authority_eligibility(plan["eligibility"]):
        published = native.validate_selection(published_description, plan["pool"])
        c.require(published["method"] in ("coincide", "datatailor"), "Reviewed native publisher required")
        percent = next((value for value in plan["percentages"]
                        if fraction(value) == published["fraction"]), None)
        c.require(percent is not None and published["selection_seed"] == SELECTION_SEED,
                  "Published budget/construction outside the frozen roster")
        _published(published, plan["target_K"][str(percent)], plan["pool"])
        c.require(0 < published["count"] < published["requested_K"], "Only genuine native underfill needs this control")
        groups = native.groups_for(groups_description, plan["pool"], ids)
    chosen = native.choose(ids, "source_sqrt", published["count"], SELECTION_SEED, groups=groups)
    original = dict(zip(ids, rows))
    return {
        "schema": "s7-source-sqrt-actual-K-selection-v1", "status": "PREPARED_NOT_NATIVE_TRAINER_RELEASE_AUTHORITY",
        "method": "source_sqrt_actualK", "base_method": "source_sqrt", "pool": plan["pool"],
        "eligibility": plan["eligibility"], "plan": plan_description, "published": published_description,
        "fraction": published["fraction"], "selection_seed": SELECTION_SEED,
        "requested_K": published["requested_K"], "actual_K": len(chosen),
        "record_ids": chosen, "groups": groups_description,
        "algorithm_source": c.desc(native.__file__), "producer": c.desc(__file__),
        "claim": "exact_count_heuristic_control_not_learned_utility",
        "Root_additive_native_trainer_integration_required": True,
    }, [original[ident] for ident in chosen]


def matched_source_sqrt(plan_description, published_description, groups_description, output):
    value, rows = _matched_source_sqrt_payload(plan_description, published_description, groups_description)
    output = Path(output)
    value["records_json"] = c.seal(output / "records.json", rows)
    return c.seal(output / "SELECTION.json", value)


def validate_matched_source_sqrt(description):
    value = c.checked(description)
    c.require(value["schema"] == "s7-source-sqrt-actual-K-selection-v1", "Wrong exact-K control schema")
    plan = validate_plan(value["plan"])
    c.require(value["pool"] == plan["pool"] and value["eligibility"] == plan["eligibility"],
              "Matched control final pool/eligibility differs")
    expected, rows = _matched_source_sqrt_payload(value["plan"], value["published"], value["groups"])
    c.require(c.checked(value["records_json"]) == rows
              and value == {**expected, "records_json": value["records_json"]},
              "Exact-K source-sqrt record, budget, algorithm or provenance drift")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("draft", "freeze", "bind"))
    parser.add_argument("--site", choices=("babel", "aces"), default="babel")
    parser.add_argument("--percent", type=int, nargs="+", default=[10])
    parser.add_argument("--pool", help="JSON file containing Root's exact POOL descriptor")
    parser.add_argument("--eligibility", help="JSON file containing independent eligibility descriptor")
    parser.add_argument("--plan", help="JSON file containing this client's frozen-plan descriptor")
    parser.add_argument("--selections", help="JSON mapping selection_key to exact native selection descriptor")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.phase == "draft":
        result = draft(args.site, args.percent)
    elif args.phase == "freeze":
        c.require(args.pool and args.eligibility, "Root pool and independent eligibility descriptors required")
        result = freeze(c.read(args.pool), c.read(args.eligibility), args.site, args.percent)
    else:
        c.require(args.plan and args.selections, "Frozen plan and complete native selections required")
        result = bind_selections(c.read(args.plan), c.read(args.selections))
    print(c.canonical(c.seal(args.output, result)).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
