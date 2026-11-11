"""Official input identity exporter compatible with the frozen S5 authority API.

No inference, predictions, scores, judges, network or scheduler RPC. The complete
contract is emitted only from a Root-reviewed complete 15-metric input scope,
verified raw/support/media source closures and a preserved authorized legacy
projection. Individual complete banks can be useful while other banks remain
explicitly NOT_READY; a partial report is not a pool or evaluation PASS.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import parsers
from .roles_projection import seal
from experiments.root_native_input_authority_20261023 import pool_authority_v1 as authority

METRICS = ("VQAv2", "GQA", "VizWiz", "ScienceQA_image", "TextVQA", "POPE",
           "MME_perception", "MME_cognition", "MMBench_EN", "MMBench_CN", "SEED_image",
           "LLaVA_Bench_in_the_Wild", "MM_Vet", "MMMU_validation", "NoCaps_validation")
ALLOWED_PARSERS = {
    "VQAv2": {"c121-author-jsonl-v1"}, "GQA": {"reserved-id-image-projection-v1"},
    "VizWiz": {"c121-author-jsonl-v1"}, "ScienceQA_image": {"c121-scienceqa-image-v1"},
    "TextVQA": {"c121-author-jsonl-v1"}, "POPE": {"c121-author-jsonl-v1"},
    "MME_perception": {"c121-author-jsonl-v1"}, "MME_cognition": {"c121-author-jsonl-v1"},
    "MMBench_EN": {"c121-mmbench-tsv-v1"}, "MMBench_CN": {"c121-mmbench-tsv-v1"},
    "SEED_image": {"c121-author-jsonl-v1"}, "LLaVA_Bench_in_the_Wild": {"c121-author-jsonl-v1"},
    "MM_Vet": {"c121-author-jsonl-v1"}, "MMMU_validation": set(),
    "NoCaps_validation": {"nocaps-validation-images-v1"},
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def desc(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class NotReady(Exception):
    pass


def checked_bytes(description):
    require(isinstance(description, dict) and set(description) == {"path", "bytes", "sha256"},
            "Exact source path/bytes/SHA descriptor required")
    require(type(description["bytes"]) is int and 0 < description["bytes"] <= 512 * 1024 ** 2,
            "Input metadata exceeds bounded single-file scope; explicit streaming adapter required")
    path = Path(description["path"])
    require(path.is_absolute() and ".." not in path.parts, "Canonical absolute source path required")
    if not path.is_file():
        raise NotReady("MISSING_EXACT_SOURCE")
    require(path.stat().st_size == description["bytes"], "Source byte count differs")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == description["sha256"], "Source SHA differs")
    return payload


def checked(description):
    return parsers.json_load(checked_bytes(description))


def review_for(review, kind, subject):
    require(review["schema"] == "root-input-only-frozen-scope-review-v1" and
            review["status"] == "PASS_ROOT_FROZEN_INPUT_SCOPE" and review["kind"] == kind and
            review["subject"] == subject and review["outcome_tuning"] is False,
            "Exact Root pre-outcome input scope review required")


def validate_scope(scope):
    require(scope["schema"] == "root-official-protected-input-scope-v1" and
            scope["status"] == "FROZEN_OFFICIAL_INPUT_SCOPE" and scope["outcome_tuning"] is False,
            "Pre-outcome official input scope required")
    require(scope["metrics"] == list(METRICS) and set(scope["banks"]) == set(METRICS),
            "All15 frozen metrics are required; no smaller substitute roster")
    roles = []
    for metric in METRICS:
        bank = scope["banks"][metric]
        require(isinstance(bank, dict) and bank.get("metric") == metric,
                "Exact metric intent required, including not-ready banks")
        require(isinstance(bank.get("edition"), str) and bank["edition"] and
                isinstance(bank.get("split"), str) and bank["split"], "Explicit official edition/split required")
        require(bank.get("role") == "official/" + metric, "Official roles cannot overwrite legacy/confirmation roles")
        require(bank.get("qa_access") in ("authorized", "image_only"), "Explicit authorized QA boundary required")
        if metric in ("GQA", "NoCaps_validation"):
            require(bank["qa_access"] == "image_only", "Existing reserved GQA/NoCaps image-only boundary required")
        roles.append(bank["role"])
    require(len(roles) == len(set(roles)), "Duplicated metric role")
    # Both MME scores share one benchmark but have disjoint category supports.
    # Their source descriptors and category partition must be frozen explicitly.
    perception, cognition = scope["banks"]["MME_perception"], scope["banks"]["MME_cognition"]
    if perception.get("components") and cognition.get("components"):
        left = {cat for part in perception["components"] for cat in part.get("included_image_directories", [])}
        right = {cat for part in cognition["components"] for cat in part.get("included_image_directories", [])}
        require(left and right and not left & right, "Disjoint frozen MME perception/cognition partition required")


def source_support(rows, component):
    return [{"component": component, "id": row["id"], "image": row["image"]} for row in rows]


def image_record(role, key, value, proof_cache=None):
    require(isinstance(value, dict) and {"namespace", "image_id", "width", "height", "rgb_sha256",
                                      "encoded_sha256", "encoded_bytes", "source_proof", "source_proof_key"} <= set(value),
            "Complete encoded+decoded image provenance required")
    require(type(value["width"]) is int and value["width"] > 0 and
            type(value["height"]) is int and value["height"] > 0 and
            type(value["encoded_bytes"]) is int and value["encoded_bytes"] > 0, "Positive image dimensions/bytes required")
    for field in ("rgb_sha256", "encoded_sha256"):
        require(isinstance(value[field], str) and len(value[field]) == 64
                and all(x in "0123456789abcdef" for x in value[field]), "Exact image digest required")
    authority.source_image_key(value["namespace"], value["image_id"])
    # Verify the referenced immutable producer receipt exists and matches, not a
    # bare self-asserted RGB status. Root's scope binds the exact inventory/proof.
    cache_key = digest(value["source_proof"])
    proof_cache = {} if proof_cache is None else proof_cache
    if cache_key not in proof_cache:
        proof_cache[cache_key] = checked(value["source_proof"])
    proof = proof_cache[cache_key]
    require(proof.get("status") in ("PASS_ALL398_NATIVE_ID_ORIGINAL_RGB_PROTECTION",
                                   "PASS_ALL_REQUESTED_OFFICIAL_IMAGE_RGB_IDENTITIES"),
            "Completed actual RGB producer receipt required")
    require(isinstance(proof.get("images"), dict) and value["source_proof_key"] in proof["images"],
            "Exact image entry must exist in immutable RGB producer receipt")
    observed = proof["images"][value["source_proof_key"]]
    require(all(observed.get(name) == value[name] for name in
                ("width", "height", "rgb_sha256", "encoded_sha256", "encoded_bytes")),
            "Image metadata differs from actual RGB source proof")
    native_key = "native_image_id" if "native_image_id" in observed else "image_id"
    require(str(observed.get(native_key)) == str(value["image_id"]), "Native image identity differs from RGB source proof")
    result = {"role": role, "key": key, **{name: value[name] for name in
              ("namespace", "image_id", "width", "height", "rgb_sha256")}}
    aliases = value.get("aliases", [])
    require(isinstance(aliases, list), "Explicit source aliases list required")
    normalized = []
    for alias in aliases:
        require(set(alias) == {"namespace", "image_id"}, "No inferred cross-dataset aliases")
        identity = authority.source_image_key(alias["namespace"], alias["image_id"])
        require(identity not in normalized, "Duplicate source alias")
        normalized.append(identity)
    if aliases:
        result["aliases"] = deepcopy(aliases)
    return result


def build_bank(spec):
    """Return a complete bank only if every frozen source/support/media matches."""
    metric, role = spec["metric"], spec["role"]
    if spec.get("components") is None or spec.get("media") is None:
        raise NotReady("EXACT_RAW_SUPPORT_OR_RGB_DESCRIPTOR_NOT_BOUND")
    components = spec["components"]
    require(isinstance(components, list) and components, "Explicit complete source component roster required")
    media = checked(spec["media"])
    require(media["schema"] == "root-authorized-protected-image-inventory-v1" and
            media["status"] == "FROZEN_ENCODED_RGB_INPUT_IDENTITIES" and media["metric"] == metric and
            media["edition"] == spec["edition"] and media["split"] == spec["split"],
            "Exact official edition/split image inventory required")
    images, qas, support, used, sources, proof_cache = [], [], [], set(), [], {}
    seen_components, seen_qas = set(), set()
    for component in components:
        name = component["name"]
        require(isinstance(name, str) and name and name not in seen_components, "Unique source components required")
        seen_components.add(name)
        require(component["parser"] in ALLOWED_PARSERS[metric],
                "Exact official source format has no reviewed adapter for this metric")
        require(component["source_provenance"]["edition"] == spec["edition"] and
                component["source_provenance"]["split"] == spec["split"] and
                component["source_provenance"].get("official_reference"), "Explicit official source provenance required")
        options = {**component.get("options", {}), "qa_access": spec["qa_access"]}
        rows = parsers.parse(checked_bytes(component["source"]), component["parser"], options)
        if metric == "SEED_image":
            require(component.get("official_seed_annotation"), "Original SEED image/video membership source required")
            seed = checked(component["official_seed_annotation"])
            require(isinstance(seed.get("questions"), list) and seed["questions"], "Original SEED questions membership required")
            wanted = [parsers.identity(row["question_id"]) for row in seed["questions"] if row["data_type"] == "image"]
            require(wanted and len(wanted) == len(set(wanted)), "Complete unique official SEED image QA support required")
            rows = [row for row in rows if row["id"] in set(wanted)]
            require({row["id"] for row in rows} == set(wanted), "Author file misses official SEED image questions")
        # MME category membership must be explicit and frozen; never call all
        # questions perception AND cognition merely to fill two cells.
        if metric.startswith("MME_"):
            categories = component.get("included_image_directories")
            require(isinstance(categories, list) and categories and len(categories) == len(set(categories)),
                    "Frozen MME category partition required")
            rows = [row for row in rows if row["image"].split("/", 1)[0] in categories]
            require(rows, "Frozen MME category support empty")
            from collections import Counter
            require(set(Counter(row["image"] for row in rows).values()) == {2},
                    "Complete MME paired-question image support required")
        actual_support = source_support(rows, name)
        expected = checked(component["support"])
        require(expected["schema"] == "root-frozen-official-ordered-support-v1" and
                expected["metric"] == metric and expected["edition"] == spec["edition"] and
                expected["split"] == spec["split"] and expected["component"] == name and
                expected["source"] == component["source"] and expected["rows"] == actual_support and
                expected["rows_sha256"] == digest(actual_support),
                "Complete ordered raw-source QA/image support differs from frozen authority")
        # The separate support projection must identify its official upstream;
        # file existence/count/status alone never establishes completeness.
        require(expected.get("official_membership_source") and expected.get("producer_source"),
                "Independent official membership provenance required")
        checked_bytes(expected["official_membership_source"])
        checked_bytes(expected["producer_source"])
        sources.append({"component": name, "source": component["source"], "support": component["support"],
                        "provenance": component["source_provenance"]})
        for row in rows:
            reference = name + ":" + row["image"]
            require(reference in media["images"], "Complete official image/RGB coverage missing")
            if "inline_encoded_sha256" in row:
                require(media["images"][reference]["encoded_sha256"] == row["inline_encoded_sha256"] and
                        media["images"][reference]["encoded_bytes"] == row["inline_encoded_bytes"],
                        "Official TSV encoded image differs from actual RGB provider input")
            image_key = role + ":" + reference
            if reference not in used:
                images.append(image_record(role, image_key, media["images"][reference], proof_cache))
                used.add(reference)
            qa_key = name + ":" + row["id"]
            require(qa_key not in seen_qas, "Duplicate compound QA identity")
            seen_qas.add(qa_key)
            if spec["qa_access"] == "authorized":
                qas.append({"role": role, "key": qa_key, "question": row["question"],
                            "image_key": image_key, "answers": []})
        support.extend(actual_support)
    require(set(media["images"]) == used, "Missing/extra metric image inventory changes frozen support")
    result = {"schema": "root-official-input-bank-v1", "status": "COMPLETE_EXACT_INPUT_IDENTITIES_ONLY",
              "metric": metric, "edition": spec["edition"], "split": spec["split"],
              "role": role, "role_scope": {"qa_access": spec["qa_access"],
                  "images": [x["key"] for x in images], "qas": [x["key"] for x in qas], "conversations": []},
              "images": images, "qas": qas, "conversations": [],
              "ordered_input_support_sha256": digest(support), "original_support_records": len(support),
              "source_inputs": sources, "media": spec["media"], "specification_sha256": digest(spec),
              "predictions_or_judges_required": False, "benchmark_scoring_complete": False}
    authority.ProtectedIndex({"schema": "native-protected-input-contract-v1",
        "status": "FROZEN_AUTHORIZED_INPUT_IDENTITIES", "normalization": authority.NORMALIZATION,
        "outcome_tuning": False, "role_scopes": {role: result["role_scope"]},
        "images": images, "qas": qas, "conversations": []})
    return result


def assemble(scope, scope_descriptor, root_review_descriptor, banks, legacy, output):
    """Freeze S5's input-only document, preserving the exact legacy contract."""
    require(set(banks) == set(METRICS), "No full contract until all15 input banks are complete")
    require(legacy["schema"] == "root-authorized-legacy-protected-projection-v1" and
            legacy["status"] == "FROZEN_AUTHORIZED_LEGACY_INPUT_IDENTITIES" and
            legacy["confirmation_QA_access"] == "image_only", "Reviewed closed-QA-safe legacy projection required")
    roles = checked(legacy["role_manifest"])
    # roles-only projection provenance must pin the original full legacy source.
    projection = checked(legacy["roles_projection_receipt"])
    require(projection["roles"] == legacy["role_manifest"] and
            projection["source"]["bytes"] == 108392717 and
            projection["source"]["sha256"] == "151b44c923f449e7472c64815357a65a09976bb70cfff33d24467b6d7c62ca8f" and
            projection["audit"]["closed_payload_strings_decoded"] == 0,
            "Exact legacy ID-only projection provenance required")
    document = {"schema": "native-protected-input-contract-v1", "status": "FROZEN_AUTHORIZED_INPUT_IDENTITIES",
                "normalization": authority.NORMALIZATION, "outcome_tuning": False,
                "role_manifest": legacy["role_manifest"], "role_scopes": deepcopy(legacy["role_scopes"]),
                "images": deepcopy(legacy["images"]), "qas": deepcopy(legacy["qas"]),
                "conversations": deepcopy(legacy["conversations"])}
    for metric in METRICS:
        bank = banks[metric]
        role = bank["role"]
        require(role not in document["role_scopes"] and bank["specification_sha256"] == digest(scope["banks"][metric]),
                "Official bank/specification collision")
        document["role_scopes"][role] = bank["role_scope"]
        for field in ("images", "qas", "conversations"):
            document[field].extend(bank[field])
    protocol = {"schema": "native-authority-protection-protocol-v1", "status": "FROZEN_PRE_OUTCOME_INPUT_CONTRACT",
                "role_manifest": document["role_manifest"], "normalization": authority.NORMALIZATION,
                "outcome_tuning": False, "role_scopes": document["role_scopes"],
                **{field + "_sha256": authority.row_hash(document[field]) for field in ("images", "qas", "conversations")},
                "official_metrics": list(METRICS), "official_input_scope": scope_descriptor,
                "Root_input_scope_review": root_review_descriptor,
                "legacy_projection": scope["legacy_projection"],
                "not_evaluation_runtime_or_scoring_completion": True}
    authority.verify_protection_authority(document, roles, protocol)
    authority.ProtectedIndex(document)
    output = Path(output)
    seal(output / "INPUT_PROTECTION_PROTOCOL.json", protocol)
    document["evaluation_protocol"] = desc(output / "INPUT_PROTECTION_PROTOCOL.json")
    seal(output / "PROTECTED_INPUTS.json", document)
    return {"contract": desc(output / "PROTECTED_INPUTS.json"),
            "protocol": document["evaluation_protocol"]}


def run(config_descriptor, output):
    config = checked(config_descriptor)
    require(config["schema"] == "root-official-input-producer-config-v1", "Versioned input producer configuration required")
    scope = checked(config["scope"])
    validate_scope(scope)
    root_review = checked(config["Root_scope_review"])
    review_for(root_review, "official15-input-scope", config["scope"])
    # All data work remains inside a registered owned CPU+GPU allocation. Root
    # reviews and registers the finite producer; this script never submits jobs.
    allocation = authority.require_allocation()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    banks, statuses = {}, {}
    for metric in METRICS:
        try:
            bank = build_bank(scope["banks"][metric])
            target = output / "banks" / (metric + ".json")
            seal(target, bank)
            banks[metric] = bank
            statuses[metric] = {"status": bank["status"], "bank": desc(target),
                                "original_support_records": bank["original_support_records"],
                                "images": len(bank["images"]), "qas": len(bank["qas"])}
        except NotReady as error:
            statuses[metric] = {"status": "NOT_READY", "reason_code": str(error)}
        except (ValueError, KeyError, TypeError) as error:
            # Malformed/corrupt input is not transformed into a success or a
            # smaller bank. Preserve compact failure type, no QA contents.
            statuses[metric] = {"status": "INVALID_INPUT_OR_UNSUPPORTED_ADAPTER", "error_type": type(error).__name__}
    complete = None
    legacy = None
    legacy_status = "NOT_BOUND"
    if scope.get("legacy_projection"):
        try:
            legacy = checked(scope["legacy_projection"])
            legacy_status = "EXACT_PROJECTION_PRESENT_NOT_YET_FULL_CONTRACT_VALIDATED"
        except NotReady:
            legacy_status = "MISSING_EXACT_LEGACY_PROJECTION"
    if len(banks) == len(METRICS) and legacy is not None:
        complete = assemble(scope, config["scope"], config["Root_scope_review"], banks, legacy, output)
        legacy_status = "LEGACY_NONTRAIN_AND_CLOSED_CONFIRMATION_VALIDATED"
    report = {"schema": "root-official-input-producer-result-v1",
              "status": "COMPLETE_INPUT_ONLY_CONTRACT" if complete else "PARTIAL_INPUT_IDENTITIES_NOT_COMPLETE_PROTECTION",
              "configuration": config_descriptor, "scope": config["scope"], "allocation": allocation,
              "completed_banks": len(banks), "required_banks": len(METRICS), "banks": statuses,
              "legacy_status": legacy_status, "complete": complete,
              "predictions_acquired": 0, "judges_called": 0, "student_release_authorized": False}
    # Attempt-specific output required. Never overwrite an earlier failed/partial
    # result; banks themselves are immutable and safely reusable when unchanged.
    seal(output / "RESULT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(desc(Path(args.config).absolute()), Path(args.output).absolute()), sort_keys=True))
