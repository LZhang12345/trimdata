import ast
from pathlib import Path

from experiments.root_native_campaign_v7_20261024 import common as c
from experiments.root_native_trained_loader_v4_20261026 import generation as bridge
from experiments.s1_gqa_native_conversion_diagnosis_v1_20261027 import diagnose as native_gqa

ROOT = Path(__file__).resolve().parents[2]
ANCESTOR_CONTROL = Path(
    "/home/luyangz/data_selection_multimodal/collaboration/root-current-four-bank-eval-v7-20261027")
METADATA_DELTAS = {
    "experiments/root_current_four_bank_eval_v1_20261027/admission.py",
    "experiments/root_current_four_bank_eval_v1_20261027/contract.py",
    "experiments/root_current_four_bank_eval_v1_20261027/checks.py",
    "experiments/root_current_four_bank_eval_v1_20261027/worker.py",
}
GQA_SCORER = "experiments/root_native_trained_evaluation_v3_20261026/banks.py"
GQA_HELPER = "experiments/s1_gqa_native_conversion_diagnosis_v1_20261027"
QUALIFIED = "PASS_ACTUAL_SOURCE7_GENERATION_AND_NATIVE_GQA_CONVERSION_REUSE_V1"
REVIEWED = "PASS_INDEPENDENT_ACTUAL_SOURCE7_GENERATION_AND_NATIVE_GQA_CONVERSION_REUSE_V1"


def expected_gqa_source(original):
    edits = (
        (
            "from experiments.session6_scienceqa_bank_v1_20261024.scoring import score_complete\n",
            "from experiments.session6_scienceqa_bank_v1_20261024.scoring import score_complete\n"
            "from experiments.s1_gqa_native_conversion_diagnosis_v1_20261027.diagnose import "
            "native_prediction, REFERENCE_URL, REFERENCE_SHA256\n",
        ),
        ("normalization='none'", "normalization='LLaVA-c121-rstrip-dot-then-lower'"),
        ("e['raw']['text']==targets[r['semantic_key']]",
         "native_prediction(e['raw']['text'])==targets[r['semantic_key']]"),
        (
            "reference_sources/GQA_c121_recommended_eval.py'))",
            "reference_sources/GQA_c121_recommended_eval.py'),\n"
            "        prediction_converter=dict(source_URL=REFERENCE_URL,source_sha256=REFERENCE_SHA256))",
        ),
    )
    for before, after in edits:
        c.require(original.count(before) == 1, "Exact original GQA source substitution point")
        original = original.replace(before, after, 1)
    return original


def function_bodies(path):
    tree = ast.parse(path.read_text())
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def source_equivalence(configuration):
    ancestor = Path(configuration["ancestor_source_root"])
    expected = configuration["ancestor_sources"]
    target = configuration["target_sources"]
    c.require(ancestor.name == "root_current_four_bank_eval_src_v7_20261027"
              and len(expected) == 417, "Exact qualified Source7 ancestor")
    for relative, digest in expected.items():
        c.require(c.sha(ancestor / relative) == digest, "Original Source7 code changed")
        if relative not in METADATA_DELTAS | {GQA_SCORER}:
            c.require(target.get(relative) == digest and c.sha(ROOT / relative) == digest,
                      "Generation, model reconstruction, decoding, bank or scorer math changed")
    for filename, allowed in (
        ("worker.py", {"sources"}),
        ("checks.py", {"complete_student", "documents"}),
    ):
        relative = "experiments/root_current_four_bank_eval_v1_20261027/" + filename
        before, after = function_bodies(ancestor / relative), function_bodies(ROOT / relative)
        c.require(set(before) == set(after)
                  and all(before[name] == after[name] for name in before if name not in allowed),
                  "Only the declared completion/source metadata function may change")
    c.require((ROOT / GQA_SCORER).read_text() ==
              expected_gqa_source((ancestor / GQA_SCORER).read_text()),
              "Only exact native GQA conversion may change; all other bank scoring stays byte-identical")
    for relative, digest in target.items():
        c.require(c.sha(ROOT / relative) == digest, "Declared target metadata/math bytes changed")
    return dict(ancestor_sources=expected, target_sources=target,
                byte_identical_numerical_source_count=len(expected) - len(METADATA_DELTAS) - 1,
                original_worker_run_actor_storage_resume_and_model_support_AST_unchanged=True,
                original_generation_and_model_files_byte_identical=True,
                original_non_GQA_scoring_byte_identical=True,
                GQA_only_exact_native_conversion_and_provenance_changed=True)


def native_conversion_evidence(evaluation, load=c.checked, describe=c.desc):
    ref = evaluation["native_GQA_conversion_readout"]
    value = load(ref)
    peer_ref = describe(evaluation["native_GQA_conversion_independent_review_path"])
    peer = load(peer_ref)
    c.require(value["status"] == "ACTUAL_GQA_AUTHOR_CONVERSION_DIAGNOSIS_REQUIRES_INDEPENDENT_ACCEPTANCE"
              and value["questions_per_model"] == 12578 and len(value["models"]) == 7
              and value["native_converter_URL"] == native_gqa.REFERENCE_URL
              and value["native_converter_sha256"] == native_gqa.REFERENCE_SHA256
              and value["exact_native_prediction_conversion"] == 'text.rstrip(".").lower()'
              and value["new_NN"] == value["paid_answers_regenerated"] == 0
              and value["original_scores_or_paid_answers_modified"] is False
              and value["raw_QA_GT_or_credit_arrays_exported"] is False
              and peer["status"] == "PASS_INDEPENDENT_ACTUAL_NATIVE_GQA_CONVERSION_DIAGNOSIS_V1"
              and peer["independent"] is True and peer["readout"] == ref
              and peer["reviewer"] not in (None, "", "Session1", "S1", "Session1-author-delegated"),
              "Actual paid native GQA conversion and independent protocol acceptance required")
    registration = load(value["registration"])
    c.require(registration["status"] == "ACTUAL_S1_CURRENT_OWNED_CPU_STEP_REGISTERED"
              and registration["actual"] == value["actual"]
              and value["actual"]["numeric_step"].isdecimal(),
              "Actual registered conversion diagnosis, not a local fixture")
    for name in ("__init__.py", "diagnose.py"):
        relative = GQA_HELPER + "/" + name
        c.require(evaluation["sources"][relative] == value["actual"]["sources"][relative]
                  == c.sha(ROOT / relative), "Exact actual-qualified native conversion helper")
    return dict(readout=ref, independent_actual_review=peer_ref,
                model_or_generation_changes=False, GQA_prediction_conversion_only=True)


def cold_prefix(evaluation, load=c.checked, describe=c.desc):
    provider_ref = describe(ANCESTOR_CONTROL / "PREFIX_PROVIDER.json")
    provider = load(provider_ref)
    binding = load(provider["binding"])
    first = load(binding["configuration"])
    c.require(provider["status"] == "ROOT_BOUND_ACTUAL_SOURCE7_COLD_PREFIX_PROVIDER_V1"
              and provider["owner"] == "Root"
              and binding["status"] == "ROOT_BOUND_CURRENT12_OFFICIAL_JOIN_FOUR_BANK_EVALUATION_V7"
              and first["student_id"] == "production-student-v1-10-source_sqrt-s43"
              and first["source7_prefix_mode"] == "PRODUCER"
              and binding["configuration"] == provider["configuration"]
              and binding["nonce"] == first["nonce"]
              and first["actual_bound19_asset_readiness"] == evaluation["actual_bound19_asset_readiness"],
              "Genuine original first Source7 cold-model producer")
    ref = describe(provider["prefix_path"])
    c.require(ref["path"] == str(Path(first["output"]) / "r0/COLD_MODEL_QA_PREFIX.json")
              and provider["independent_review_path"] == str(ANCESTOR_CONTROL / "PREFIX_REVIEW.json"),
              "Actual original cold-prefix paths")
    value = load(ref)
    peer_ref = describe(provider["independent_review_path"])
    peer = load(peer_ref)
    c.require(peer["status"] == "PASS_INDEPENDENT_ACTUAL_SOURCE7_COLD_MODEL_QA_PREFIX_V1"
              and peer["independent"] is True and peer["readout"] == ref
              and peer["binding"] == provider["binding"]
              and peer["configuration"] == binding["configuration"],
              "Actual independently accepted original cold-model QA proof")
    c.require(value["status"] == "ACTUAL_SOURCE7_COLD_MODEL_TWO_NATIVE_QA_PREFIX_NOT_FULL_BANK"
              and value["configuration"] == binding["configuration"]
              and value["actual"]["job_id"] == binding["job_id"]
              and value["actual"]["nonce"] == binding["nonce"] and value["actual"]["restart"] == 0
              and value["accepted_native_QAs"] == 2 and value["bank"] == "POPE"
              and value["scientific_FULL_bank_or_SOTA_PASS"] is False
              and value["no_extra_or_replayed_QAs"] is True
              and value["model_identity_unchanged_after_prefix"] is True,
              "Actual two-QA unchanged model prefix, not future or journal-only evidence")
    calls = value["generation_bridge"]
    c.require(calls["status"] == "ACTUAL_NATIVE_BRIDGE_PREFIX_INSIDE_UNCHANGED_OFFICIAL_AUDIT_CONTEXT"
              and calls["calls"] == len(calls["records"]) == 2 and calls["policy"] == bridge.POLICY
              and calls["adapter_active"] is True and calls["adapter_merged"] is False
              and calls["unchanged_official_Recorder_wrapper_authenticated"] is True
              and calls["generate_or_preparation_hook_mutated_by_observer"] is False
              and all(all(record[field] is True for field in
                  ("success", "input_object_unchanged", "all_keyword_objects_unchanged",
                   "prepare_hook_restored", "original_PEFT_generate_invoked", "output_object_unchanged"))
                      for record in calls["records"]), "Exact qualified original PEFT/native bridge calls")
    from experiments.root_current_four_bank_eval_v1_20261027.worker import strict_loaded_support
    strict_loaded_support(load(value["model_reconstruction"]))
    c.require(load(value["model_identity"]) == load(value["post_prefix_model_identity"]),
              "Original prefix parameter/buffer/tokenizer/processor identity")
    return dict(provider=provider_ref, prefix=ref, independent_review=peer_ref,
                original_sources=first["sources"], original_qualified_NN_calls_reused=2,
                new_NN=0, no_new_cold_model_or_science_claim=True)


def validate_reuse(evaluation, load=c.checked, describe=c.desc):
    result_ref = describe(evaluation["numerical_reuse_result_path"])
    result = load(result_ref)
    peer = load(describe(evaluation["numerical_reuse_actual_review_path"]))
    c.require(result["status"] == QUALIFIED
              and result["target_source_root"] == evaluation["source_root"]
              and result["target_sources"] == evaluation["sources"]
              and result["actual_bound19_asset_readiness"] == evaluation["actual_bound19_asset_readiness"]
              and result["new_NN"] == 0 and result["checkpoint_writes"] is False
              and result["model_loading_and_generation_unchanged"] is True
              and result["non_GQA_scoring_unchanged"] is True
              and result["GQA_score_changed_to_exact_author_native_conversion"] is True
              and peer["status"] == REVIEWED and peer["independent"] is True
              and peer["readout"] == result_ref and peer["target_sources"] == evaluation["sources"]
              and peer["reviewer"] not in (None, "", "Session1", "S1", "Session1-author-delegated"),
              "Actual metadata-only interface/source equivalence qualification plus independent acceptance")
    c.require(result["native_GQA_conversion"] == native_conversion_evidence(evaluation, load, describe),
              "Actual accepted GQA protocol evidence changed")
    return result_ref
