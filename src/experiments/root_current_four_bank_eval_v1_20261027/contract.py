"""Exactly twelve current10% models and genuine V4 prerequisite bindings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from experiments.root_gqa_bank_qualification_v1_20261026 import worker as u
from experiments.root_native_trained_evaluation_v3_20261026 import banks
from experiments.root_native_trained_loader_v4_20261026 import generation as bridge
from experiments.root_comparator_four_bank_eval_v1_20261026 import contract as old
from experiments.root_native_trained_qual_review_step_v6_20261027 import consumer as registered

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HOME = old.HOME
CONTROL = HOME / "collaboration/root-current-four-bank-eval-v7-20261027"
SOURCE_ROOT = HOME / "root_current_four_bank_eval_src_v7_20261027"
OUTPUT = Path("/data/hf_cache/project_features/DS-MultiModal/current_four_bank_eval_20261027/v7")
LOCK = old.LOCK
PACKAGE = "experiments.root_current_four_bank_eval_v1_20261027"
NAME = "dsmm-current-four-bank-v7"
COMMENT = "dsmm-current-four-bank-v7:"
STATUS = "ROOT_CONFIGURED_CURRENT12_OFFICIAL_JOIN_FOUR_BANK_EVALUATION_V7"
BINDING_STATUS = "ROOT_BOUND_CURRENT12_OFFICIAL_JOIN_FOUR_BANK_EVALUATION_V7"
REVIEW_STATUS = "PASS_INDEPENDENT_CURRENT12_OFFICIAL_JOIN_FOUR_BANK_SOURCE_CONFIG_V7"
SCOPE = "FINAL_CURRENT_ROOT_AND_S1_TWELVE10PCT_MODELS_FOUR_COMPLETE_OFFICIAL_BANKS_V1"
RESOURCES = dict(cpus=8, memory_mb=65536, gpus=1, minutes=360)
OFFICIAL_BANK_TEMPLATE = old.OFFICIAL_BANK_TEMPLATE
STUDENTS = {
    **{"production-student-v1-10-" + method + "-s" + str(seed): job
       for (method, seed), job in old.JOBS.items()},
    **{"s1-student-v1-10-" + arm + "-s" + str(seed): str(10937436 + index)
       for index, (arm, seed) in enumerate(
           (arm, seed) for arm in ("0R", "H0") for seed in (17, 29, 43))},
}
BANK_FIELDS = ("bank_inputs", "banks", "counts", "decoding", "scienceqa_decoding",
               "frozen_scope", "official_POPE_configuration", "phase3_configuration",
               "actual_media_observation")
require, checked, desc, literal = u.require, u.checked, u.desc, u.literal


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def key(value):
    return hashlib.sha256(canonical(value)).hexdigest()


REGISTERED_SOURCES={'experiments/root_native_trained_qual_review_step_v6_20261027/__init__.py': '8c6e3d48b6c9d8f7633dde881f5b22d7205fec2a1b440b3ccb93b8d287dc316d', 'experiments/root_native_trained_qual_review_step_v6_20261027/consumer.py': '0fb20ccf057da2edb6736c9973296b7129f4592ad9c1ad27287ef0653b121d74', 'experiments/root_native_trained_qual_review_step_v6_20261027/worker.py': '693d039f67e5a8bbeff32537287659fdde9c954812f52edbfdb7b43c485c0472', 'experiments/root_native_trained_qual_review_v5_20261027/__init__.py': '3d6294389be6d186c2b65ba7dc7dbf2b3c240c826e6f827a74e07978d5123869', 'experiments/root_native_trained_qual_review_v5_20261027/cpu_inputs.py': '7b0b83f4c8a92621ec5c6bd76ca5fbf8f4659a7932106f4cd2ace46c83221348'}
REGISTERED_SCHEMA='materialized-independent-review-registered-step-v6'
NODE_EXCLUSION='babel-p9-20'


def augmented_sources(original):
    result=dict(original)
    for name,digest in REGISTERED_SOURCES.items():
        require(name not in result or result[name]==digest,'Conflicting independently reviewed V6 verifier dependency')
        result[name]=digest
    return dict(sorted(result.items()))


def materialized_reviewer(cfg,load=checked):
    require('required_independent_reviewer_job_id'not in cfg
        and cfg['required_reviewer_parent_job_id']=='10942711','Registered reviewer is not a scalar batch job or parent afterok')
    for name,digest in REGISTERED_SOURCES.items():
        require(cfg['sources'].get(name)==digest and u.sha((ROOT/name).read_bytes())==digest,'Exact unchanged V6 verifier bytes must execute')
    contract=load(cfg['registered_independent_review_contract'])
    require(contract['schema']==REGISTERED_SCHEMA and contract['parent_afterok_dependency_required']is False,'Actual completed-step contract, never future source approval')
    value=load(contract['review']);qref=value['readout']
    require(contract['review']['path']==cfg['required_independent_qualification_review_path']
        and qref['path']==cfg['required_actual_qualification_result_path'],'Original exact independent review and qualified readout paths')
    proof=registered.materialized_review(cfg,value,qref,contract,load)
    return contract,value,qref,proof


def validate_node_policy(cfg,load=checked):
    value=cfg['operational_node_exclusion']
    require(value['excluded_nodes']==[NODE_EXCLUSION]
        and value['classification']=='TEMPORARY_OPERATIONAL_PRECAUTION_PENDING_DIAGNOSTIC_10948160'
        and value['confirmed_whole_NFS_fault']is False and value['applies_to']=='NEW_PENDING_EVALUATION_REQUESTS_ONLY'
        and len(value['actual_evidence'])==3 and len({r['path']for r in value['actual_evidence']})==3,'Narrow temporary evaluation-node precaution, not a confirmed NFS diagnosis')
    for ref in value['actual_evidence']:
        require(set(ref)=={'path','bytes','sha256'}and 0<ref['bytes']<=32<<20 and ref['path'].startswith(str(HOME)+'/'),'Bounded actual HOME operational evidence only')
        load(ref)
    return value


def prerequisite_roster(cfg, load=checked):
    value = load(cfg['qualification_prerequisite_roster'])
    require(value['schema']=='root-actual-v4-qualified-consumer-prerequisites-materialized-v6'
        and value['owner']=='Root'and value['no_future_result_SHA']is True,'Authentic completed registered reviewer, not pending ACK')
    q=value['qualifier'];reviewer=value['reviewer']
    require(q['owner']=='Root'and isinstance(q['job_id'],str)and re.fullmatch(r'[1-9]\d*',q['job_id'])is not None
        and q['job_id']not in{'10942687','10942858'}and cfg['required_loader_qualifier_job_id']==q['job_id']
        and cfg['qualified_loader_configuration']==value['qualified_loader_configuration']
        and cfg['required_actual_qualification_result_path']==q['output']+'/RESULT.json'
        and cfg['required_independent_qualification_review_path']==value['review_output_path'],
        'Exact original successful scalar qualifier and readout paths remain required')
    qualification=load(cfg['qualified_loader_configuration'])
    require(qualification['status']=='ROOT_CONFIGURED_NATIVE_TRAINED_EVAL_QUALIFICATION_V4'
        and qualification['generation_API_policy']==bridge.POLICY and qualification['sources']==value['qualified_sources']
        and qualification['output']==q['output'],'Unchanged exact V4 bridge/model qualification')
    contract,actual,qref,proof=materialized_reviewer(cfg,load)
    binding=load(contract['binding'])
    require(reviewer['owner']=='Root'and reviewer['kind']==REGISTERED_SCHEMA and 'job_id'not in reviewer
        and reviewer['contract']==cfg['registered_independent_review_contract']and reviewer['parent_job_id']=='10942711'
        and reviewer['registered_step']==binding['registered_step'],'Truthful materialized step reviewer identity, no scalar ACK')
    return value,qualification


def validate_configuration(cfg, load=checked):
    sid, cell = cfg["student_id"], cfg["training_cell"]
    require(cfg["status"] == STATUS and cfg["owner"] == "Root" and cfg["scope"] == SCOPE
            and sid in STUDENTS and cell["student_id"] == sid
            and cell["training_job_id"] == cfg["required_training_job_id"] == STUDENTS[sid]
            and cell["family"] == ("S1" if sid.startswith("s1-") else "Root")
            and cell["fraction"] == "0.10" and cell["selected_whole_conversations"] == 66076
            and cell["expected_update_horizon"] == 516,
            "Exactly twelve current10% final models, never partial paid5% or a future budget")
    roster = load(cfg["evaluation_roster"])
    require(len(roster["entries"]) == 12
            and {entry["student_id"]: entry["training_job_id"] for entry in roster["entries"]} == STUDENTS
            and [entry for entry in roster["entries"] if entry["student_id"] == sid] == [cell],
            "Exact source-reviewed twelve-cell training roster")
    require(cfg["training_configuration"] == cell["training_configuration"]
            and cfg["student_completion_path"] == cell["student_completion_path"]
            and cfg["source_root"] == str(SOURCE_ROOT) and cfg["resources"] == RESOURCES
            and cfg["banks"] == ["POPE", "GQA", "TextVQA", "ScienceQA_image"]
            and cfg["counts"] == banks.COUNTS and cfg["max_physical_attempts"] == 28506
            and cfg["output_limit_bytes"] == 512 << 20
            and literal(cfg["output"]) == OUTPUT / sid,
            "Unchanged complete banks, finite output and distinct namespace")
    training = load(cfg["training_configuration"])
    require(training["id"] == sid and training["method"] == cell["method"]
            and training["fraction"] == "0.10" and training["training_seed"] == cell["training_seed"],
            "Original exact trained method/fraction/seed configuration")
    if cell["family"] == "Root":
        require(old.JOBS[(cell["method"], cell["training_seed"])] == cell["training_job_id"]
                and cell["required_logical_sources"] == 264 and cell["required_physical_sources"] == 272,
                "Root264/physical272 cannot be relabelled S1")
    else:
        legacy = load(cell["prior_formal_configuration"])
        require(legacy["student_id"] == sid and legacy["training_configuration"] == cfg["training_configuration"]
                and legacy["student_completion_path"] == cfg["student_completion_path"],
                "Exact existing S1 model/consumer lineage")
    template = load(OFFICIAL_BANK_TEMPLATE)
    require(cfg["official_bank_contract_configuration"] == OFFICIAL_BANK_TEMPLATE
            and all(cfg[field] == template[field] for field in BANK_FIELDS),
            "Only qualification/runtime integration changes; official bank/decoder/scorer identities do not")
    source7_asset_repair(cfg, load)
    source7_empty_prior(cfg, load)
    official_join_preflight(cfg, load)
    validate_node_policy(cfg, load)
    _, qualification = prerequisite_roster(cfg, load)
    require(all(cfg["sources"].get(path) == digest for path, digest in qualification["sources"].items()),
            "Consumer must execute the exact admitted V4 qualifier/model/bridge dependency bytes")
    return training


# A completed no-model input qualification is not neural or benchmark evidence.
INPUT_REVIEW_STATUS='PASS_INDEPENDENT_ACTUAL_OFFICIAL_TEXTVQA_FOUR_BANK_INPUT_PREFLIGHT_V1'
INPUT_SUBJECT={'bytes': 75345, 'path': '/home/luyangz/data_selection_multimodal/collaboration/root-current-four-bank-eval-v5-20261027/configs/CONFIG_s1-student-v1-10-0R-s29.json', 'sha256': '96a75a7d03ecd4e571336bad8a9cc945d53b96c5d80b5311f662fd43bbe07971'}
INPUT_SOURCE_PACKET=dict(path=str(HOME/'collaboration/root-current-four-bank-eval-v5-20261027/SOURCE_PACKET_V1.json'),bytes=68928,sha256='b4e70a0edf749f11ebb9319eb9bc346fa184ef19bde10e6df12f6919b8e60187')

def official_join_preflight(cfg,load=checked):
    from experiments.root_official_textvqa_join_v1_20261027 import join
    ref=cfg['official_join_input_qualification'];q=load(ref);peer=load(cfg['official_join_input_qualification_review'])
    require(peer['status']==INPUT_REVIEW_STATUS and peer['independent']is True
        and peer['reviewer']not in(None,'','/root/repair_independent_review')and peer['readout']==ref
        and peer['subject_configuration']==INPUT_SUBJECT and peer['subject_source_packet']==INPUT_SOURCE_PACKET,
        'Genuine independent actual full-bank input-readout review required, never a source-only review')
    require(q['status']=='COMPLETE_ACTUAL_FOUR_BANK_INPUT_PREFLIGHT_NOT_MODEL_OR_EVALUATION'
        and q['configuration']==INPUT_SUBJECT and q['subject_source_packet']==INPUT_SOURCE_PACKET
        and q['source5_root']==str(HOME/'root_current_four_bank_eval_src_v5_20261027')
        and q['actual_QA_count']==28505 and q['actual_unique_RGB_images_each_bank']==6081
        and q['neural_forwards']==0 and q['model_constructed']is False and q['score_computed']is False
        and q['evaluator_or_student_or_SOTA_PASS']is False,
        'Actual full28505 QA/6081 per-bank unique RGB qualification only')
    subject=load(INPUT_SUBJECT)
    require(all(cfg[field]==subject[field]for field in BANK_FIELDS),'Qualified bank inputs/scoring/decoding cannot be changed')
    require(set(q['reports'])==set(banks.COUNTS),'All four full-bank reports required')
    for bank,n in banks.COUNTS.items():
        report=q['reports'][bank]
        require(report['questions']==n and report['images']==banks.IMAGE_COUNTS[bank]
            and report['actual_unique_encoded_RGB_images']==banks.IMAGE_COUNTS[bank]
            and report['all_declared_QA_GT_media_membership']is True
            and report['actual_all_unique_native_images_encoded_RGB_revalidated']is True
            and report['neural_forwards']==0 and report['model_constructed']is False and report['score_computed']is False,
            'Full original bank QA/GT/actual encoded-RGB scope required: '+bank)
    receipt=load(q['textvqa_join']);mapping=load(receipt['mapping'])
    require(receipt['schema']=='official-author-textvqa-image-question-bijection-v1'
        and receipt['questions']==receipt['official_questions']==receipt['unique_semantic_QAs']==receipt['unique_author_join_keys']==5000
        and receipt['unique_native_image_ids']==3166 and receipt['repeated_native_image_ids']==1834
        and receipt['all_native_and_official_rows_joined_once']is True
        and receipt['native_IDs_prompts_semantic_keys_and_answers_unchanged']is True
        and receipt['answer_targets_are_scorer_side_only']is True and receipt['model_or_inference_or_score_performed']is False
        and receipt['prompt_processor_file_sha256']==join.SCORER_SHA
        and len(mapping)==len({row['semantic_key']for row in mapping})==5000
        and {row['original_row_index']for row in mapping}==set(range(5000))
        and join.digest(mapping)==receipt['mapping_sha256'],
        'Actual bijective5000 TextVQA mapping and original author parser required')
    return q,receipt


SOURCE7_FIRST='production-student-v1-10-source_sqrt-s43'
SOURCE7_READY=('production-student-v1-10-source_sqrt-s43','production-student-v1-10-uniform-s17','s1-student-v1-10-0R-s17','s1-student-v1-10-0R-s29','s1-student-v1-10-0R-s43','s1-student-v1-10-H0-s29','s1-student-v1-10-H0-s43')
ASSET_REVIEW_STATUS='PASS_INDEPENDENT_ACTUAL_BOUND19_CACHE_AND_INPUT_READINESS_V3'
PREFIX_REVIEW_STATUS='PASS_INDEPENDENT_ACTUAL_SOURCE7_COLD_MODEL_QA_PREFIX_V1'

def source7_asset_repair(cfg,load=checked):
    ref=cfg['actual_bound19_asset_readiness'];q=load(ref);peer=load(cfg['actual_bound19_asset_review'])
    require(peer['status']==ASSET_REVIEW_STATUS and peer['readout']==ref and peer['independent']is True
        and peer['reviewer']not in(None,'','/root/repair_independent_review')
        and peer['subject_configuration']==q['configuration'],
        'Actual independent19asset/full-input readout review, never source-only approval')
    b=load(q['binding']);repair=q['cache_repair']
    require(q['status']=='ACTUAL_REPAIRED_ORIGINAL_JSON_AND_FULL_INPUT_QUALIFICATION_V2_NOT_ACTIVATION'
        and b['status']=='ROOT_BOUND_ALL19_CACHE_REPAIR_V3'and b['owner']=='Root'
        and q['actual']['physical_route']=='NEW_STANDALONE_ALL19_BOUND_CACHE_REPAIR_AND_FULL_INPUT_QUALIFICATION_V3'
        and q['actual']['job_id']==b['job_id']and len(q['sources'])==111
        and q['new_NN']==0 and q['new_image_measurements']==0 and q['target_activation_PASS']is False
        and repair['status']=='ACTUAL_ALL19_BOUND_SHA_AND_CANONICAL_ALIAS_READBACK_COMPLETE_NO_MODEL_FORWARD'
        and repair['qualified_file_count']==repair['expected_file_count']==len(repair['rows'])==19
        and repair['downloaded_bytes']==repair['payload_bytes_written']==repair['payload_copies']==0
        and repair['model_loaded']is False and repair['new_neural_forwards']==0
        and repair['mount_before']==repair['mount_after']and q['full_consumer_mount_before']==q['full_consumer_mount_after'],
        'Actual standaloneV3 full19SHA/alias/input proof with truthful source/actor and zero new payloads')
    require(cfg['source7_prefix_mode']==('PRODUCER'if cfg['student_id']==SOURCE7_FIRST else'CONSUMER'if cfg['student_id']in SOURCE7_READY else'DEFERRED_TRAINING_UNREADY')
        and cfg['source7_prefix_provider_path']==str(CONTROL/'PREFIX_PROVIDER.json'),'Exact source7first/remaining/deferred physical policy')
    return q

def source7_prefix_gate(cfg,load=checked):
    if cfg['source7_prefix_mode']=='PRODUCER':return None
    require(cfg['source7_prefix_mode']=='CONSUMER','Five failed/unready training targets are not authorized by empty-eval recovery')
    provider=load(desc(literal(cfg['source7_prefix_provider_path'])))
    require(provider['status']=='ROOT_BOUND_ACTUAL_SOURCE7_COLD_PREFIX_PROVIDER_V1'and provider['owner']=='Root','Actual first Source7 binding/provider required')
    b=load(provider['binding']);first=load(b['configuration'])
    require(first['student_id']==SOURCE7_FIRST and first['source7_prefix_mode']=='PRODUCER'and first['sources']==cfg['sources']
        and first['actual_bound19_asset_readiness']==cfg['actual_bound19_asset_readiness']and b['status']==BINDING_STATUS
        and b['owner']=='Root'and b['configuration']==provider['configuration']and b['nonce']==first['nonce'],
        'Real first current Source7 physical producer, never a future ACK or old failed job')
    path=Path(first['output'])/'r0/COLD_MODEL_QA_PREFIX.json'
    require(provider['prefix_path']==str(path)and provider['independent_review_path']==str(CONTROL/'PREFIX_REVIEW.json'),'Literal late provider paths, no future SHA')
    ref=desc(literal(path));value=load(ref);review=load(desc(literal(provider['independent_review_path'])))
    require(review['status']==PREFIX_REVIEW_STATUS and review['readout']==ref and review['binding']==provider['binding']
        and review['configuration']==b['configuration']and review['independent']is True and review['reviewer']not in(None,'','/root/repair_independent_review'),
        'Independent actual cold-model/nativeQA prefix review required before remaining consumers')
    require(value['status']=='ACTUAL_SOURCE7_COLD_MODEL_TWO_NATIVE_QA_PREFIX_NOT_FULL_BANK'
        and value['configuration']==b['configuration']and value['actual']['job_id']==b['job_id']and value['actual']['nonce']==b['nonce']
        and value['actual']['restart']==0 and value['accepted_native_QAs']==2 and value['bank']=='POPE'
        and value['scientific_FULL_bank_or_SOTA_PASS']is False and value['no_extra_or_replayed_QAs']is True
        and value['model_identity_unchanged_after_prefix']is True and value['actual_bound19_asset_readiness']==cfg['actual_bound19_asset_readiness'],
        'Actual source7same-process cold model + two durably audited nativeQAs, not journal-only inference')
    bridgeproof=value['generation_bridge']
    require(bridgeproof['status']=='ACTUAL_NATIVE_BRIDGE_PREFIX_INSIDE_UNCHANGED_OFFICIAL_AUDIT_CONTEXT'
        and bridgeproof['calls']==len(bridgeproof['records'])==2
        and bridgeproof['policy']==bridge.POLICY and bridgeproof['adapter_active']is True and bridgeproof['adapter_merged']is False
        and bridgeproof['unchanged_official_Recorder_wrapper_authenticated']is True
        and bridgeproof['generate_or_preparation_hook_mutated_by_observer']is False
        and all(all(r[k]is True for k in('success','input_object_unchanged','all_keyword_objects_unchanged','prepare_hook_restored','original_PEFT_generate_invoked','output_object_unchanged'))for r in bridgeproof['records']),
        'Two actual unchanged native/PEFT calls inside pinned official audit, no extra inference')
    proof=load(value['model_reconstruction']);values=proof['portable_value_verification'];injection=proof['native_adapter_construction']
    require(values['exact_parameter_inventory']is True and values['frozen_parameters']==682 and values['trained_parameters']==452
        and values['LoRA_tensors']==448 and values['projector_tensors']==4 and values['portable_trainable_values_exact']is True
        and len(injection['language_adapter_targets'])==224 and len(injection['loaded_vision_generic_targets_excluded'])==72
        and proof['persistent_buffer_verification']['exact_persistent_buffer_support']is True
        and proof['persistent_buffer_verification']['persistent_buffer_values_exact']is True
        and load(value['model_identity'])==load(value['post_prefix_model_identity']),
        'Actual full1134/452/224/72/buffer reconstruction and matching pre/post model state')
    return ref


EMPTY_REVIEW_STATUS='PASS_INDEPENDENT_ACTUAL_SOURCE6_SEVEN_EMPTY_QA_OUTPUTS_V1'
SOURCE6_JOBS={'production-student-v1-10-source_sqrt-s43':'10953668','production-student-v1-10-uniform-s17':'10953669','s1-student-v1-10-0R-s17':'10953670','s1-student-v1-10-0R-s29':'10953672','s1-student-v1-10-0R-s43':'10953673','s1-student-v1-10-H0-s29':'10953678','s1-student-v1-10-H0-s43':'10953679'}
def source7_empty_prior(cfg,load=checked):
    ref=cfg['actual_source6_empty_QA_readout'];value=load(ref);peer=load(cfg['actual_source6_empty_QA_review'])
    require(peer['status']==EMPTY_REVIEW_STATUS and peer['readout']==ref and peer['independent']is True
        and peer['reviewer']not in(None,'','/root/repair_independent_review'),
        'Actual independent old-seven empty-QA review required; trace/source alone is insufficient')
    require(value['status']=='ACTUAL_TERMINAL_SOURCE6_FAILED_EMPTY_QA_FUNCTIONS_NOT_RECOVERY_AUTHORIZATION'
        and value['functions']==7 and value['accepted_QAs']==0 and value['recovery_authorized']is False
        and value['old_sources_or_outputs_modified']is False and len(value['rows'])==7
        and {r['student_id']for r in value['rows']}==set(SOURCE6_JOBS),
        'Exact seven authentic terminal failed empty-QA functions, not old-source success')
    for row in value['rows']:
        b=load(row['old_binding']);oldcfg=load(row['old_configuration'])
        require(b['job_id']==SOURCE6_JOBS[row['student_id']]and b['owner']=='Root'
            and b['status']=='ROOT_BOUND_CURRENT12_OFFICIAL_JOIN_FOUR_BANK_EVALUATION_V6'
            and b['configuration']==row['old_configuration']and oldcfg['student_id']==row['student_id']
            and oldcfg['output']==row['output']and row['accepted_QAs']==0
            and row['bank_directories_present']==[]and row['attempts']==['r0']
            and row['old_job_state']=='FAILED'and row['recovery_authorized']is False,
            'Actual Source6 binding/output/zero-QA custody differs')
    return value
