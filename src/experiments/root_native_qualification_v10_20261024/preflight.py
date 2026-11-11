"""Reuse Root's actual saved-token differential; never acquire another scan."""
import importlib
import importlib.metadata
import os
from pathlib import Path
import socket
import sys
from . import common as c
from .masking import PREFIX_SOURCE, REPAIRED_ID

V5_PACKET_SHA='c91872f28987de0f3fd98b472e51b9579fe94227981439244a9a102c80f80fe0'
V5_PROOF_SHA='7af946e3b2c05293f383e24564d2a41b8ccca754d3af79d0a37db4f73c06230e'

def source_equivalence_bridge(proof,target_sources,*,raw=None,assets=None,environment=None):
    """Reuse measured V5 masks without relabelling any measured source map.

    V7 changes optimizer lifecycle/inventory, not the input operator. All native
    V7 forward/LR/save/restart evidence must still be acquired in V10.
    """
    from experiments.root_native_campaign_v5_20261024 import common as old
    from experiments.root_native_campaign_v7_20261024 import common as new
    from experiments.root_native_campaign_v5_20261024.config import RECIPE as prior_recipe
    from experiments.root_native_campaign_v7_20261024.config import RECIPE as target_recipe
    c.require(proof['source_packet_sha256']==V5_PACKET_SHA and len(proof['repair_sources'])==90,
              'Only actual measured V5/90 prefix evidence may cross this explicit bridge')
    old.verify_sources(proof['repair_sources']);new.verify_sources(target_sources)
    c.require(len(target_sources)==91 and prior_recipe==target_recipe,'Unexpected V7/91 target recipe')
    identities={}
    for name in ('structured_preprocess.py','config.py'):
        a=old.HERE/name;b=new.HERE/name
        c.require(old.sha(a)==new.sha(b),'V5/V7 input operator differs: '+name)
        identities[name]=dict(measured=old.desc(a),target=new.desc(b))
    old_shared={k:v for k,v in proof['repair_sources'].items() if not k.startswith('experiments/root_native_campaign_v5_20261024/')}
    new_shared={k:v for k,v in target_sources.items() if not k.startswith('experiments/root_native_campaign_v7_20261024/')}
    c.require(old_shared==new_shared,'Frozen author/shared/oracle dependencies changed across input bridge')
    if raw is not None:
        c.require(raw['repair_sources']==proof['repair_sources'] and raw['transformers_version']=='4.37.2'
            and raw['tokenizers_version']=='0.15.1','Measured tokenizer source/API changed')
        for rel,sha in raw['oracle_sources'].items():
            c.require(c.sha(c.ROOT/rel)==sha,'Measured independent oracle source changed')
        c.require(assets is not None and environment is not None,'Current target tokenizer providers required')
        wanted={__import__('pathlib').Path(d['path']).name:(d['bytes'],d['sha256']) for d in raw['tokenizer_files']}
        current={__import__('pathlib').Path(d['path']).name:(d['bytes'],d['sha256']) for d in assets['assets']['vicuna']['files']}
        c.require('tokenizer.model' in wanted and all(current.get(k)==v for k,v in wanted.items()),
                  'Current tokenizer/config bytes differ from measured V5 operator')
        # Historical receipts are immutable and may omit tokenizers. This is
        # logical provider/proof compatibility, not a current-runtime claim.
        versions=environment['versions']
        c.require(versions['transformers']=='4.37.2',
                  'Declared transformer provider differs from measured operator')
        if 'tokenizers' in versions:
            c.require(versions['tokenizers']=='0.15.1',
                      'Recorded historical tokenizer provider differs from measured operator')
    return dict(status='EXPLICIT_BYTE_IDENTICAL_V5_TO_V7_INPUT_OPERATOR_BRIDGE_ONLY',
        measured_native_package='experiments.root_native_campaign_v5_20261024',measured_packet_sha256=V5_PACKET_SHA,
        measured_sources=proof['repair_sources'],target_native_package='experiments.root_native_campaign_v7_20261024',
        target_sources=target_sources,byte_identical_operators=identities,recipe_identical=True,
        measured_source_map_unchanged=True,V7_GPU_forward_LR_save_restart_qualified=False)


def actual_tokenizer_runtime(raw, environment):
    """Mandatory live NativePython check, separate from historical metadata.

    Called only at actual supervisor/child execution boundaries. Logical
    metadata preparation never imports these packages or fabricates ENV keys.
    """
    c.require(Path(sys.executable).absolute()==Path(environment['python_executable']).absolute(),
              'Current tokenizer attestation requires the exact NativePython executable')
    expected={'transformers':'4.37.2','tokenizers':'0.15.1'}
    c.require(raw['transformers_version']==expected['transformers'] and
              raw['tokenizers_version']==expected['tokenizers'], 'Measured tokenizer API differs')
    versions={};origins={}
    for name,wanted in expected.items():
        module=importlib.import_module(name)
        observed=getattr(module,'__version__',None)
        distribution=importlib.metadata.version(name)
        c.require(observed==distribution==wanted,'Actual current tokenizer package differs: '+name)
        c.require(getattr(module,'__file__',None), 'Actual package module origin missing: '+name)
        origins[name]=c.desc(Path(module.__file__).absolute())
        versions[name]=observed
    return dict(status='PASS_ACTUAL_CURRENT_TOKENIZER_API_ONLY',
        python_executable=str(Path(sys.executable).absolute()),actual_module_versions=versions,
        actual_module_origins=origins, historical_environment_recorded_versions=environment['versions'],
        historical_ENV_mutated=False,model_weights_read=False,neural_invoked=False,
        native_GPU_LR_save_restart_qualified=False)


def runtime_preflight(study, *, site):
    """Cheap actual gate before any native full-asset validation/model load."""
    c.require(site in ('babel','aces'),'Explicit site required for actual preflight')
    c.require(os.environ.get('SLURM_JOB_ID','').isdigit(),
              'Actual tokenizer runtime must execute inside a registered owned GPU allocation')
    if site=='babel':
        c.require(study['driver_sources']==c.sources(),'Actual Babel V10 driver source differs')
        proof_desc=study['masking_contract']['actual_prefix_proof']
    else:
        from experiments.root_h100_native_qualification_v10_20261024 import protocol
        c.require(study['driver_sources']==protocol.sources(),'Actual H100 V10 driver source differs')
        proof_desc=study['actual_prefix_proof']
    c.require(study['native_target']['package']=='experiments.root_native_campaign_v7_20261024' and
              study['native_target']['source_packet']['sha256']==
              'a704d9716719a9ad2a0cd0890c24fec35c15bd085b7c2c98fe38ea0c98961830',
              'Actual preflight cannot change the NativeV7/91 target')
    c.require(proof_desc['sha256']==V5_PROOF_SHA,'Actual measured V5 proof descriptor differs')
    proof=c.checked(proof_desc)
    raw_desc=proof['raw_remote_only'] if site=='babel' else study['raw_prefix_proof_copy']
    c.require(all(raw_desc[k]==proof['raw_remote_only'][k] for k in ('bytes','sha256')),
              'Current site raw proof copy differs from actual measured V5 evidence')
    raw=c.checked(raw_desc)
    pair={arm:c.checked(study['native_configurations'][arm]) for arm in ('reference','resume')}
    for name in ('assets','environment'):
        c.require(pair['reference']['providers'][name]==pair['resume']['providers'][name],
                  'Actual runtime preflight has unpaired provider: '+name)
    config=pair['reference'];env_desc=config['providers']['environment'];environment=c.checked(env_desc)
    assets=c.checked(config['providers']['assets'])
    bridge=source_equivalence_bridge(proof,config['sources'],raw=raw,assets=assets,environment=environment)
    result=actual_tokenizer_runtime(raw,environment)
    result.update(site=site,actual_job_id=os.environ['SLURM_JOB_ID'],
        actual_restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0')),actual_node=socket.gethostname(),
        proof=proof_desc,raw_proof=raw_desc,environment=env_desc,
        measured_source_map_unchanged=bridge['measured_source_map_unchanged'],
        full_model_asset_hashes_performed=False,UTC=c.now())
    return result


def compare_records(original, observed):
    old = {row["record_id"]: row for row in original}
    new = {row["record_id"]: row for row in observed}
    c.require(len(old) == len(new) == len(original) == len(observed) == 1024
              and set(old) == set(new), "Exact full1024 per-ID evidence required")
    unchanged, changed, failures = [], [], []
    for record_id, before in old.items():
        after = new[record_id]
        if after["input_ids"] != before["input_ids"]:
            failures.append((record_id, "input_ids_changed"))
        if after["labels"] != after["independent_labels"]:
            failures.append((record_id, "independent_oracle_disagrees"))
        if not after["unexpanded_targets"] or not after["expanded_targets"]:
            failures.append((record_id, "zero_targets"))
        (unchanged if after["labels"] == before["unexpanded_labels"] else changed).append(record_id)
    return dict(passed=not failures and len(unchanged) == 1023 and changed == [REPAIRED_ID],
                unchanged_records=len(unchanged), changed_record_ids=changed,
                unchanged_record_ids=unchanged, failures=failures,
                exact_original_input_ids_preserved=not any(
                    reason == "input_ids_changed" for _, reason in failures))


def verify_shared(study, config, *, raw):
    proof = c.checked(study["masking_contract"]["actual_prefix_proof"])
    c.require(study['masking_contract']['actual_prefix_proof']['sha256']==V5_PROOF_SHA,'Exact actual measured V5 prefix summary required')
    source_equivalence_bridge(proof,config['sources'])
    c.require(proof["status"] == "PASS_ACTUAL_TOKENIZER_PARITY_ONLY"
              and proof["records_checked"] == 1024 and proof["violations"] == []
              and proof["changed_labels"] == [REPAIRED_ID]
              and proof["all_input_ids_identical"] is True
              and proof["all_unaffected_labels_identical"] is True
              and proof["all_independent_prefix_oracle_identical"] is True,
              "Exact actual Root full1024 prefix evidence required before any neural work")
    if raw:
        evidence = c.checked(proof["raw_remote_only"])
        c.require(evidence["status"] == proof["status"]
                  and evidence["repair_sources"] == proof["repair_sources"]
                  and evidence["oracle_sources"][PREFIX_SOURCE] == c.sha(c.ROOT / PREFIX_SOURCE)
                  and evidence["records_checked"] == 1024 and evidence["violations"] == []
                  and evidence["native_bad_record_ids"] == [REPAIRED_ID]
                  and evidence["changed_label_record_ids"] == [REPAIRED_ID]
                  and evidence["repaired_bad_record_ids"] == [],
                  "Actual saved-token/raw-source evidence differs")
        rows = evidence["records"]
        c.require(len(rows) == len({r["record_id"] for r in rows}) == 1024,
                  "Raw prefix proof coverage is not exact1024")
        for row in rows:
            c.require(row["input_ids_identical"] and row["ordinary_labels_identical"]
                      and row["independent_prefix_oracle_identical"]
                      and row["repaired_unexpanded_targets"] > 0
                      and row["repaired_expanded_targets"] > 0,
                      "Raw per-ID proof failed: " + row["record_id"])
            if row["record_id"] != REPAIRED_ID:
                c.require(row["original_labels"] == row["repaired_labels"],
                          "Ordinary saved labels changed")
        c.require(c.desc(proof["raw_remote_only"]["path"]) == proof["raw_remote_only"],
                  "Raw prefix proof changed during validation")
        source_equivalence_bridge(proof,config['sources'],raw=evidence,
            assets=c.checked(config['providers']['assets']),environment=c.checked(config['providers']['environment']))
    return proof
