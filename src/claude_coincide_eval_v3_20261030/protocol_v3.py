"""Finite COINCIDE v3 recovered-parent four-bank configuration; banks/decoder stay frozen.

Deep-copies the qualified Root v7 basis exactly like v2 protocol.configuration,
but binds the v3 recovery training records (REQUEST/INTENT/ACK chain) instead of
an original v2 driver binding. Importable without torch (login node, py3.9).
"""
from copy import deepcopy
from pathlib import Path
import re

from experiments.copilot7_coincide_native_exec_v2_20261028 import protocol as native
from experiments.root_current_four_bank_eval_v1_20261027 import contract as basis_contract
from . import common_v3 as c

STATUS = 'CLAUDE_CONFIGURED_COINCIDE_V3_RECOVERED_PARENT_FOUR_BANKS'
OWNER = 'Claude-baseline-recovery'
SCOPE = 'FINAL_COINCIDE_FIXED10_V3_RECOVERED_THREE_SEEDS_FOUR_COMPLETE_OFFICIAL_BANKS'
RESOURCES = dict(cpus=8, memory_mb=65536, gpus=1, minutes=c.MINUTES)
COUNTS = dict(POPE=8910, GQA=12578, TextVQA=5000, ScienceQA_image=2017)
BANKS = ['POPE', 'GQA', 'TextVQA', 'ScienceQA_image']
BANK_FIELDS = basis_contract.BANK_FIELDS
GQA_CONVERSION = "prediction.rstrip('.').lower(); original target exact equality"
CONTINUATION = 'SEPARATE_REVIEWED_V3_SUCCESSOR_JOB_PER_QA_JOURNAL_NO_REQUEUE'
# Basis fields that name another student's identity/lineage; never inherited.
BASIS_ONLY = ('evaluation_roster', 'training_cell', 'student_completion_path', 'required_training_job_id',
              'prerequisites', 'source7_prefix_mode', 'source7_prefix_provider_path', 'checkpoint_retention')


def sources(cfg, load=c.checked, root=None):
    """Frozen archive (429) + qualified basis overlay (v7) + this package, all verified on disk."""
    root = Path(root) if root is not None else c.ROOT
    handoff = load(cfg['frozen_handoff'])
    c.require(c.same_bytes(handoff['source_archive'], c.ARCHIVE), 'Frozen handoff names a different archive')
    tree, superseded = c.frozen_tree(handoff['sources'], cfg['qualified_basis_sources'])
    c.require(superseded == cfg['superseded_archive_files'], 'Declared basis-superseded archive files differ')
    for name, digest in native.sources().items():
        c.require(tree.get(name) == digest, 'Unchanged native COINCIDE dependency missing/changed: ' + name)
    own = c.package_sources(root)
    c.require(not set(own) & set(tree), 'Package collides with frozen tree')
    tree.update(own)
    for name, digest in tree.items():
        c.require(c.sha(root / name) == digest, 'Staged evaluation source drift: ' + name)
    return dict(sorted(tree.items()))


def expected_prefix(seed):
    pre = c.PREFIX[seed]
    if pre['mode'] == 'fresh':
        return dict(mode='fresh', step=0, v2_child=None, v2_job=pre['v2_job'])
    child = c.V2_REPORT / c.cell_id(seed) / 'attempts' / pre['v2_attempt'] / 'child'
    return dict(mode='resume', step=pre['step'], v2_child=str(child), v2_job=pre['v2_job'])


def training_chain(seed, training_ref, records, load=c.checked):
    """Ordered v3 training REQUEST/INTENT/ACK chain for this cell; returns job ids."""
    cell = c.cell_id(seed)
    c.require(type(records) is list and records, 'At least one bound v3 training request')
    jobs, previous = [], None
    for k, rec in enumerate(records):
        c.require(type(rec) is dict and set(rec) == {'request', 'intent', 'ack', 'job_id'}, 'Training record shape')
        folder = Path(rec['request']['path']).parent
        c.require(folder.parent == c.TRAINING_RECORDS / cell and Path(rec['intent']['path']) == folder / 'INTENT.json'
                  and Path(rec['ack']['path']) == folder / 'ACK.json', 'Training records outside the v3 cell namespace')
        request, intent, ack = load(rec['request']), load(rec['intent']), load(rec['ack'])
        c.require(request['schema'] == c.TRAINING_REQUEST_SCHEMA and request['seed'] == seed
                  and request['cell'] == cell and request['nonce'] == folder.name
                  and request['configuration'] == training_ref and intent['request'] == rec['request']
                  and ack['intent'] == rec['intent'] and ack['job_id'] == rec['job_id']
                  and type(rec['job_id']) is str and rec['job_id'].isdecimal() and rec['job_id'] not in jobs,
                  'Actual v3 training request/intent/ACK identity')
        pred = request['predecessor']
        if k == 0:
            want = expected_prefix(seed)
            if want['mode'] == 'fresh':
                c.require(pred['kind'] == 'fresh_v2_never_started' and pred['mode'] == 'fresh' and pred['step'] == 0
                          and pred['accepted'] is None and pred['job'] == want['v2_job'], 'Fresh s43 predecessor')
            else:
                c.require(pred['kind'] == 'v2_site_interrupted' and pred['mode'] == 'resume'
                          and pred['job'] == want['v2_job'] and pred['step'] == want['step']
                          and pred['attempt'] == want['v2_child'] and type(pred['accepted']) is dict,
                          'Exact site-interrupted v2 accepted predecessor')
        else:
            c.require(pred['kind'] == 'v3_continuation' and pred['mode'] == 'resume' and pred['job'] == previous
                      and pred['quarantine'] == [], 'v3 successor must continue the previous bound v3 job')
        jobs.append(rec['job_id']); previous = rec['job_id']
    return jobs


def validate_science(cfg, basis, training, load=c.checked):
    seed = cfg['training_seed']
    cell = c.cell_id(seed)
    c.require(cfg['status'] == STATUS and cfg['owner'] == OWNER and cfg['scope'] == SCOPE
              and cfg['student_id'] == cell == training['id'] and cfg['method'] == 'coincide'
              and cfg['fraction'] == '0.10' and training['training_seed'] == seed
              and cfg['source_root'] == str(c.SOURCE_ROOT) and cfg['python'] == c.PYTHON
              and cfg['resources'] == RESOURCES and cfg['banks'] == BANKS and cfg['counts'] == COUNTS
              and cfg['max_physical_attempts'] == sum(COUNTS.values()) + 1
              and cfg['output_limit_bytes'] == 512 << 20 and re.fullmatch('[a-f0-9]{32}', cfg['nonce'])
              and cfg['output'] == str(c.OUTPUT / cell / cfg['nonce']),
              'Only finite COINCIDE fixed10 seeds and four complete native banks')
    native.supported(training)
    c.require(Path(cfg['training_configuration']['path']) == c.CELLS / cell / 'CONFIG_FINAL.json',
              'Exact unchanged v2 COINCIDE training configuration')
    c.require(cfg['qualified_basis'] == c.BASIS and cfg['frozen_handoff'] == c.HANDOFF
              and cfg['frozen_archive'] == c.ARCHIVE and cfg['qualified_basis_sources'] == basis['sources']
              and all(cfg[field] == basis[field] for field in BANK_FIELDS)
              and cfg['sources'] == sources(cfg, load),
              'Same qualified v7 bank/media/processor/decoder inputs and exact staged sources')
    c.require(not any(name in cfg for name in BASIS_ONLY if name != 'checkpoint_retention')
              and cfg['checkpoint_retention'] == dict(training_configuration=cfg['training_configuration'],
                  natural_final_step=516, no_weight_copy=True, checkpoint_hash_late_bound=True,
                  retain_until='COINCIDE_V3_FOUR_BANK_RESULTS_INDEPENDENTLY_ACCEPTED'),
              'No inherited foreign student lineage; COINCIDE final checkpoint retention')
    jobs = training_chain(seed, cfg['training_configuration'], cfg['training_records'], load)
    c.require(cfg['required_training_job_ids'] == jobs and cfg['dependency_training_job_id'] == jobs[-1]
              and cfg['prefix'] == expected_prefix(seed), 'Exact bound v3 training chain and accepted prefix')
    c.require(cfg['adaptation_label'] == native.ADAPTATION and cfg['GQA_conversion'] == GQA_CONVERSION
              and cfg['GQA_basis_unconverted_score_also_reported'] is True
              and cfg['evaluation_continuation'] == CONTINUATION and cfg['automatic_requeue'] is False,
              'Declared native GQA conversion only; no self-requeue')
    c.require(cfg['no_model_downloads'] is True and cfg['no_base_copies'] is True
              and cfg['no_START_journal_reuse'] is True and cfg['no_evaluation_targets_for_selection'] is True,
              'No reacquisition, foreign journals, target tuning or START/retention relabel')
    return training


def configuration(seed, training_ref, training_records, nonce, basis_ref=None, load=c.checked):
    basis_ref = c.BASIS if basis_ref is None else basis_ref
    basis, training = load(basis_ref), load(training_ref)
    native.supported(training)
    cell = c.cell_id(seed)
    c.require(training['id'] == cell and training['training_seed'] == seed, 'Training config belongs to this seed')
    handoff = load(c.HANDOFF)
    _, superseded = c.frozen_tree(handoff['sources'], basis['sources'])
    jobs = training_chain(seed, training_ref, training_records, load)
    cfg = deepcopy(basis)
    for name in BASIS_ONLY:
        cfg.pop(name, None)
    cfg.update(status=STATUS, owner=OWNER, scope=SCOPE, student_id=cell, method='coincide', fraction='0.10',
        training_seed=seed, nonce=nonce, source_root=str(c.SOURCE_ROOT), python=c.PYTHON, resources=RESOURCES,
        banks=list(BANKS), counts=dict(COUNTS), max_physical_attempts=sum(COUNTS.values()) + 1,
        output_limit_bytes=512 << 20, output=str(c.OUTPUT / cell / nonce),
        training_configuration=training_ref, training_records=training_records,
        required_training_job_ids=jobs, dependency_training_job_id=jobs[-1], prefix=expected_prefix(seed),
        checkpoint_retention=dict(training_configuration=training_ref, natural_final_step=516, no_weight_copy=True,
            checkpoint_hash_late_bound=True, retain_until='COINCIDE_V3_FOUR_BANK_RESULTS_INDEPENDENTLY_ACCEPTED'),
        qualified_basis=basis_ref, qualified_basis_sources=basis['sources'], frozen_handoff=c.HANDOFF,
        frozen_archive=c.ARCHIVE, superseded_archive_files=superseded,
        adaptation_label=native.ADAPTATION, GQA_conversion=GQA_CONVERSION,
        GQA_basis_unconverted_score_also_reported=True, evaluation_continuation=CONTINUATION,
        automatic_requeue=False, no_evaluation_targets_for_selection=True)
    cfg['sources'] = sources(cfg, load)
    validate_science(cfg, basis, training, load)
    return cfg
