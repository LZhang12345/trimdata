"""Immutable S1 treated-arm held design; activation only through a separately reviewed target.

Reuses Root's reviewed held-cohort helpers by import (sites, profiles, scheduler ceiling, target
closure). Each S1 request is paired with Root's same-site/same-budget/same-seed uniform and source_sqrt
held requests and inherits their exact site profile, scheduler and timing evidence.
"""
import re
from pathlib import Path
from experiments.root_fulltrajectory_held_cohort_v1_20261025 import protocol as rp
c = rp.c
ncfg = rp.ncfg
ROOT = rp.ROOT
HERE = Path(__file__).resolve().parent
OWNER = 'Session1'
AUTHORS = ('Session1', 'S1', 'session1')
SCHEMA = 's1-fulltrajectory-held-request-v1'
STATUS = 'CONDITIONAL_HELD_ONLY_NOT_EXECUTION_AUTHORITY'
REVIEW = 'PASS_INDEPENDENT_S1_CONDITIONAL_FULLTRAJECTORY_HELD_DESIGN_V1'
ACTIVATION_REVIEW = 'PASS_INDEPENDENT_ACTUAL_S1_FULLTRAJECTORY_JOB_ACTIVATION_V1'
COMMENT = 'dsmm-s1-fulltrajectory:'
ARMS = ('0R', 'H0', 'HR', 'L0', 'LR')
METHOD_PREFIX = 's1_r1_select_v4_'
SEEDS = rp.SEEDS
BUDGETS = rp.BUDGETS
# Finalized S1 selection receipts (HOME, produced inside owned General allocations).
RECEIPTS = {'0.10': dict(sha256='1f3f45fad0c13ca2e55047c74a27677a370c0f04e96e16160867ca81bbb34cc3',
                         status='FINALIZED_FIVE_NATIVE7_VALID_SELECTIONS', job='10924262')}
GATES = ['actual_matching_natural_runtime', 'actual_full_source_student_inputs',
         'actual_full_selected_canonical_pilot', 'reviewed_longwall_runtime_entrypoint',
         'actual_collective_storage_reservation', 'exact_config_and_outer_permit_reviews',
         'reviewed_custom_method_selection_route_in_target']
REGISTRY = 's1-conditional-fulltrajectory-v1'
# Leave real submission headroom for other workloads sharing the same scheduler user.
SLOT_RESERVE = {'babel': 4, 'aces': 6}


def sources():
    own = {str(f.relative_to(ROOT)): c.sha(f) for f in sorted(HERE.iterdir())
           if f.suffix in ('.py', '.sbatch') and not f.name.startswith('test_')}
    return dict(sorted({**rp.sources(), **own}.items()))


def independent(value, status):
    c.require(value['status'] == status and isinstance(value.get('reviewer'), str) and value['reviewer'].strip()
              and value['reviewer'] not in AUTHORS, 'Genuine non-Session1 independent review required')
    return value


def percent_of(fraction):
    matches = [p for p in BUDGETS if f'{p / 100:.2f}' == fraction]
    c.require(len(matches) == 1, 'Registered budget fraction only')
    return matches[0]


def cell_id(fraction, arm, seed):
    c.require(arm in ARMS and seed in SEEDS, 'Registered S1 arm/seed only')
    return f's1-student-v1-{percent_of(fraction)}-{arm}-s{seed}'


def design(request, load=c.checked, admission=False):
    c.require(request['schema'] == SCHEMA and request['owner'] == OWNER and request['status'] == STATUS
              and request['sources'] == sources() and request['native_sources'] == c.sources()
              and request['recipe'] == ncfg.RECIPE, 'Exact S1 prospective held design/source required')
    site, arm, seed, fraction = request['site'], request['arm'], request['training_seed'], request['fraction']
    c.require(site in rp.SITES, 'Registered site only')
    percent = percent_of(fraction)
    c.require(request['cell_id'] == cell_id(fraction, arm, seed), 'Cell identity')
    receipt_ref = request['selection_receipt']
    pin = RECEIPTS.get(fraction)
    c.require(pin is not None and receipt_ref['sha256'] == pin['sha256'], 'Pinned finalized S1 receipt only')
    receipt = load(receipt_ref)
    c.require(receipt['status'] == pin['status'] and receipt['error'] is None and receipt['actual']['job_id'] == pin['job']
              and receipt['actual']['partition'] == 'general' and receipt['downstream_utility_established'] is False,
              'Finalized receipt status/ownership')
    entry = receipt['finalized'][arm]
    c.require(entry['returncode'] == 0 and entry['native7_validate_selection'] == 'PASS', 'Arm not natively sealed')
    paired = {}
    for method in ('uniform', 'source_sqrt'):
        ref = request['paired_root_requests'][method]
        root_request = load(ref)
        cell, _ = rp.design(root_request, load)
        sd = root_request['scientific_design']
        c.require(root_request['site'] == site and sd['method'] == method and sd['fraction'] == fraction
                  and sd['training_seed'] == seed and sd['selected_count'] == BUDGETS[percent]
                  and sd['selection_seed'] == 1234, 'Paired Root control differs in site/budget/seed')
        paired[method] = root_request
    c.require(sorted(request['paired_root_requests']) == ['source_sqrt', 'uniform'], 'Exactly two paired controls')
    anchor = paired['source_sqrt']
    for name in ('site_profile', 'scheduler_time_evidence', 'timing_basis'):
        c.require(request[name] == anchor[name] == paired['uniform'][name], 'Inherited route evidence differs: ' + name)
    asd = anchor['scientific_design']
    expected = dict(method=METHOD_PREFIX + arm, fraction=fraction, training_seed=seed, selection_seed=1234,
                    selected_count=BUDGETS[percent], selection=entry['selection'], logical_pool=asd['logical_pool'],
                    logical_eligibility=asd['logical_eligibility'], epoch=1, max_steps=-1, global_batch=128,
                    replicate_site=site)
    c.require(request['scientific_design'] == expected, 'Scientific factors or selected population changed')
    profile = rp.validate_historical_profile(load(request['site_profile']), load)
    if admission:
        ncfg.validate_profile(profile)
    c.require(profile['site'] == site and profile['code_root'] == str(ROOT), 'Authentic current source/site profile')
    allocation = request['outer_allocation']
    minutes = (rp.BABEL_MINUTES if site == 'babel' else rp.ACES_MINUTES)[percent]
    c.require(allocation == dict(gpus=1, cpus=8, memory_mb=65536, minutes=minutes)
              and allocation == anchor['outer_allocation'], 'Same outer route as paired controls only')
    c.require(request['future_release_gates'] == GATES and request['release_authorized'] is False
              and request['reserved_storage_bytes'] == 0, 'No fabricated gate or storage reservation')
    path = Path(request['control_root'])
    base = Path(profile['registry_root']) / REGISTRY
    c.require(path.is_absolute() and '..' not in path.parts and path.parent == base and path.name == request['cell_id']
              and not any(x.is_symlink() for x in (path, *path.parents)), 'Literal S1 registry control route')
    return profile, expected


def bare(request):
    return {k: v for k, v in request.items() if k != 'review'}


def reviewed(request_ref, load=c.checked, admission=False):
    request = load(request_ref)
    profile, expected = design(request, load, admission)
    review = independent(load(request['review']), REVIEW)
    c.require(review['request_identities'].get(request['cell_id']) == c.key(bare(request))
              and review['sources'] == sources() and review['native_sources'] == c.sources()
              and review['held_only'] is True, 'Exact independent S1 held design/source review mismatch')
    return request, expected, profile


def activation(binding_ref, activation_ref, load=c.checked):
    b = load(binding_ref)
    request, expected, _ = reviewed(b['request'], load)
    a = load(activation_ref)
    c.require(a['schema'] == 's1-fulltrajectory-actual-activation-v1' and a['binding'] == binding_ref
              and a['scientific_design'] == expected and a['held_sources'] == sources(),
              'Foreign physical/scientific activation')
    review = independent(load(a['review']), ACTIVATION_REVIEW)
    c.require(review['activation_identity'] == c.key(bare(a)) and review['binding'] == binding_ref
              and review['target_sources'] == a['target_sources'] and review['native_sources'] == c.sources()
              and review['actual_gate_descriptors'] == a['gates'], 'Actual job/source/config/gates review mismatch')
    c.require(set(a['gates']) == set(request['future_release_gates']) and all(a['gates'].values()),
              'All actual gates must be present; no future placeholders')
    for ref in a['gates'].values():
        load(ref)
    cfg = load(a['configuration'])
    runtime_profile = ncfg.validate_profile(load(cfg['site_profile']))
    route = rp.SITES[request['site']]
    c.require(runtime_profile['site'] == request['site'] and runtime_profile['code_root'] == str(ROOT)
              and tuple(runtime_profile[k] for k in ('slurm_user', 'account', 'partition', 'qos', 'gpu_type'))
              == tuple(route[k] for k in ('user', 'account', 'partition', 'qos', 'gpu'))
              and runtime_profile['max_minutes'] >= request['outer_allocation']['minutes'],
              'Fresh actual activation profile must authorize this route')
    c.require(cfg['sources'] == c.sources() and cfg['recipe'] == ncfg.RECIPE and cfg['role'] == 'student'
              and cfg['method'] == expected['method'] and cfg['fraction'] == expected['fraction']
              and cfg['training_seed'] == expected['training_seed'] and cfg['selection_seed'] == 1234
              and cfg.get('qualification_max_steps') is None, 'Actual config changed immutable S1 design')
    c.require(all(cfg['allocation'].get(k) == v for k, v in request['outer_allocation'].items()),
              'Actual config/outer allocation mismatch')
    logical = a['logical_input_registration']
    c.require(logical['original_pool'] == expected['logical_pool'] and logical['original_selection'] == expected['selection']
              and logical['actual_pool'] == cfg['providers']['pool']
              and logical['actual_selection'] == cfg['providers']['selection'],
              'Explicit original-to-current full source/selection binding required')
    target = a['target_module']
    c.require(re.fullmatch(r'experiments\.(root|s1|native)_[a-z0-9_]+\.[a-z0-9_]+', target) is not None
              and target.replace('.', '/') + '.py' in a['target_sources']
              and a['target_callable'] == 'run_conditional_student',
              'Explicit separately reviewed complete runtime entrypoint required')
    rp.verify_target_sources(a['target_sources'], target)
    return b, request, runtime_profile, a
