"""Explicit reviewed observational overlay; never changes an accepted identity.

This API is not a scheduler/launcher and does not bypass the child's original
actual allocation, authorization, capsule, storage or Native91 checks. Root must
separately review the physical outer route that invokes it; old drivers do not
automatically use this API. No old activation/source projection is performed.
"""
import contextlib
import hashlib
import importlib
import os
from pathlib import Path

from experiments.root_native_campaign_v7_20261024 import common as n
from experiments.root_production_epoch_qualification_v1_20261024 import runtime as prior_runtime
from . import runtime, policy

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUTHOR = '/root/aces_fixture_review'
BASES = ('experiments.root_fulltrajectory_student_v2_20261025',
         'experiments.root_fulltrajectory_student_v4_20261025')
REVIEW = 'PASS_INDEPENDENT_EXACT_OBSERVATIONAL_ATTENTION_EXTENSION_V2'


def sources():
    return dict(sorted((str(p.relative_to(ROOT)), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in HERE.iterdir() if p.suffix == '.py' and not p.name.startswith('test_')))


def validate(extension_ref, configuration, authorization, launch_binding, child, load=n.checked):
    e = load(extension_ref)
    cfg, launch = load(configuration), load(launch_binding)
    base = child.p
    policy.require(base.PACKAGE in BASES and child.__name__ == base.PACKAGE+'.child'
        and Path(child.__file__).resolve() == ROOT/Path(base.PACKAGE.replace('.', '/'))/'child.py'
        and child.runtime is prior_runtime, 'Only an unchanged explicit old child/runtime may be overlaid')
    identity = n.run_identity(cfg)
    expected_adapter = dict(package=base.PACKAGE, sources=base.sources(), runtime=base.RUNTIME,
        natural_epoch=True, ordinary_native_launch_forbidden=True)
    policy.require(cfg['execution_adapter'] == expected_adapter and cfg['sources'] == n.sources(),
        'Original scientific adapter/config/Native91 identity must remain unchanged')
    policy.require(e['status'] == 'ROOT_REGISTERED_OBSERVATIONAL_ATTENTION_TELEMETRY_EXTENSION_V2'
        and e['owner'] == 'Root' and e['configuration'] == configuration
        and e['authorization'] == authorization and e['launch_binding'] == launch_binding
        and e['scientific_config_identity'] == identity and e['base_adapter'] == expected_adapter
        and e['extension_sources'] == sources() and e['math_policy'] == policy.POLICY
        and e['telemetry_schema'] == policy.TELEMETRY
        and e['state_policy'] == 'KEEP_ORIGINAL_CONFIG_ACCEPTED_IDENTITY_NO_STATE_MIGRATION'
        and e['no_state_relabel_or_tensor_change'] is True
        and e['actual_GPU_or_restore_qualification_claimed'] is False,
        'Exact independently reviewed observational extension/config contract required')
    policy.require(launch['configuration'] == configuration and launch['authorization'] == authorization
        and launch['adapter_sources'] == base.sources()
        and e['physical_job_id'] == launch['physical_job_id'] == os.environ.get('SLURM_JOB_ID')
        and e['actual_restart_count'] == launch['actual_restart_count']
        == int(os.environ.get('SLURM_RESTART_COUNT', '0'))
        and e['current_capsule_guard'] == launch['current_storage_guard'],
        'Exact actual physical extension/child/capsule custody graph required')
    guard = load(launch['current_storage_guard'])
    policy.require(guard['configuration'] == configuration and guard['config_identity'] == identity
        and guard['physical_job_id'] == e['physical_job_id']
        and guard['actual_restart'] == e['actual_restart_count']
        and guard['latest_step'] == e['base_accepted_cursor'], 'Current original-identity state guard required')
    history = launch['accepted_prior_attempts']
    continuation = history[-1] if history else None
    # Reuse the exact old accepted/header/cursor/hash custody validation; do not
    # rewrite any ACCEPTED.json, optimizer/RNG/cursor file, config, or old receipt.
    driver = importlib.import_module(base.PACKAGE+'.driver')
    driver.accepted_cursor_guard(guard, guard['latest_step'], continuation, configuration)
    review = load(e['review'])
    policy.require(review['status'] == REVIEW
        and review['extension_identity'] == n.key({k:v for k,v in e.items() if k != 'review'})
        and review['extension_sources'] == sources() and review['base_sources'] == base.sources()
        and review.get('reviewer') not in (None, '', AUTHOR),
        'Genuine independent exact source/extension/current-state review required')
    return e, dict(extension=extension_ref, review=e['review'], extension_sources=sources(),
        base_adapter=expected_adapter, original_configuration=configuration,
        original_config_identity=identity, original_capsule_guard=e['current_capsule_guard'],
        original_accepted_cursor=e['base_accepted_cursor'], state_migration_performed=False,
        actual_GPU_or_restore_qualification_claimed=False)


@contextlib.contextmanager
def scope(extension_ref, configuration, authorization, launch_binding, child, load=n.checked):
    policy.require(runtime.ACTIVE_EXTENSION is None, 'Nested child/runtime overlay forbidden')
    _, attestation = validate(extension_ref, configuration, authorization, launch_binding, child, load)
    original = child.runtime
    runtime.ACTIVE_EXTENSION = attestation
    child.runtime = runtime
    try:
        yield attestation
    finally:
        try:
            runtime.cleanup()
        finally:
            child.runtime = original
            runtime.ACTIVE_EXTENSION = None


def run(extension_ref, configuration, authorization, launch_binding, attempt, stop_at):
    e = n.checked(extension_ref)
    package = e['base_adapter']['package']
    policy.require(package in BASES, 'No unreviewed child-family fallback')
    child = importlib.import_module(package+'.child')
    with scope(extension_ref, configuration, authorization, launch_binding, child):
        return child.run(configuration, authorization, launch_binding, attempt, stop_at)
