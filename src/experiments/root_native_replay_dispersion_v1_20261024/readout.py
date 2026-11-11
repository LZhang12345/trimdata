"""Measured fresh-process contrasts. A numeric failure stays a failure."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from . import protocol as p
from experiments.root_native_qualification_v10_20261024 import state
from experiments.root_native_qualification_v10_20261024.readout import check_behavior,verified_phase
from experiments.root_native_qualification_v10_20261024.observer import validate_lr_trajectory

OLD_RAW_SHA='efa9788cdab8b279a65cc854393ab52701c73d161fed102c6f3e6f11c9e1124d'

def compare_paths(a,b):
    import torch
    tol=p.TOLERANCE
    result={}
    # Sequential loads keep peak CPU use bounded to one pair of components.
    for name,loader in (('portable_weights',state.portable_weights),('Adam_FP32_master_scaler',state.native_optimizer)):
        first=loader(a);second=loader(b)
        result[name]=state.compare_tree(first,second,tol['absolute'],tol['relative']);del first,second
    result['saved_RNG']=state.compare_tree(state.normalize_saved_rng(torch.load(a/'rng_state.pth',map_location='cpu')),
        state.normalize_saved_rng(torch.load(b/'rng_state.pth',map_location='cpu')),0,0)
    def scheduler(path):
        if (path/'scheduler.pt').exists():return torch.load(path/'scheduler.pt',map_location='cpu')
        files=list(path.rglob('*model_states.pt'));p.require(len(files)==1,'Wrong scheduler shard count')
        return torch.load(files[0],map_location='cpu')['lr_scheduler']
    result['scheduler']=state.compare_tree(scheduler(a),scheduler(b),0,0)
    return result

def moment_state(checkpoint):
    value=state.native_optimizer(checkpoint)
    core=value['optimizer_state_dict']['state']
    p.require(len(core)==3 and all('exp_avg' in row and 'exp_avg_sq' in row for row in core.values()),
        'Actual three semantic Adam moment groups required')
    return {str(k):{name:v[name] for name in ('exp_avg','exp_avg_sq')} for k,v in core.items()}

def collect(study_ref):
    study=p.checked(study_ref);pair,modules=p.validate(study)
    p.require(os.environ.get('SLURM_JOB_ID') in set(study['jobs'].values()),
        'Tensor readout must run in a current owned diagnostic GPU allocation')
    p.require(int(os.environ['SLURM_JOB_END_TIME'])-time.time()>=600,
        'Insufficient real allocation lifetime for bounded native state readout; preserve cells')
    root=p.literal(study['report']);root.mkdir(parents=True,exist_ok=True)
    with (root/'READOUT.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/'RAW_DIAGNOSTIC.json').exists():return p.desc(root/'RAW_DIAGNOSTIC.json')
        phases={};observations={};paths={};receipts={};first={};ends={};losses={};criteria={};lr={}
        for cell,cfg in pair.items():
            receipt=p.read(Path(cfg['report'])/'CELL_COMPLETE.json')
            p.require(receipt['cell']==cell and receipt['study']==study_ref,'Foreign control result')
            phase=verified_phase(receipt['child']);phases[cell]=phase
            p.require(phase['cell']==cell and phase['actual_job_id']==study['jobs'][cell]
                and phase['actual_restart_count']==0 and phase['driver_sources']==p.sources()
                and phase['native_configuration']==study['native_configurations'][cell]
                and phase['native_target']==study['native_target'],'Actual fresh control identity differs')
            o=p.checked(phase['observation']);observations[cell]=o
            endpoint=p.CELLS[cell];end=p.checked(phase['files']['STEP-'+str(endpoint)+'.json'])
            p.expected_endpoint(cell,phase['status'],end['global_step'],o['actual_record_exposure'])
            p.require(o['actual_microbatches']==32*endpoint and o['mask_failures']==0
                and end['DS_skipped_steps']==0 and end['all_weights_finite'],'Incomplete/nonfinite/masked actual control')
            criteria[cell]=check_behavior(o);p.require(all(criteria[cell].values()),'Actual forward/mask/kernel criterion failed')
            first[cell]=p.checked(phase['files']['FIRST_MICRO_STATE.json']);ends[cell]=end
            lrrows=[json.loads(x) for x in Path(phase['files']['LEARNING_RATES.jsonl']['path']).read_text().splitlines()]
            lr[cell]=validate_lr_trajectory(lrrows,0,endpoint,8)
            paths[cell]=Path(cfg['output'])/'training'/('checkpoint-'+str(endpoint))
            receipts[cell]=modules['native'].checkpoint_receipt(paths[cell],p.n.run_identity(cfg))
            p.require(receipts[cell]['global_step']==endpoint,'Wrong accepted control step')
            losses[cell]=[json.loads(x)['returned_native_scaled_loss'] for x in
                Path(phase['files']['MICROBATCHES.jsonl']['path']).read_text().splitlines()]
        keys=('weights','optimizer','scheduler','rng','global_step','actual_DS_global_steps','DS_skipped_steps')
        initial={cell:{name:first[cell][name]==first['G1a'][name] for name in keys} for cell in p.CELLS}
        p.require(all(all(v.values()) for v in initial.values()),'Fresh controls did not start from identical state')
        p.require(observations['G1a']['ordered_record_ids']==observations['G1b']['ordered_record_ids']
            ==observations['U2']['ordered_record_ids'][:128],'First128 record order differs')
        p.require(len(set(observations['U2']['ordered_record_ids']))==1024,'U2 did not visit full construction without replacement')
        for cell in ('G1a','G1b'):
            p.require(first[cell]['weights']==ends[cell]['weights'],'Update1 has zero LR; portable weights changed')
        contrasts={'G1a_vs_G1b':compare_paths(paths['G1a'],paths['G1b'])}
        ma=moment_state(paths['G1a']);mb=moment_state(paths['G1b'])
        contrasts['G1a_vs_G1b']['Adam_first_and_second_moments']=state.compare_tree(ma,mb,**dict(atol=1e-6,rtol=1e-5))
        del ma,mb
        import numpy as np
        contrasts['G1a_vs_G1b']['microbatch_losses']=state.compare_tree(
            np.asarray(losses['G1a']),np.asarray(losses['G1b']),atol=1e-6,rtol=1e-5)
        old=p.checked(study['anchor_study']);oldraw_ref=p.desc(Path(old['report'])/'RAW_READOUT.json')
        p.require(oldraw_ref['sha256']==OLD_RAW_SHA,'Only authenticated actual original H100 comparison may anchor U2')
        raw=p.checked(oldraw_ref)
        p.require(raw['study_identity']==p.key(p.bare(old)) and raw['qualification_promoted'] is False,
            'Original failed readout identity differs')
        for name,arm in (('U1','reference'),('R1','resume')):
            cfg=p.checked(old['native_configurations'][arm]);path=Path(cfg['output'])/'training/checkpoint-8'
            modules['native'].checkpoint_receipt(path,p.n.run_identity(cfg))
            contrasts['U2_vs_'+name]=compare_paths(paths['U2'],path)
            old_phase=verified_phase(raw['phase_results']['reference' if arm=='reference' else 'resumed'])
            old_observation=p.checked(old_phase['observation'])
            expected_order=old_observation['ordered_record_ids']
            if arm=='resume':expected_order=p.checked(verified_phase(raw['phase_results']['interrupted'])['observation'])['ordered_record_ids']+expected_order
            p.require(expected_order==observations['U2']['ordered_record_ids'],'Historical endpoint data order differs')
        return p.seal(root/'RAW_DIAGNOSTIC.json',dict(schema='native7-fresh-process-dispersion-readout-v1',
            status='ACTUAL_FINITE_REPLAY_CONTRASTS_MEASURED_UNREVIEWED',owner='Root',study=study_ref,
            sources=p.sources(),native_target=study['native_target'],actual_cells={k:p.desc(Path(v['attempt'])/'CHILD_RESULT.json') for k,v in phases.items()},
            exact_initial_states=initial,actual_criteria=criteria,semantic_LR_trajectories=lr,
            observed_G1_weights_unchanged_at_LR0=True,numerical_comparisons=contrasts,
            original_failed_restart_comparison=oldraw_ref,original_numerical_comparisons=raw['numerical_comparisons'],
            actual_capsule_bytes={k:v['logical_bytes'] for k,v in receipts.items()},
            observed_costs={k:{name:o.get(name) for name in ('elapsed_seconds','observer_overhead_seconds','CUDA_peak_allocated','CUDA_peak_reserved')} for k,o in observations.items()},
            numerical_tolerance_unchanged=p.TOLERANCE,qualification_promoted=False,
            no_general_noise_bound_established=True,no_restore_qualification=True,no_scientific_pool_or_method_claim=True))

def main():
    a=argparse.ArgumentParser();a.add_argument('--study',required=True);x=a.parse_args()
    print(p.n.canonical(collect(p.desc(x.study))).decode())

if __name__=='__main__':main()
