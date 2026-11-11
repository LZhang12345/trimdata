"""Explicit Root10 observational runtime route, never an implicit old-child fallback."""
import argparse,contextlib,datetime,hashlib,json,os,re,socket,subprocess,time
from pathlib import Path
from zoneinfo import ZoneInfo
from experiments.root_fulltrajectory_student_v4_20261025 import protocol as p,shared_runtime as logical_runtime
from experiments.root_fulltrajectory_student_v4_20261025 import driver as base_driver,child as base_child
from experiments.root_fulltrajectory_held_cohort_v2_20261025 import protocol as held
from experiments.root_fulltrajectory_held_cohort_v2_20261025.admit import wall
from experiments.root_production_student_launcher_v4_20261025 import control as transport
from experiments.root_production_epoch_qualification_v1_20261024 import runtime as prior_runtime
from experiments.root_native_deterministic_attention_v2_20261025 import runtime as telemetry,bridge
from experiments.root_native_transition_20261023 import file_guards_v1 as guards

ROOT=p.ROOT;HERE=Path(__file__).resolve().parent;PACKAGE='experiments.root_control10_telemetry_physical_v1_20261026'
AUTHOR='/root/cache_alias_finish';ACTIVE=False;ORIGINAL_CURRENT=base_driver.current
ALLOWED_CELLS=tuple(sorted(p.cell_id('10:'+method,seed) for method in ('uniform','source_sqrt') for seed in (17,29,43)))
POLICY_REVIEW='PASS_INDEPENDENT_EXPLICIT_ROOT_CONTROL10_TELEMETRY_PHYSICAL_POLICY_V1'
EXPORT=dict(mountpoint='/data/hf_cache',filesystem_type='nfs4',source='babel-nas-2:/srv/nfs/data/hf_cache',root='/')


def sources():
    own={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.iterdir()
         if f.suffix in ('.py','.sbatch') and not f.name.startswith('test_')}
    guard=Path(guards.__file__).resolve()
    return dict(sorted({**p.sources(),**bridge.sources(),str(guard.relative_to(ROOT)):hashlib.sha256(guard.read_bytes()).hexdigest(),**own}.items()))


def physical_policy(binding_ref,activation_ref,execution_ref,e,cfg,load=p.checked):
    a=load(activation_ref);b=load(binding_ref);ref=a['physical_extension_policy'];reviewref=a['physical_extension_review']
    value=load(ref);review=load(reviewref);packet=load(value['source_packet'])
    p.require(a['target_module']==PACKAGE+'.driver' and a['target_sources']==sources()
        and e['binding']==binding_ref and e['activation']==activation_ref and e['configuration']==a['configuration']
        and e['physical_extension_policy']==ref and e['physical_extension_review']==reviewref
        and e['physical_sources']==sources() and e['physical_source_packet']==value['source_packet'] and e['mode']=='production',
        'Exact independent physical activation/execution graph; old target cannot route implicitly')
    p.require(value['schema']=='root-control10-explicit-telemetry-physical-policy-v1' and value['owner']=='Root'
        and value['status']=='ROOT_REGISTERED_EXPLICIT_ROOT_CONTROL10_OBSERVATIONAL_PHYSICAL_POLICY_V1'
        and value['executed_sources']==sources() and value['logical_sources']==p.sources()
        and value['native_sources']==p.n.sources() and value['telemetry_sources']==bridge.sources()
        and value['numerical_bodies']==logical_runtime.numerical_bodies()
        and value['allowed_cells']==list(ALLOWED_CELLS) and cfg['id'] in value['allowed_cells']
        and value['logical_runtime']==p.RUNTIME and value['telemetry_math_policy']==telemetry.policy.POLICY
        and value['telemetry_schema']==telemetry.policy.TELEMETRY
        and value['checkpoint_identity_migration'] is False and value['new_training_pilot_claimed'] is False,
        'Exact source-qualified observational extension, never changed logical math/state identity')
    p.require(packet['executed_sources']==sources() and packet['logical_sources']==p.sources()
        and packet['native_sources']==p.n.sources() and packet['telemetry_sources']==bridge.sources(), 'Exact physical source packet')
    p.require(review['status']==POLICY_REVIEW and review['policy']==ref and review['source_packet']==value['source_packet']
        and review['executed_sources']==sources() and review['logical_sources']==p.sources()
        and review['independent'] is True and review['reviewer'] not in (None,'','Root','/root',AUTHOR),
        'Independent explicit Root10 physical source/policy review required')
    logical_runtime.source_bridge(value['logical_numerical_selection_bridge'],load)
    gpu=load(value['actual_GPU_operator_qualification']);peer=load(value['actual_GPU_operator_review']);gp=load(gpu['binding']['source_packet'])
    p.require(gpu['status']=='PASS_ACTUAL_GPU_LAYOUT_TELEMETRY_EQUIVALENCE_V1' and gpu['distinct_planned_layouts']>512
        and gpu['late_unsupported_rejected'] is True and gpu['deterministic_flag_reset_rejected'] is True
        and all(gp['sources'].get(k)==v for k,v in bridge.sources().items())
        and peer['status']=='PASS_INDEPENDENT_ACTUAL_LAYOUT_GPU_QUALIFICATION_V1'
        and peer['readout']==value['actual_GPU_operator_qualification'] and peer['reviewer'] not in (None,'','Root','/root',AUTHOR),
        'Reuse only actual independently reviewed exact telemetry GPU qualification')
    cert=load(e['runtime_certificate'])
    p.require(value['actual_v4_canary']==cert['actual_v4_canary'] and value['actual_v4_canary_review']==cert['actual_v4_canary_review'],
        'Same genuine independently qualified V4 canary; never invented Root10 pilot labels')
    for rel,sha in sources().items():p.require(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,'Executed physical source changed: '+rel)
    return dict(schema='EXPLICIT_ROOT_CONTROL10_PHYSICAL_TELEMETRY_EXECUTION_V1',binding=binding_ref,activation=activation_ref,execution=execution_ref,
        physical_extension_policy=ref,physical_extension_review=reviewref,source_packet=value['source_packet'],
        executed_sources=sources(),logical_sources=p.sources(),telemetry_sources=bridge.sources(),
        original_configuration=a['configuration'],original_config_identity=p.n.run_identity(cfg),
        actual_job_id=b['job_id'],actual_restart=e['actual_restart'],checkpoint_identity_migration=False,new_training_pilot_claimed=False)


def documents(binding_ref,activation_ref,load=p.checked):
    b,request,profile,a=held.activation(binding_ref,activation_ref,load)
    cfg=load(a['configuration']);restart=os.environ.get('SLURM_RESTART_COUNT','0')
    p.require(restart.isdecimal(),'Actual restart number')
    eref=p.desc(Path(cfg['report'])/'ROOT_LONG/requests'/('restart-'+restart+'.json'));e=load(eref)
    proof=physical_policy(binding_ref,activation_ref,eref,e,cfg,load)
    return b,request,profile,a,cfg,eref,e,proof


def current(binding_ref,activation_ref,runner=None,minimum_seconds=600):
    runner=runner or subprocess.check_output
    b,request,profile,a,cfg,eref,e,proof=documents(binding_ref,activation_ref);held.actor('babel')
    p.require(request['site']=='babel' and profile['site']=='babel' and b['schema']=='root-fulltrajectory-held-binding-v2'
        and b['sources']==held.sources() and b['command']==p.desc(held.HERE/'run.sbatch')
        and binding_ref==p.desc(Path(request['control_root'])/'BINDING.json'), 'Original fresh Root10 physical/source binding required')
    intent=p.checked(b['intent']);ack=p.checked(b['ack'])
    p.require(intent['request']==b['request'] and intent['nonce']==b['nonce'] and ack['job_id']==b['job_id'] and ack['intent']==b['intent'], 'Actual original intent/ACK custody')
    job=os.environ.get('SLURM_JOB_ID','');restart=os.environ.get('SLURM_RESTART_COUNT','0')
    p.require(job==b['job_id'] and job.isdecimal() and restart.isdecimal(),'Genuine current physical job/restart')
    raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30);p.require(len(raw.strip().splitlines())==1,'One actual scalar')
    f=transport.fields(raw);expected=dict(JobId=job,JobName=b['job_name'],JobState='RUNNING',UserId='luyangz(2708899)',Account='cx',Partition='general',QOS='normal',
        NumNodes='1',NumTasks='1',NumCPUs='8',TimeLimit=wall(request['outer_allocation']['minutes']),Command=b['command']['path'],WorkDir=str(ROOT),
        Comment='dsmm-fulltrajectory:'+b['nonce'],TresPerNode='gres/gpu:L40S:1',Restarts=restart,NodeList=socket.gethostname().split('.')[0])
    p.require(all(f.get(k)==v for k,v in expected.items()) and f.get('ArrayJobId') in (None,'(null)'), 'Actual original General scalar/UID/nonce/resources')
    t=dict(v.split('=',1) for v in f.get('AllocTRES','').split(',') if '=' in v)
    p.require(t.get('cpu')=='8' and t.get('mem') in ('64G','65536M') and t.get('gres/gpu')=='1','AllocatedCPU8/64GiB/L40Root10')
    p.require(p.desc(b['spool']['path'])==b['spool'] and runner(['scontrol','write','batch_script',job,'-'],timeout=30)==Path(b['spool']['path']).read_bytes(),'Actual bound heldV2 spool')
    end=int(datetime.datetime.fromisoformat(f['EndTime']).replace(tzinfo=ZoneInfo('America/New_York')).timestamp())
    p.require(end>time.time()+minimum_seconds and (not os.environ.get('SLURM_JOB_END_TIME') or int(os.environ['SLURM_JOB_END_TIME'])==end), 'Actual Eastern EndTime/reserve')
    # Retain all original logical execution/history/certificate/permission checks.
    execution(binding_ref,activation_ref,dict(job_id=job,restart=int(restart)))
    return b,request,dict(job_id=job,restart=int(restart),node=f['NodeList'],end_epoch=end,scheduler=raw,physical_extension=proof),a


def verify_physical_receipt(receipt_ref,binding_ref,activation_ref,restart,load=p.checked):
    b=load(binding_ref);a=load(activation_ref);cfg=load(a['configuration'])
    expected=Path(cfg['report'])/'attempts'/(b['job_id']+'-r'+str(restart)+'-'+b['nonce'][:8])/'child'
    p.require(Path(receipt_ref['path']).parent==expected,'Exact actual physical child attempt path')
    receipt=load(receipt_ref);proof_ref=p.desc(expected/'PHYSICAL_EXECUTION_PROOF.json');proof=load(proof_ref)
    eref,e,_,_,_=base_driver.execution(binding_ref,activation_ref,dict(restart=restart))
    extension=physical_policy(binding_ref,activation_ref,eref,e,cfg,load)
    p.require(proof['physical_extension']==extension and proof['child_receipt']==receipt_ref
        and proof['configuration']==a['configuration'] and proof['config_identity']==p.n.run_identity(cfg)
        and proof['executed_sources']==sources() and proof['logical_sources']==p.sources()
        and proof['operators']==receipt['operators'] and receipt['actual_job_id']==b['job_id']
        and receipt['actual_restart_count']==restart
        and load(proof['operators'])['observational_execution_extension']==extension,
        'Every accepted segment requires genuine exact physical sources/operators')
    actual=proof['actual'];export=load(proof['checkpoint_export']);storage=load(p.desc(expected/'ACTUAL_STORAGE.json'))
    p.require(actual['job_id']==b['job_id'] and actual['restart']==restart
        and export['export_identity']==EXPORT and export['actual_job_id']==actual['job_id']
        and export['actual_restart']==restart and export['actual_node']==actual['node']
        and export['observed_device']==storage['checkpoint_root_device']
        and export['after_segment_observation']['export_identity']==EXPORT
        and export['after_segment_observation']['observed_device']==export['observed_device'],
        'Same-segment pinned export/storage/actor proof, no historical device equality')
    return proof_ref


def execution(binding_ref,activation_ref,actual):
    result=base_driver.execution(binding_ref,activation_ref,actual)
    physical_policy(binding_ref,activation_ref,result[0],result[1],result[2])
    for restart,ref in enumerate(result[1]['accepted_history']):
        verify_physical_receipt(ref,binding_ref,activation_ref,restart)
    return result


@contextlib.contextmanager
def scope(binding_ref,activation_ref,*,use_telemetry=False):
    global ACTIVE
    p.require(not ACTIVE and base_driver.current is ORIGINAL_CURRENT and base_child.runtime is prior_runtime
        and telemetry.ACTIVE_EXTENSION is None,'Unmodified single Root10 physical scope only')
    b,r,profile,a,cfg,eref,e,proof=documents(binding_ref,activation_ref)
    def bound_current(physical,activation,runner=None,minimum_seconds=600):
        p.require(physical==binding_ref and activation==activation_ref,'Exact scoped Root10 physical descriptors')
        return current(physical,activation,runner,minimum_seconds)
    ACTIVE=True;base_driver.current=bound_current
    try:
        if use_telemetry:telemetry.ACTIVE_EXTENSION=proof;base_child.runtime=telemetry
        yield proof
    finally:
        try:
            if use_telemetry:telemetry.cleanup()
        finally:base_driver.current=ORIGINAL_CURRENT;base_child.runtime=prior_runtime;telemetry.ACTIVE_EXTENSION=None;ACTIVE=False


def checkpoint_export(root,actual):
    p.require(root==p.CHECKPOINT_ROOT and Path(root).is_dir(),'Exact genuine shared checkpoint root')
    def decode(x):return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m.group(1),8)),x)
    matches=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,sep,right=line.partition(' - ')
        if not sep:continue
        a=left.split();z=right.split()
        if len(a)<6 or len(z)<3:continue
        item=dict(mountpoint=decode(a[4]),filesystem_type=z[0],source=decode(z[1]),root=decode(a[3]))
        if item==EXPORT:matches.append(dict(item,raw=line,major_minor=a[2]))
    p.require(len(matches)==1,'Exact backing NFS export, not same-point autofs')
    dev=os.stat(root).st_dev;mount=matches[0]
    p.require(mount['major_minor']=='%d:%d'%(os.major(dev),os.minor(dev)),'Same-process live mount/device identity')
    return dict(status='ACTUAL_ROOT_CONTROL10_PHYSICAL_SHARED_EXPORT_V1',checkpoint_root=root,export_identity=EXPORT,observed_device=dev,mount=mount,
        mount_namespace=os.readlink('/proc/self/ns/mnt'),actual_job_id=actual['job_id'],actual_restart=actual['restart'],actual_node=actual['node'],device_number_is_namespace_local=True)


def capsule_mode(binding_ref,activation_ref,output):
    from experiments.root_fulltrajectory_student_v4_20261025 import capsule
    _,_,actual,_=current(binding_ref,activation_ref,minimum_seconds=30)
    pin=guards.Chain(Path(p.CHECKPOINT_ROOT)/'ROOT_CONTROL10_PHYSICAL_ANCHOR',Path('/data/hf_cache'))
    try:
        pin.validate_live(pin.ids);export=checkpoint_export(p.CHECKPOINT_ROOT,actual)
        with scope(binding_ref,activation_ref) as proof:
            ref=capsule.check(binding_ref,activation_ref,output)
            pin.validate_live(pin.ids);after=checkpoint_export(p.CHECKPOINT_ROOT,actual)
            p.require(after['export_identity']==export['export_identity'] and after['observed_device']==export['observed_device'], 'Same-process capsule export changed')
            export['after_segment_observation']=after
            p.seal(str(output)+'.PHYSICAL.json',dict(guard=ref,physical_extension=proof,executed_sources=sources(),checkpoint_export=export))
            return ref
    finally:pin.close()


def child_mode(binding_ref,activation_ref,launch_ref,attempt,stop):
    b,r,actual,a=current(binding_ref,activation_ref);eref,e,cfg,cursor,root=execution(binding_ref,activation_ref,actual)
    launch=p.checked(launch_ref);proof=physical_policy(binding_ref,activation_ref,eref,e,cfg)
    p.require(launch['physical_execution']==eref and launch['physical_sources']==sources()
        and launch['physical_extension']==proof and launch['physical_source_packet']==proof['source_packet'], 'Explicit new physical child launch proof required')
    guardproof=p.checked(launch['physical_capsule_guard'])
    p.require(guardproof['guard']==launch['current_storage_guard'] and guardproof['physical_extension']==proof and guardproof['executed_sources']==sources(), 'Exact scoped physical capsule proof')
    pin=guards.Chain(Path(p.CHECKPOINT_ROOT)/'ROOT_CONTROL10_PHYSICAL_ANCHOR',Path('/data/hf_cache'))
    try:
        pin.validate_live(pin.ids);export=checkpoint_export(p.CHECKPOINT_ROOT,actual)
        with scope(binding_ref,activation_ref,use_telemetry=True) as active:
            code,receipt=base_child.run(a['configuration'],e['authorization'],launch_ref,attempt,stop)
            measured=p.checked(p.checked(receipt)['operators'])
            p.require(measured['observational_execution_extension']==active,'Actual operator receipt must retain exact physical runtime extension')
            pin.validate_live(pin.ids);after=checkpoint_export(p.CHECKPOINT_ROOT,actual)
            p.require(after['export_identity']==export['export_identity'] and after['observed_device']==export['observed_device'],'Pinned same-process checkpoint export changed')
            export['after_segment_observation']=after
            p.seal(Path(attempt)/'ACTUAL_SHARED_CHECKPOINT_EXPORT.json',export)
            p.seal(Path(attempt)/'PHYSICAL_EXECUTION_PROOF.json',dict(actual=actual,physical_extension=active,executed_sources=sources(),logical_sources=p.sources(),
                configuration=a['configuration'],config_identity=p.n.run_identity(cfg),child_receipt=receipt,operators=p.checked(receipt)['operators'],
                logical_exit_code=code,checkpoint_export=p.desc(Path(attempt)/'ACTUAL_SHARED_CHECKPOINT_EXPORT.json')))
            base_child.emit_exit(a['configuration'],code,receipt,attempt)
            return 0
    finally:pin.close()


def main():
    a=argparse.ArgumentParser()
    for name in ('binding','activation'):a.add_argument('--'+name,required=True)
    a.add_argument('--mode',choices=('capsule','child'),required=True)
    for name in ('output','launch','attempt'):a.add_argument('--'+name)
    a.add_argument('--stop-at-epoch',type=int);v=a.parse_args()
    if v.mode=='capsule':return capsule_mode(p.desc(v.binding),p.desc(v.activation),v.output)
    return child_mode(p.desc(v.binding),p.desc(v.activation),p.desc(v.launch),v.attempt,v.stop_at_epoch)


if __name__=='__main__':print(json.dumps(main()))
