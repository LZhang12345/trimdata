"""V10 identities; all executed V5 native and V9 driver sources stay immutable."""
import importlib
from pathlib import Path
from experiments.root_native_campaign_20261023 import common as frozen
from experiments.root_native_campaign_20261023.common import canonical,checked,desc,key,now,read,require,seal,sha,within

ROOT=frozen.ROOT
HERE=Path(__file__).resolve().parent

def sources():
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.sbatch'))
    paths += [ROOT/'experiments/root_native_campaign_20261023/common.py']
    paths += [ROOT/'experiments/root_native_qualification_v6_20261024'/name
              for name in ('__init__.py','common.py','masking.py')]
    return dict(sorted((str(p.relative_to(ROOT)),sha(p)) for p in paths if not p.name.startswith('test_')))

def target_modules(target):
    """Explicit corrected package only; no fallback to an older trainer."""
    package=target['package']
    require(package=='experiments.root_native_campaign_v7_20261024',
        'V10 qualifies only the reviewed early-empty-pruned NativeV7 target, never old V3/V4/V5')
    require(all(part.isidentifier() for part in package.split('.')),'Unsafe target module')
    packet=checked(target['source_packet'])
    require(isinstance(packet.get('sources'),dict) and packet['sources'],'Target source packet missing')
    modules={name:importlib.import_module(package+'.'+name) for name in ('common','config','native')}
    expected=ROOT/Path(*package.split('.'))
    for name,module in modules.items():
        require(Path(module.__file__).resolve()==(expected/(name+'.py')).resolve(),'Foreign target module: '+name)
    require(modules['common'].sources()==packet['sources'],'Target sources changed after review')
    return modules,packet

def validate_study(study):
    require(study['schema']=='native-engineering-parity-study-v10' and study['owner']=='Root','Wrong engineering study')
    require(study['driver_sources']==sources(),'Qualification driver source changed')
    require(study['total_steps']>=4 and 0<study['interrupt_step']<study['total_steps'],'Need a real pre/post restart schedule')
    require(study['numeric_tolerance']=={'absolute':1e-6,'relative':1e-5},'Freeze numeric tolerance before outcomes')
    review=checked(study['review'])
    require(review['status']=='PASS_INDEPENDENT_ENGINEERING_DRIVER_AND_STUDY'
        and review['driver_sources']==study['driver_sources']
        and review['study_identity']==key({k:v for k,v in study.items() if k!='review'}),'Exact independent engineering review required')
    modules,packet=target_modules(study['native_target'])
    from . import masking
    contract=study['masking_contract']
    require(contract['name']==masking.CONTRACT
        and contract['oracle_sha256']==sha(HERE/'masking.py')
        and contract['records_sha256']==masking.RECORDS_SHA256
        and contract['repaired_record_ids']==[masking.REPAIRED_ID]
        and contract['unaffected_records']==1023,
        'V6 structured oracle, exact unchanged fixture and one explicit repair required')
    diagnosis=checked(contract['original_diagnosis'])
    require(len(diagnosis['records'])==1024 and diagnosis['bad_records']==1
        and [r['record_id'] for r in diagnosis['records'] if r['status']!='TARGETS_SURVIVE']==[masking.REPAIRED_ID],
        'Actual Root full1024 original-tokenizer diagnosis required')
    configs={arm:checked(study['native_configurations'][arm]) for arm in ('reference','resume')}
    for arm,config in configs.items():
        modules['config'].validate(config)
        require(config['sources']==packet['sources'],'Native config target source identity differs')
        require(config['role']=='qualify' and config['method']=='full' and config['fraction']=='1','Only full engineering fixtures, no scientific selector')
        require(config['allocation']['gpus']==1 and config['allocation']['microbatch']==4,'One GPU micro4/global128 qualification')
        require(config['qualification_max_steps']==study['total_steps'],'Engineering step limit differs')
        pool=checked(config['providers']['pool'])
        require(pool['schema']=='native-engineering-pool-v1' and pool['engineering_only'] is True and pool['count']>=1024,'Actual explicit engineering fixture required')
        claim=checked(config['providers']['ownership'])
        require(claim.get('qualification_study')==study['id'] and claim.get('qualification_arm')==arm,'Arm must be registered as engineering replication')
        require(config['storage_reservation'],'Measured two-arm accepted-state storage required')
        from .preflight import verify_shared
        verify_shared(study,config,raw=True)
    for field in ('recipe','sources','site_profile','training_seed','selection_seed','allocation'):
        require(configs['reference'][field]==configs['resume'][field],'Unpaired engineering '+field)
    for field in ('assets','environment','pool','selection','eligibility'):
        require(configs['reference']['providers'][field]==configs['resume']['providers'][field],'Unpaired provider '+field)
    require(configs['reference']['output']!=configs['resume']['output'],'Reference and restart must own distinct directories')
    return configs,modules
