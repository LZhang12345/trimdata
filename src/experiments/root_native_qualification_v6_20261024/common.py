"""Versioned engineering driver identities; scientific V2 remains immutable."""
import importlib
from pathlib import Path
from experiments.root_native_campaign_20261023 import common as frozen
from experiments.root_native_campaign_20261023.common import canonical,checked,desc,key,now,read,require,seal,sha,within

ROOT=frozen.ROOT
HERE=Path(__file__).resolve().parent

def sources():
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.sbatch'))
    paths += [ROOT/'experiments/root_native_campaign_20261023/common.py']
    return dict(sorted((str(p.relative_to(ROOT)),sha(p)) for p in paths if not p.name.startswith('test_')))

def fixture_sources():
    paths=[HERE/'fixture.py',HERE/'common.py',ROOT/'experiments/root_native_campaign_20261023/common.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def target_modules(target):
    """Explicit corrected package only; no fallback to an older trainer."""
    package=target['package']
    require(package.startswith('experiments.root_native_') and package not in (
        'experiments.root_native_campaign_20261023','experiments.root_native_selection_20261023'),
        'Known V2 attention/checkpoint API issues prohibit engineering qualification of V2')
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
    require(study['schema']=='native-engineering-parity-study-v1' and study['owner']=='Root','Wrong engineering study')
    require(study['driver_sources']==sources(),'Qualification driver source changed')
    require(study['total_steps']>=4 and 0<study['interrupt_step']<study['total_steps'],'Need a real pre/post restart schedule')
    require(study['numeric_tolerance']=={'absolute':1e-6,'relative':1e-5},'Freeze numeric tolerance before outcomes')
    review=checked(study['review'])
    require(review['status']=='PASS_INDEPENDENT_ENGINEERING_DRIVER_AND_STUDY'
        and review['driver_sources']==study['driver_sources']
        and review['study_identity']==key({k:v for k,v in study.items() if k!='review'}),'Exact independent engineering review required')
    modules,packet=target_modules(study['native_target'])
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
    for field in ('recipe','sources','site_profile','training_seed','selection_seed','allocation'):
        require(configs['reference'][field]==configs['resume'][field],'Unpaired engineering '+field)
    for field in ('assets','environment','pool','selection','eligibility'):
        require(configs['reference']['providers'][field]==configs['resume']['providers'][field],'Unpaired provider '+field)
    require(configs['reference']['output']!=configs['resume']['output'],'Reference and restart must own distinct directories')
    return configs,modules
