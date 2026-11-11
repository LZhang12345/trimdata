"""Additional driver-only guards; actual executed fixture common stays frozen."""
from . import common as c

def assert_scoped_pair(study,configs):
    c.require(study['total_steps']==8 and study['interrupt_step']==4,'This actual qualification is frozen to8updates/cut4')
    c.require(isinstance(study['minimum_launch_seconds'],int) and study['minimum_launch_seconds']>=60,'A bounded launch reserve is required')
    for arm,config in configs.items():
        c.require('media_transport' not in config['providers'] and 'runtime_images' not in config['providers'],
            'This first actual driver is Babel canonical-images only; transported qualification requires a reviewed next driver')
        c.require(c.checked(config['site_profile'])['site']=='babel','This first actual driver is Babel1L40S only')
        pool=c.checked(config['providers']['pool'])
        c.require(pool['count']==1024,'This engineering study requires the exact1024-record fixture')
        from .masking import RECORDS_SHA256
        c.require(pool['records_json']['sha256']==RECORDS_SHA256,
            'The repaired qualification must preserve every original fixture conversation')
        c.require(config['checkpoint_seconds']>config['allocation']['minutes']*60,
            'Periodic save could create/rotate a different boundary during bounded parity')
        storage=c.checked(config['storage_reservation'])
        c.require(storage.get('allow_new_intermediate_retirement',False) is False,
            'This study requires actual cut4 state retained for independent raw readout')
        expected=2 if arm=='resume' else 1
        c.require(storage.get('retained_accepted_states',1)>=expected,'Insufficient accepted-state retention for scoped arm')
    return configs

def validate_study(study):
    configs,modules=c.validate_study(study)
    assert_scoped_pair(study,configs)
    return configs,modules


def native_identity(config,modules):
    c.require(callable(getattr(modules['common'],'run_identity',None)),'Target must expose reviewed physical-alias-aware run_identity')
    return modules['common'].run_identity(config)
