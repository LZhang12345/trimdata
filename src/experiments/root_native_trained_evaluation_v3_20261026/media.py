"""Physical GQA member adapter; original encoded bytes/RGB stay authoritative."""
from pathlib import Path,PurePosixPath
from experiments.root_babel_start_evaluation_v2_20261025.media import Store as OriginalStore,fingerprint
from experiments.root_gqa_bank_qualification_v1_20261026 import worker as u
CACHE_ROOT=Path('/data/hf_cache')

class Store(OriginalStore):
    def payload(self,name):
        if 'explicit_members' not in self.cfg:return super().payload(name)
        u.require(name in self.mapping and name in self.cfg['explicit_members'],'Undeclared GQA native image')
        expected=self.images[self.mapping[name]];ref=self.cfg['archive'];member=self.cfg['explicit_members'][name]
        u.require(Path(ref['path']).resolve().is_relative_to(CACHE_ROOT) and
            fingerprint(ref['path'])==self.archive_snapshot and not PurePosixPath(member).is_absolute() and '..' not in PurePosixPath(member).parts,
            'Qualified GQA source archive/member changed')
        z,names=self.archive(ref['path']);u.require(names[member]==1,'Absent/duplicate original GQA member')
        info=z.getinfo(member)
        u.require(not info.is_dir() and (info.external_attr>>16)&0o170000!=0o120000 and
            0<info.file_size==expected['encoded_bytes']<=64<<20,'Bounded regular original GQA image required')
        with z.open(info) as f:raw=f.read((64<<20)+1)
        u.require(fingerprint(ref['path'])==self.archive_snapshot and len(raw)==expected['encoded_bytes'] and u.sha(raw)==expected['encoded_sha256'],
            'Actual GQA encoded bytes/archive changed')
        return raw,expected
