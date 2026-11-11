"""Explicit byte-identical HOME metadata projection, never a DATA fallback."""
from pathlib import Path
from . import protocol as p

ROOTS=('/home/luyangz/data_selection_multimodal/','/scratch/user/u.lz341613/')

class MetadataBindings:
 def __init__(self,pairs):
  self.pairs={}
  for pair in pairs:
   original=pair['original'];copy=pair['copy'];path=Path(copy['path'])
   p.require(path.is_absolute() and '/data/' not in str(path) and not any(x.is_symlink() for x in (path,*path.parents)),'Literal HOME/scratch metadata copy only')
   p.require(any(str(path).startswith(root) for root in ROOTS),'Approved owned metadata copy root')
   p.require(all(copy[k]==original[k] for k in ('bytes','sha256')) and p.desc(path)==copy,'Projection must preserve exact original bytes/SHA')
   key=p.key(original);p.require(key not in self.pairs,'Duplicate original binding');self.pairs[key]=(original,copy)
 def __call__(self,ref):
  value=self.pairs.get(p.key(ref))
  if value is not None:p.require(value[0]==ref,'Original descriptor changed');return p.checked(value[1])
  p.require(not str(ref['path']).startswith('/data/'),'Unbound DATA metadata; no login read/stat fallback')
  path=Path(ref['path']);p.require(not any(x.is_symlink() for x in (path,*path.parents)),'No symlink metadata fallback');return p.checked(ref)
