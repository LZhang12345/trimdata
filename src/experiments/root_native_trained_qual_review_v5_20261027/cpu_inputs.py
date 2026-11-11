"""Read-only isolated execution of exact native CPU input source modules.

No native package/model initializer runs; no scientific function is rewritten.
The sole native intra-module import resolves the separately authenticated
constants object. The calling reviewer retains its model/network denial guard.
"""
import builtins
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

PREFIX='_dsmm_independent_native_cpu_inputs_v5'
NAMES=('constants','conversation','mm_utils')

def require(ok,message):
    if not ok:raise ValueError(message)

@contextmanager
def primitives(source_root,llava_root,sources):
    root=Path(source_root);native=Path(llava_root)/'llava'
    require(root.is_absolute()and native.is_absolute()and native.is_relative_to(root),
        'Pinned native input sources must reside inside authenticated subject root')
    modules={};proof=[]
    try:
        for name in NAMES:
            path=native/(name+'.py');relative=str(path.relative_to(root))
            require(relative in sources and not any(p.is_symlink()for p in(path,*path.parents)),
                'Exact literal native CPU source required')
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            require(digest==sources[relative],'Native CPU input source SHA differs')
            private=PREFIX+'_'+name
            require(private not in sys.modules,'Private native CPU module namespace already occupied')
            spec=importlib.util.spec_from_file_location(private,path)
            module=importlib.util.module_from_spec(spec)
            def imported(target,globals=None,locals=None,fromlist=(),level=0):
                if target=='llava.constants':
                    require(level==0 and tuple(fromlist)==('IMAGE_TOKEN_INDEX',)and 'constants'in modules,
                        'Only original exact native constants import is eligible')
                    return modules['constants']
                require(not(target=='llava'or target.startswith('llava.')),
                    'Native model/package imports are forbidden in CPU input reviewer')
                return builtins.__import__(target,globals,locals,fromlist,level)
            module.__dict__['__builtins__']=dict(vars(builtins),__import__=imported)
            modules[name]=module;sys.modules[private]=module
            exec(compile(raw,str(path),'exec'),module.__dict__)
            require(path.read_bytes()==raw,'Native CPU source changed during execution')
            proof.append(dict(path=str(path),bytes=len(raw),sha256=digest,private_module=private))
        yield modules,dict(schema='exact-native-cpu-input-source-isolation-v1',
            sources=proof,original_source_bodies_unchanged=True,
            model_package_initializer_executed=False,
            only_native_intra_module_import='from llava.constants import IMAGE_TOKEN_INDEX')
        for row in proof:
            require(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],
                'Native CPU source changed during input recomputation')
    finally:
        for name,module in modules.items():
            key=PREFIX+'_'+name
            require(sys.modules.get(key)is module,'Private native input module replaced')
            del sys.modules[key]
