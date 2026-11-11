"""Stage the new v3 evaluation source root (login node, stdlib, python3.9 compatible).

<target> = exact extraction of the frozen COINCIDE v2 archive (every file checked
against EVALUATION_EXECUTABLE_HANDOFF_V2.json) + the qualified v7 basis overlay
(basis files missing from the archive and the 4 Root core files the v7 basis
supersedes, each checked against basis['sources']) + this package (tests
excluded). Built in a sibling temporary directory and renamed into place only
after the whole tree verifies; an existing target is never touched.
Usage (from the directory containing this package):
  python3 -B -m claude_coincide_eval_v3_20261030.stage --archive <EVALUATION_EXECUTABLE_SOURCE_V2.tar.gz>
"""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import uuid
from . import common_v3 as c


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def _safe(name):
    pure = PurePosixPath(name)
    c.require(not pure.is_absolute() and '..' not in pure.parts and str(pure) == name and name, 'Unsafe member: ' + name)
    return pure


def tree_files(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        c.require(not path.is_symlink(), 'Symlink in staged root: ' + str(path))
        if path.is_file():
            result[str(path.relative_to(root))] = c.sha(path)
        else:
            c.require(path.is_dir(), 'Foreign file type in staged root: ' + str(path))
    return result


def stage(archive, target, package_dir=None, handoff_path=None, basis_path=None, basis_source_root=None,
          receipt_dir=None):
    package_dir = Path(package_dir) if package_dir is not None else c.HERE
    handoff_ref = c.desc(handoff_path or c.HANDOFF['path'])
    basis_ref = c.desc(basis_path or c.BASIS['path'])
    c.require(c.same_bytes(handoff_ref, c.HANDOFF) and c.same_bytes(basis_ref, c.BASIS), 'Pinned handoff/basis bytes differ')
    handoff, basis = c.read(handoff_ref['path']), c.read(basis_ref['path'])
    archive_ref = c.desc(archive)
    c.require(c.same_bytes(archive_ref, c.ARCHIVE) and c.same_bytes(archive_ref, handoff['source_archive']),
              'Archive is not the frozen v2 evaluation source')
    tree, superseded = c.frozen_tree(handoff['sources'], basis['sources'])
    target = Path(target).absolute()
    c.require(not target.exists() and not target.is_symlink() and target.parent.is_dir(), 'Target must be new')
    temporary = target.parent / ('.' + target.name + '.staging-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            c.require(all(m.isfile() or m.isdir() for m in members), 'Archive holds links/devices')
            files = [m for m in members if m.isfile()]
            names = [m.name for m in files]
            c.require(len(names) == len(set(names)) and set(names) == set(handoff['sources']),
                      'Archive members differ from the handoff source map')
            for member in files:
                data = tar.extractfile(member).read()
                c.require(hashlib.sha256(data).hexdigest() == handoff['sources'][member.name],
                          'Archive member bytes differ from handoff: ' + member.name)
                _write(temporary / _safe(member.name), data)
        source_root = Path(basis_source_root or basis['source_root'])
        overlay = []
        for name, digest in sorted(basis['sources'].items()):
            if handoff['sources'].get(name) == digest:
                continue
            data = (source_root / _safe(name)).read_bytes()
            c.require(hashlib.sha256(data).hexdigest() == digest, 'Basis source root drifted: ' + name)
            destination = temporary / name
            if destination.exists():
                c.require(name in superseded, 'Undeclared overwrite: ' + name)
                destination.unlink()
            _write(destination, data)
            overlay.append(name)
        own = sorted(p for p in package_dir.iterdir() if p.suffix in ('.py', '.sbatch') and not p.name.startswith('test_'))
        for path in own:
            _write(temporary / c.PACKAGE / path.name, path.read_bytes())
        expected = dict(tree)
        expected.update(c.package_sources(temporary))
        actual = tree_files(temporary)
        c.require(actual == expected, 'Staged tree differs from frozen archive + basis overlay + package')
        for name in actual:
            os.chmod(temporary / name, 0o444)
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    c.require(tree_files(target) == expected, 'Staged tree changed during rename')
    receipt = dict(status='STAGED_COINCIDE_V3_EVALUATION_SOURCE_ROOT_NOT_EXECUTION', target=str(target),
        handoff=handoff_ref, basis=basis_ref, archive=archive_ref, archive_files=len(handoff['sources']),
        basis_overlay_files=overlay, superseded_archive_files=superseded, basis_source_root=str(source_root),
        package_sources=c.package_sources(target), total_files=len(expected), tree_sha256=hashlib.sha256(
            c.canonical(expected)).hexdigest(), UTC=c.now())
    receipt_dir = Path(receipt_dir) if receipt_dir is not None else c.EVAL_RECORDS
    return c.seal(receipt_dir / ('STAGE_RECEIPT_' + target.name + '.json'), receipt)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--archive', required=True)
    a.add_argument('--target', default=str(c.SOURCE_ROOT))
    a.add_argument('--handoff'); a.add_argument('--basis'); a.add_argument('--basis-source-root')
    a.add_argument('--receipt-dir')
    v = a.parse_args()
    print(c.canonical(stage(v.archive, v.target, None, v.handoff, v.basis, v.basis_source_root, v.receipt_dir)).decode())


if __name__ == '__main__':
    main()
