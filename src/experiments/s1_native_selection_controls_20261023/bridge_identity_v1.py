"""Path-independent content identity for selections compared across sites.

Bound descriptors embed absolute paths, so equal ACES/Babel selections have
different descriptor hashes. This audit hashes CONTENT only: pool IDs and
records, selected IDs and records, frozen group mapping, and feature-row values
(loss, targets, bucket, embedding bytes). Path strings are never hashed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from experiments.root_native_campaign_v3_20261023 import common as c

SCHEMA = 'native-bridge-content-identity-v1'
FIELDS = ('method', 'fraction', 'selection_seed', 'count', 'record_ids', 'records',
          'pool_record_ids', 'pool_records', 'pool_source_indices', 'groups', 'features')


def _file_content(descriptor):
    c.checked(descriptor)
    return c.sha(descriptor['path'])


def _features(descriptor):
    if descriptor is None:
        return None
    receipt = c.checked(descriptor)
    digest = hashlib.sha256()
    rows = []
    for db in receipt['databases']:
        c.require(c.desc(db['path']) == db, 'Feature database bytes changed')
        con = sqlite3.connect('file:' + str(Path(db['path']).absolute()) + '?mode=ro', uri=True)
        try:
            for ident, loss, targets, bucket, emb in con.execute('SELECT record_id,loss,targets,vision_bucket,embedding FROM features'):
                rows.append((ident, repr(float(loss)), int(targets), bucket, hashlib.sha256(emb).hexdigest() if emb is not None else None))
        finally:
            con.close()
    for row in sorted(rows):
        digest.update(c.canonical(list(row)))
    return dict(rows=len(rows), sha256=digest.hexdigest(), stage=receipt.get('stage'))


def content_identity(selection_desc):
    selected = c.checked(selection_desc)
    pool = c.checked(selected['pool'])
    groups = None
    if selected.get('groups'):
        g = c.checked(selected['groups'])
        groups = dict(status=g['status'], label_kind=g.get('label_kind'), sha256=c.key(sorted(g['groups'].items())))
    return dict(method=selected['method'], fraction=selected['fraction'], selection_seed=selected['selection_seed'],
                count=selected['count'], record_ids=c.key(selected['record_ids']),
                records=_file_content(selected['records_json']),
                pool_record_ids=c.key(pool['record_ids']), pool_records=_file_content(pool['records_json']),
                pool_source_indices=c.key(pool['source_indices']), groups=groups,
                features=_features(selected.get('features')))


def compare(a, b):
    mismatched = [f for f in FIELDS if a.get(f) != b.get(f)]
    return dict(schema=SCHEMA, status='PASS_SAME_CONTENT' if not mismatched else 'FAIL_CONTENT_DIFFERS',
                mismatched_fields=mismatched, left=a, right=b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('identity'); one.add_argument('--selection', required=True); one.add_argument('--output', required=True)
    two = sub.add_parser('compare'); two.add_argument('--left', required=True); two.add_argument('--right', required=True); two.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'identity':
        value = dict(schema=SCHEMA + '-side', identity=content_identity(c.read(args.selection)))
    else:
        value = compare(c.read(args.left)['identity'], c.read(args.right)['identity'])
    print(json.dumps(c.seal(args.output, value)))


if __name__ == '__main__':
    main()
