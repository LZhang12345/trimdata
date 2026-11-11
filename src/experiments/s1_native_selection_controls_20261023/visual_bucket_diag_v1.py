"""Features-only diagnostic of the bound 12-bit visual buckets; centered draft.

The bound feature writer hashes the UNCENTERED pooled CLIP vector with a fixed
N(0,1) projection (default_rng(1234), 12 x D). A large common mean can make the
sign bits nearly constant, collapsing coverage into few buckets. This reads the
stored fp16 embeddings, recomputes the bound buckets, and compares them with
buckets after subtracting the eligible TRAIN image mean (same projection).

GATE_V1 is declared here before any student result exists and reads features
only. It labels the bound variant; it never switches a method on outcomes.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import sqlite3

import numpy as np

from experiments.root_native_campaign_v3_20261023 import common as c

GATE_V1 = dict(
    min_mean_bit_entropy_bits=0.5,
    min_effective_buckets=256,
    max_bucket_share=0.05,
    rule=('BOUND_UNCENTERED_ADEQUATE if all three thresholds hold for the bound buckets; '
          'otherwise BOUND_UNCENTERED_DEGENERATE. If degenerate and centered buckets satisfy all three, '
          'propose a separately versioned centered-bucket variant BEFORE release; never chosen from training outcomes.'),
    reads='stored native features only (no student checkpoints or evaluation outputs)')


def bucket_stats(buckets, bits):
    counts = Counter(buckets)
    n = len(buckets)
    p = np.array(list(counts.values()), dtype=np.float64) / n
    entropy = float(-(p * np.log(p)).sum())
    per_bit = []
    for j in range(12):
        q = sum((b >> j) & 1 for b in buckets) / n
        per_bit.append(0.0 if q in (0.0, 1.0) else float(-(q * math.log2(q) + (1 - q) * math.log2(1 - q))))
    return dict(records=n, occupied_buckets=len(counts), effective_buckets=math.exp(entropy),
                max_bucket_share=max(counts.values()) / n, per_bit_positive_fraction=[
                    sum((b >> j) & 1 for b in buckets) / n for j in range(12)],
                per_bit_entropy_bits=per_bit, mean_bit_entropy_bits=sum(per_bit) / 12)


def passes(stats):
    return (stats['mean_bit_entropy_bits'] >= GATE_V1['min_mean_bit_entropy_bits']
            and stats['effective_buckets'] >= GATE_V1['min_effective_buckets']
            and stats['max_bucket_share'] <= GATE_V1['max_bucket_share'])


def hash_bits(matrix, projection):
    signs = (matrix @ projection.T) >= 0
    return [int(sum(int(v) << j for j, v in enumerate(row))) for row in signs]


def diagnose(ids, stored, embeddings, groups=None):
    c.require(len(ids) == len(stored) == len(embeddings) > 0, 'Aligned image-record features required')
    matrix = np.stack(embeddings).astype(np.float32)
    rng = np.random.default_rng(1234)
    projection = rng.standard_normal((12, matrix.shape[1]), dtype=np.float32)
    recomputed = hash_bits(matrix, projection)
    agreement = sum(a == b for a, b in zip(recomputed, stored)) / len(ids)
    centered = hash_bits(matrix - matrix.mean(axis=0, keepdims=True), projection)
    bound = bucket_stats(list(stored), 12)
    cent = bucket_stats(centered, 12)
    mean_norm = float(np.linalg.norm(matrix.mean(axis=0)))
    resid_norm = float(np.linalg.norm(matrix - matrix.mean(axis=0), axis=1).mean())
    decision = ('BOUND_UNCENTERED_ADEQUATE' if passes(bound) else
                'BOUND_UNCENTERED_DEGENERATE_PROPOSE_CENTERED_VARIANT' if passes(cent) else
                'BOUND_UNCENTERED_DEGENERATE_CENTERED_ALSO_FAILS_REVIEW_FEATURE')
    out = dict(gate=GATE_V1, decision=decision, bound_bucket_stats=bound, centered_bucket_stats=cent,
               fp16_recompute_agreement_with_stored_bucket=agreement,
               mean_vector_norm=mean_norm, mean_residual_norm=resid_norm,
               centered_buckets={i: b for i, b in zip(ids, centered)})
    if groups is not None:
        per = {}
        for g in sorted(set(groups[i] for i in ids)):
            members = [k for k, i in enumerate(ids) if groups[i] == g]
            per[g] = dict(bound=bucket_stats([stored[k] for k in members], 12)['effective_buckets'],
                          centered=bucket_stats([centered[k] for k in members], 12)['effective_buckets'],
                          records=len(members))
        out['per_source_effective_buckets'] = per
    return out


def read_features(receipt_desc):
    receipt = c.checked(receipt_desc)
    ids, stored, embeddings = [], [], []
    for db in receipt['databases']:
        c.require(c.desc(db['path']) == db, 'Feature database bytes changed')
        con = sqlite3.connect('file:' + str(Path(db['path']).absolute()) + '?mode=ro', uri=True)
        try:
            for ident, bucket, emb in con.execute('SELECT record_id,vision_bucket,embedding FROM features WHERE embedding IS NOT NULL ORDER BY record_id'):
                ids.append(ident); stored.append(int(bucket)); embeddings.append(np.frombuffer(emb, dtype='<f2'))
        finally:
            con.close()
    return ids, stored, embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True); parser.add_argument('--groups')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ids, stored, emb = read_features(c.read(args.features))
    groups = c.checked(c.read(args.groups))['groups'] if args.groups else None
    result = diagnose(ids, stored, emb, groups)
    result.update(schema='native-visual-bucket-diagnostic-v1', features=c.read(args.features),
                  producer=dict(path='experiments/s1_native_selection_controls_20261023/visual_bucket_diag_v1.py', sha256=c.sha(Path(__file__))))
    print(json.dumps({k: v for k, v in c.seal(args.output, result).items()}))


if __name__ == '__main__':
    main()
