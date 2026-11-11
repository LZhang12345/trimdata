"""Published mathematical operators, with explicit full-pool adaptation boundaries.

No feature approximation, quota top-up, nearest-neighbour replacement for Ward,
or fabricated values are permitted here. Runtime/source conformance is required
before output can receive an independent method review.
"""
import ast
import logging
import math
from pathlib import Path
import numpy as np
from experiments.root_native_campaign_20261023 import common as c
from . import pins


def author_functions(path, names, namespace):
    """Execute only the unchanged, pinned top-level function ASTs requested."""
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    c.require({n.name for n in nodes} == set(names), 'Pinned author function absent')
    code = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    env = dict(namespace)
    exec(compile(code, str(path), 'exec'), env)
    return {name: env[name] for name in names}


def datatailor_values(states, verify_author=False):
    from experiments.copilot7_published_foundation_20261022.datatailor_v2 import native_values, author_values
    actual = native_values(states)
    if not verify_author:return actual
    expected = author_values(pins.TAILOR_MODEL, states)
    import torch
    for name in ('informativeness', 'tau', 'hidden', 'length'):
        a, b = actual[name], expected[name]
        if isinstance(a, torch.Tensor): c.require(torch.equal(a,b), 'Pinned SVD kernel parity: '+name)
        else: c.require(a==b, 'Pinned SVD kernel parity: '+name)
    return actual


def source_blocks(indices):
    from experiments.copilot7_published_foundation_20261022.datatailor_v1 import source_blocks as native
    c.require(indices == sorted(set(indices)), 'Original source positions must remain unique and ordered')
    return native(indices)


def datatailor_ward_block(hidden):
    """The author's float64 cdist -> condensed -> nn_chain(method5) -> fcluster.

    A one-record or degenerate source is rejected: the original code cannot
    establish its final linkage distance. Do not silently invent a cluster.
    """
    import torch
    from scipy.cluster import _hierarchy
    from scipy.cluster.hierarchy import fcluster
    c.require(hidden.ndim==2 and hidden.shape[1]==4096 and len(hidden)>=2,
              'Native Ward requires at least two genuine 4096D source records')
    c.require(bool(torch.isfinite(hidden).all()), 'Nonfinite actual hidden state')
    # Preserve the author list -> float64 tensor operation order.
    weight=torch.tensor([row.double().tolist() for row in hidden],dtype=torch.float64)
    f=author_functions(pins.TAILOR_ROOT/'make_clustering.py',
        ('extract_upper_triangle','linkage'),dict(np=np,_hierarchy=_hierarchy))
    dist=torch.cdist(weight,weight,p=2)
    condensed=f['extract_upper_triangle'](dist.cpu().numpy())
    z=f['linkage'](condensed,[row.double() for row in hidden],method='ward')
    return torch.tensor(fcluster(z,t=z[-1][2]*.1,criterion='distance'))


def datatailor_relationships(hidden, informativeness, labels, indices, on_cluster=None,
                            already=None):
    """Author FP32 relation arithmetic, with durable per-cluster callback seams.

    Original contiguous source blocks are intersected with the frozen eligible
    pool. All means, cosine calculations, exp summation and cdist retain author
    order/dtypes. No regularizer or epsilon is inserted for invalid denominators.
    """
    import torch
    n=len(indices)
    c.require(hidden.device.type=='cpu' and hidden.dtype==torch.float32 and hidden.shape==(n,4096), 'Saved native FP32 4096D hidden matrix required')
    c.require(informativeness.dtype==torch.float32 and informativeness.shape==(n,), 'Saved native FP32 informativeness required')
    c.require(labels.shape==(n,) and int(labels.min())==1,'Author labels start at1')
    count=int(labels.max());zero=labels-1
    members=[[] for _ in range(count)];avg=[torch.zeros(4096) for _ in range(count)]
    global_avg=torch.zeros(4096)
    blocks=source_blocks(indices);belongs={i:g for g,b in enumerate(blocks) for i in b}
    for k in range(n):
        group=int(zero[k]);members[group].append(k);avg[group]+=hidden[k];global_avg+=hidden[k]
    global_avg/=n
    for group in range(count):
        c.require(members[group], 'Noncontiguous/missing author cluster labels')
        for k in range(4096):avg[group][k]=avg[group][k]/len(members[group])
        c.require(len({belongs[p] for p in members[group]})==1,'A Ward cluster crossed a source boundary')
    represent=torch.zeros(n);unique=torch.zeros(n);done=already or {}
    for group,positions in enumerate(members):
        if group in done:
            r,u=done[group]
            c.require(r.shape==u.shape==(len(positions),),'Saved cluster relation scope differs')
        else:
            total=0.;denominator=0.
            for other in range(count):
                if group!=other and belongs[positions[0]]==belongs[members[other][0]]:
                    cosine=torch.dot(avg[group],avg[other])/torch.norm(avg[group])/torch.norm(avg[other])
                    total+=math.exp(cosine);denominator+=1.
            c.require(denominator>0 and math.isfinite(total),'Author source has no other cluster or nonfinite cosine; diagnose, do not invent a score')
            source_rep=torch.tensor(total/denominator,dtype=torch.float32)
            r=source_rep*informativeness[positions]
            # Same tolist -> default FP32 constructor as the author relation file.
            tensor=torch.tensor([hidden[p].tolist() for p in positions])
            distances=torch.cdist(tensor,tensor,p=2)
            scores=torch.tensor([informativeness[p] for p in positions])
            average=torch.sum(distances)/len(positions)
            c.require(bool(torch.isfinite(average)) and float(average)>0,
                      'Author uniqueness denominator is zero/nonfinite; no epsilon substitution')
            u=torch.zeros(len(positions))
            for local in range(len(positions)):u[local]=torch.sum(distances[local]*scores)/average
            c.require(bool(torch.isfinite(r).all() and torch.isfinite(u).all()),'Nonfinite native relation')
            if on_cluster is not None:on_cluster(group,positions,r,u)
        represent[positions]=r;unique[positions]=u
    return unique,represent


def datatailor_select(records,indices,informativeness,tau,uniqueness,representativeness,fraction):
    from experiments.copilot7_published_foundation_20261022.datatailor_v2 import select
    # This reviewed kernel preserves the original FP32 quota operation order,
    # including twelve independent integer floors. It never tops up to nominalK.
    return select(records,indices,informativeness,tau,uniqueness,representativeness,float(fraction))


def coincide_transfer(centroids):
    from sklearn.metrics.pairwise import cosine_similarity
    c.require(centroids.ndim==2 and centroids.shape[1]==20480,'Actual five-layer TinyLLaVA20480D centroids required')
    language=centroids.reshape(-1,5,4096)[:,:,2048:].reshape(-1,5*2048)
    similarity=cosine_similarity(language,language)
    mask=similarity>.9;similarity[mask]=0
    result=similarity.sum(axis=-1)/(~mask).sum(axis=-1)
    c.require(np.isfinite(result).all(),'Native transferability is nonfinite; no fabricated zero')
    return result


def coincide_cluster(embeddings, output):
    """Call the exact pinned FAISS implementation,10000/50/spherical/seed1234."""
    import torch,faiss,time,os,pickle,pathlib
    from typing import Union
    c.require(torch.cuda.is_available(),'Pinned author clustering/saving requires an actual CUDA GPU')
    c.require(embeddings.ndim==2 and embeddings.shape[1]==20480 and len(embeddings)>=10000,
              'No fixture/small-pool replacement of the native10000-cluster baseline')
    c.require(np.isfinite(embeddings).all(),'Native embeddings must have finite author-cleaned values')
    namespace=dict(torch=torch,faiss=faiss,time=time,np=np,logging=logging,os=os,pickle=pickle,pathlib=pathlib,Union=Union)
    f=author_functions(pins.COIN_SCORE/'clustering.py',('faiss_index_to_gpu','compute_centroids'),namespace)
    # Both function globals must share the namespace created by author_functions.
    return f['compute_centroids'](embeddings,ncentroids=10000,niter=50,seed=1234,
        Kmeans_with_cos_dist=True,save_folder=str(output),logger=logging.getLogger('native-coincide'),verbose=True)


def coincide_select(embeddings, clusters, transfer, fraction):
    """Pinned CUDA float64 Gaussian/MMD and sequential native quota allocation."""
    import torch
    c.require(torch.cuda.is_available(),'The pinned greedy_mmd_selection explicitly uses CUDA')
    c.require(embeddings.shape==(len(clusters),20480),'Wrong native embedding scope')
    values=np.unique(clusters)
    c.require(np.array_equal(values,np.arange(len(values))),'Pinned author cluster-ID array must be dense from0')
    c.require(transfer.shape==(len(values),) and np.isfinite(transfer).all(),'Incomplete/nonfinite native transfer')
    f=author_functions(pins.COIN_SCORE/'cluster_wise_prune.py',('gaussian_kernel','greedy_mmd_selection'),dict(np=np,torch=torch))
    kernels=[];density=np.zeros(len(values),dtype='float64')
    for cluster in values:
        indices=np.where(clusters==cluster)[0]
        points=torch.from_numpy(embeddings[indices]).cuda().to(torch.float64)
        kernel=f['gaussian_kernel'](points,points,1.)
        density[cluster]=kernel.mean().item();kernels.append(kernel)
    c.require(np.isfinite(density).all() and (density>0).all(),'Native cluster density invalid')
    score=(1/density)*transfer
    # Preserve the author's direct exp and sorting, rather than silently changing
    # overflow/tie behaviour to a numerically different softmax implementation.
    ratio=np.exp(score/.1)/np.sum(np.exp(score/.1))
    c.require(np.isfinite(ratio).all(),'Pinned native allocation overflow; preserve and diagnose')
    order=np.argsort(ratio)[::-1];ratio=ratio[order];score=score[order]
    remaining=int(float(fraction)*len(embeddings));selected=[];quotas=[]
    for i,cluster in enumerate(order):
        positions=np.where(clusters==cluster)[0];quota=round(remaining*ratio[i]);quotas.append([int(cluster),int(quota)])
        chosen=positions if quota>len(positions) else positions[f['greedy_mmd_selection'](kernels[cluster],quota)]
        selected.append(chosen);remaining-=len(chosen)
        if i+1<len(order):ratio[i+1:]=np.exp(score[i+1:]/.1)/np.sum(np.exp(score[i+1:]/.1))
    positions=np.concatenate(selected)
    c.require(len(positions)==len(set(positions.tolist())) and len(positions)<=int(float(fraction)*len(embeddings)), 'Native MMD selection budget/uniqueness failure')
    return dict(positions=positions.tolist(),count=len(positions),remaining=remaining,quotas=quotas,top_up=False,
        source_order_restored_for_training=True,author_selection_order=positions.tolist())
