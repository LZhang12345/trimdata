"""Use the paid complete matrix once; preserve native operator outputs and failures."""
import argparse
import fcntl
import importlib
import os
from pathlib import Path
import signal
import stat
import sys
import time

from . import protocol as p


N = 660763
D = 20480
CLUSTERS = 10000


def directory(path):
    path = Path(path)
    p.m.require(path.is_absolute() and path.resolve() == path, "Canonical operator output required")
    if path.exists():
        p.m.require(path.is_dir() and path.stat().st_uid == os.getuid(), "Exact project-owned output directory required")
        for entry in path.rglob("*"):
            p.m.require(not entry.is_symlink(), "Aliased native output refused")
            if entry.is_file():
                value = entry.stat()
                p.m.require(stat.S_ISREG(value.st_mode) and value.st_nlink == 1,
                            "Native artifacts must be regular single-link files")
    return path


def save_array(path, value):
    from experiments.root_native_baselines_20261023.select import save_array as native_save
    directory(Path(path).parent)
    return native_save(path, value)


def runtime(spec):
    p.m.require(str(Path(sys.executable).absolute()) == p.PYTHON,
                "Use the already-qualified isolated FAISS Python, not the feature/training environment")
    environment = p.m.checked(spec["environment_configuration"])
    p.m.require(environment["owner"] == "Root"
        and environment["environment"] == str(Path(p.PYTHON).parent.parent)
        and all(spec["sources"].get(name) == digest for name,digest in environment["sources"].items()),
        "Exact completed FAISS environment configuration/source provenance required")
    probe = p.m.checked(spec["environment_probe"])
    p.m.require(probe["returncode"] == 0 and probe["timed_out"] is False
        and probe["argv"][0] == p.PYTHON, "Existing actual FAISS software/GPU fixture receipt required")
    expected = {"numpy":"1.26.4", "scipy":"1.15.3", "sklearn":"1.2.2", "torch":"2.2.0", "faiss":"1.8.0"}
    observed = {}
    root = Path(p.PYTHON).parent.parent
    for name, version in expected.items():
        module = importlib.import_module(name)
        actual = module.__version__
        p.m.require(actual.split("+")[0] == version and Path(module.__file__).absolute().is_relative_to(root),
                    "Actual qualified operator module/version/origin differs: " + name)
        observed[name] = dict(version=actual, source=p.m.reference(Path(module.__file__).absolute()))
    import torch
    import faiss
    p.m.require(torch.version.cuda == "12.1" and torch.cuda.is_available()
        and torch.cuda.device_count() == faiss.get_num_gpus() == 1
        and "L40S" in torch.cuda.get_device_name(0), "One actual CUDA12.1 L40S and GPU-enabled FAISS required")
    torch.set_num_threads(8)
    faiss.omp_set_num_threads(8)
    return dict(modules=observed, GPU=torch.cuda.get_device_name(0), python=sys.executable,
        torch_cuda=torch.version.cuda, torch_TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        software_fixture_reused_not_reexecuted=spec["environment_probe"],
        new_models_or_forwards_or_downloads=0)


def paid_inputs(spec):
    import numpy as np
    receipt = p.m.checked(spec["matrix_receipt"])
    p.m.require(receipt["status"] == "PASS_COMPLETE_NATIVE_COINCIDE_INPUT_BYTES_NOT_OPERATOR_QUALIFICATION"
        and receipt["count"] == N and receipt["method"] == "coincide"
        and receipt["measurement_identity"] == p.MEASUREMENT
        and receipt["owner"]["pool"] == spec["pool"]
        and receipt["owner"]["eligibility"] == spec["eligibility"]
        and receipt["original_qualified_rows_reused"] == 16 and receipt["production_unique_rows"] == N - 16,
        "Exact complete paid660763 matrix function/pool/primary-provider identity required")
    hidden = receipt["hidden"]
    path = Path(hidden["path"])
    p.m.require(path == p.MATRIX_ROOT / "hidden.npy" and p.m.reference(path) == hidden,
                "Paid matrix bytes/path changed")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    p.m.require(matrix.shape == (N, D) and matrix.dtype == np.dtype("<f2"), "Native complete FP16x20480 matrix required")
    pool, eligibility = p.m.checked(spec["pool"]), p.m.checked(spec["eligibility"])
    ids = pool["record_ids"]
    p.m.require(pool["status"] == "PASS_POOL" and pool["count"] == len(ids) == len(set(ids)) == N
        and eligibility["status"] == "PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY"
        and eligibility["pool"] == spec["pool"]
        and p.key(ids) == "076b7214072a07cbcb1cb8653e8f6760c13dd4cc6cd1cd9eecafc86db9eb6e17",
        "Exact canonical protected record order required, never a new construction or availability denominator")
    return matrix, ids, hidden


def check_file(reference, root, expected=None):
    path = Path(reference["path"])
    p.m.require(root in path.parents and (expected is None or path == expected)
                and p.m.reference(path) == reference,
                "Accepted operator artifact changed or escaped output")
    return path


def validate_cluster_receipt(receipt, output, owner):
    import numpy as np
    directory(output)
    p.m.require(receipt["owner"] == owner
        and receipt["status"] == "ACTUAL_NATIVE_FAISS10000_50_SPHERICAL_GEOMETRY_PENDING_ACTUAL_REVIEW"
        and receipt["no_CPU_fallback"] is True
        and receipt["nearest_index_order_is_canonical_pool_order"] is True,
        "Accepted native clustering status or owner differs")
    names = {"kmeans_index.pickle","kmeans_centroids.npy","dist_to_cent.npy","nearest_cent.npy"}
    p.m.require(set(receipt["files"]) == names, "Exact native clustering artifact set required")
    for name, reference in receipt["files"].items():
        check_file(reference,output,output/"native-faiss"/name)
    centers = np.load(output/"native-faiss/kmeans_centroids.npy",mmap_mode="r",allow_pickle=False)
    labels = np.load(output/"native-faiss/nearest_cent.npy",mmap_mode="r",allow_pickle=False)
    distances = np.load(output/"native-faiss/dist_to_cent.npy",mmap_mode="r",allow_pickle=False)
    p.m.require(centers.shape == (CLUSTERS,D) and centers.dtype == np.dtype("<f4")
        and labels.shape == distances.shape == (N,) and labels.dtype == np.dtype("<i8")
        and distances.dtype == np.dtype("<f4") and np.isfinite(centers).all()
        and np.isfinite(distances).all() and np.array_equal(np.unique(labels),np.arange(CLUSTERS)),
        "Exact native centroid/label/distance dimensions, dtypes, coverage and finiteness required")
    counts = np.bincount(labels,minlength=CLUSTERS)
    p.m.require(receipt["cluster_sizes"] == counts.tolist()
        and receipt["cluster_size_min"] == int(counts.min()) > 0
        and receipt["cluster_size_max"] == int(counts.max()),
        "Accepted cluster counts differ from the actual native labels")
    validate_gpu_metadata(receipt["actual_GPU_index"])
    return receipt


def validate_transfer_receipt(receipt, output, owner):
    import numpy as np
    directory(output)
    p.m.require(receipt["owner"] == owner
        and receipt["status"] == "ACTUAL_NATIVE_TRANSFERABILITY_PENDING_ACTUAL_REVIEW"
        and receipt["cluster"] == p.m.reference(output/"CLUSTER_COMPLETE.json"),
        "Accepted transferability status/owner/cluster dependency differs")
    validate_cluster_receipt(p.m.checked(receipt["cluster"]),output,owner)
    path = check_file(receipt["transfer"],output,output/"transfer.npy")
    values = np.load(path,mmap_mode="r",allow_pickle=False)
    p.m.require(values.shape == (CLUSTERS,) and values.dtype == np.dtype("<f8") and np.isfinite(values).all(),
                "Accepted native transferability dtype/shape/finiteness differs")
    return receipt


def validate_gpu_metadata(actual):
    import faiss
    name = actual["concrete_index_class"]
    gpu_base = getattr(faiss,"GpuIndex",None)
    recorded_type = getattr(faiss,name,None) if isinstance(name,str) else None
    p.m.require(isinstance(gpu_base,type) and isinstance(recorded_type,type)
        and issubclass(recorded_type,gpu_base)
        and actual["dimension"] == D and actual["centroids"] == CLUSTERS
        and actual["niter"] == 50 and actual["spherical"] is True and actual["seed"] == 1234,
        "Accepted native GPU class/dimension/parameter proof differs")
    return actual


def gpu_index_metadata(kmeans):
    import faiss
    index = kmeans.index
    if hasattr(index,"count") and hasattr(index,"at"):
        index = faiss.downcast_index(index.at(0))
    p.m.require(isinstance(index,faiss.GpuIndex) and index.d == D and index.ntotal == CLUSTERS
        and kmeans.cp.niter == 50 and kmeans.cp.spherical is True and kmeans.cp.seed == 1234,
        "Actual full-dimensional native FAISS must use the exact GPU index and published parameters")
    return validate_gpu_metadata(dict(concrete_index_class=type(index).__name__,dimension=index.d,
        centroids=index.ntotal,niter=kmeans.cp.niter,spherical=kmeans.cp.spherical,seed=kmeans.cp.seed))


def cluster_stage(spec, matrix, output, owner, stop):
    import numpy as np
    from experiments.root_native_baselines_20261023 import operators
    accepted = output / "CLUSTER_COMPLETE.json"
    if accepted.exists():
        receipt = p.m.checked(p.m.reference(accepted))
        return validate_cluster_receipt(receipt,output,owner)
    p.m.require(not stop(), "No accepted-boundary reserve before native clustering")
    target = output / "native-faiss"
    p.m.require(not target.exists(), "Unaccepted partial FAISS output preserved; explicit reviewed recovery required")
    started = time.monotonic()
    kmeans = operators.coincide_cluster(matrix, target)
    GPU_index = gpu_index_metadata(kmeans)
    names = {"kmeans_index.pickle", "kmeans_centroids.npy", "dist_to_cent.npy", "nearest_cent.npy"}
    p.m.require({path.name for path in target.iterdir()} == names, "Original native clustering artifact set differs")
    directory(target)
    files = {path.name:p.m.reference(path) for path in sorted(target.iterdir())}
    centers = np.load(target / "kmeans_centroids.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(target / "nearest_cent.npy", mmap_mode="r", allow_pickle=False)
    distances = np.load(target / "dist_to_cent.npy", mmap_mode="r", allow_pickle=False)
    p.m.require(centers.shape == (CLUSTERS,D) and centers.dtype == np.dtype("float32")
        and np.isfinite(centers).all() and labels.shape == distances.shape == (N,)
        and labels.dtype == np.dtype("<i8") and distances.dtype == np.dtype("<f4")
        and np.isfinite(distances).all()
        and np.array_equal(np.unique(labels),np.arange(CLUSTERS)),
        "Actual full native10000-cluster geometry is incomplete, nonfinite or has missing cluster IDs")
    counts = np.bincount(labels, minlength=CLUSTERS)
    receipt = dict(owner=owner, files=files, elapsed_seconds=time.monotonic()-started,
        status="ACTUAL_NATIVE_FAISS10000_50_SPHERICAL_GEOMETRY_PENDING_ACTUAL_REVIEW",
        cluster_size_min=int(counts.min()),cluster_size_max=int(counts.max()),
        cluster_sizes=counts.tolist(), actual_GPU_index=GPU_index,
        no_CPU_fallback=True, nearest_index_order_is_canonical_pool_order=True)
    validate_cluster_receipt(receipt,output,owner)
    p.m.seal(accepted,receipt)
    return receipt


def transfer_stage(cluster, output, owner, stop):
    import numpy as np
    from experiments.root_native_baselines_20261023 import operators
    path = output / "TRANSFER_COMPLETE.json"
    if path.exists():
        value = p.m.checked(p.m.reference(path))
        return validate_transfer_receipt(value,output,owner)
    p.m.require(not stop(), "No time to start native transferability; accepted clustering preserved")
    centers = np.load(cluster["files"]["kmeans_centroids.npy"]["path"],mmap_mode="r",allow_pickle=False)
    transfer = operators.coincide_transfer(centers)
    p.m.require(transfer.shape == (CLUSTERS,) and np.isfinite(transfer).all(),
                "Native language-half transferability is undefined; do not impute zero")
    receipt = dict(owner=owner, cluster=p.m.reference(output / "CLUSTER_COMPLETE.json"),
        transfer=save_array(output / "transfer.npy",transfer),
        status="ACTUAL_NATIVE_TRANSFERABILITY_PENDING_ACTUAL_REVIEW")
    validate_transfer_receipt(receipt,output,owner)
    p.m.seal(path,receipt)
    return receipt


def memory_profile(cluster):
    import torch
    counts = cluster["cluster_sizes"]
    free, total = torch.cuda.mem_get_info()
    return dict(largest_cluster_records=max(counts), smallest_cluster_records=min(counts),
        all_FP64_kernel_value_bytes_lower_bound=sum(value*value*8 for value in counts),
        largest_FP64_point_bytes_lower_bound=max(counts)*D*8,
        largest_FP64_kernel_bytes_lower_bound=max(counts)**2*8,
        necessary_peak_buffer_bytes_lower_bound=max(sum(value*value*8 for value in counts),max(counts)*D*8),
        combined_simultaneous_sum_not_claimed=True,
        actual_GPU_free_bytes=free,actual_GPU_total_bytes=total,
        is_peak_upper_bound=False,actual_full_MMD_completed=False)


def validate_selection(result, fraction, ids):
    positions = result["positions"]
    requested = p.BUDGETS[fraction]
    p.m.require(isinstance(positions,list) and all(type(value) is int and 0 <= value < len(ids) for value in positions)
        and type(result["count"]) is int and type(result["remaining"]) is int
        and 0 < len(positions) == len(set(positions)) == result["count"] <= requested
        and result["remaining"] == requested-len(positions) and result["top_up"] is False,
        "Original MMD result violates unique whole-conversation budget or is unusably empty; no top-up")
    return sorted(positions)


def validate_budget_receipt(value, spec, ids, output, owner, fraction):
    directory(output)
    target = output/("budget-"+fraction)
    p.m.require(value["owner"] == owner and value["fraction"] == fraction
        and value["status"] == "ACTUAL_NATIVE_MMD_CANDIDATE_PENDING_INDEPENDENT_REVIEW",
        "Accepted native budget status or identity differs")
    raw_path = check_file(value["operator_result"],output,target/"AUTHOR_OPERATOR_RESULT.json")
    candidate_path = check_file(value["candidate"],output,target/"CANDIDATE.json")
    raw = p.m.checked(p.m.reference(raw_path))
    positions = validate_selection(raw,fraction,ids)
    quotas = raw.get("quotas")
    p.m.require(isinstance(quotas,list) and len(quotas) == CLUSTERS
        and all(isinstance(row,list) and len(row) == 2 and type(row[0]) is int and type(row[1]) is int
                and 0 <= row[0] < CLUSTERS and row[1] >= 0 for row in quotas)
        and {row[0] for row in quotas} == set(range(CLUSTERS))
        and raw.get("source_order_restored_for_training") is True
        and raw.get("author_selection_order") == raw["positions"],
        "Exact native raw MMD quota/position custody differs")
    requested = p.BUDGETS[fraction]
    p.m.require(value["requested_K"] == requested and value["actual_K"] == raw["count"] == len(positions),
                "Accepted native budget count disagrees with raw operator output")
    validate_transfer_receipt(p.m.checked(p.m.reference(output/"TRANSFER_COMPLETE.json")),output,owner)
    import numpy as np
    labels = np.load(output/"native-faiss/nearest_cent.npy",mmap_mode="r",allow_pickle=False)
    populations = np.bincount(labels,minlength=CLUSTERS)
    chosen_counts = np.bincount(labels[np.asarray(raw["positions"],dtype=np.int64)],minlength=CLUSTERS)
    p.m.require(all(int(chosen_counts[label]) == min(quota,int(populations[label])) for label,quota in quotas),
                "Accepted raw selected positions disagree with native per-cluster quotas")
    candidate = p.m.checked(p.m.reference(candidate_path))
    chosen = [ids[position] for position in positions]
    p.m.require(candidate["schema"] == "s7-native-COINCIDE-index-candidate-v1"
        and candidate["status"] == "PENDING_INDEPENDENT_AUTHOR_OPERATOR_OUTPUT_REVIEW_NOT_STUDENT_READY"
        and candidate["method"] == "coincide" and candidate["fraction"] == fraction
        and candidate["selection_seed"] == 1234 and candidate["pool"] == spec["pool"]
        and candidate["source_configuration"] == owner["configuration"]
        and candidate["operator_sources"] == spec["operator_sources"]
        and candidate["requested_K"] == requested and candidate["actual_K"] == len(positions)
        and candidate["record_ids"] == chosen and candidate["positions"] == positions
        and candidate["canonical_record_ids_sha256"] == p.key(ids)
        and candidate["selected_record_ids_sha256"] == p.key(chosen)
        and candidate["operator_result"] == value["operator_result"]
        and candidate["cluster"] == p.m.reference(output/"CLUSTER_COMPLETE.json")
        and candidate["transfer"] == p.m.reference(output/"TRANSFER_COMPLETE.json")
        and candidate["author_floor_not_topped_up"] is True
        and candidate["matched_actual_K_uniform_required"] == (len(positions) != requested)
        and candidate["original_whole_records_not_yet_materialized"] is True
        and candidate["records_json"] is None and candidate["student_READY"] is False,
        "Accepted candidate pool/configuration/geometry/raw/IDs/count graph differs")
    return value


def selection_stage(spec, matrix, ids, cluster, transfer, output, owner, fraction, stop):
    import numpy as np
    import torch
    from experiments.root_native_baselines_20261023 import operators
    target = output / ("budget-" + fraction)
    accepted = target / "COMPLETE.json"
    if accepted.exists():
        value = p.m.checked(p.m.reference(accepted))
        return validate_budget_receipt(value,spec,ids,output,owner,fraction)
    p.m.require(p.m.checked(p.m.reference(output/"CLUSTER_COMPLETE.json")) == cluster
        and p.m.checked(p.m.reference(output/"TRANSFER_COMPLETE.json")) == transfer,
        "Current native geometry arguments differ from accepted owned stage receipts")
    validate_transfer_receipt(transfer,output,owner)
    p.m.require(not stop() and not target.exists(),
                "No time or unaccepted partial budget output; preserve geometry and diagnose, no blind recomputation")
    target.mkdir()
    started = time.monotonic()
    labels = np.load(cluster["files"]["nearest_cent.npy"]["path"],mmap_mode="r",allow_pickle=False)
    scores = np.load(transfer["transfer"]["path"],mmap_mode="r",allow_pickle=False)
    torch.cuda.reset_peak_memory_stats()
    result = operators.coincide_select(matrix,labels,scores,fraction)
    raw = p.m.seal(target / "AUTHOR_OPERATOR_RESULT.json",result)
    positions = validate_selection(result,fraction,ids)
    candidate = dict(schema="s7-native-COINCIDE-index-candidate-v1",
        status="PENDING_INDEPENDENT_AUTHOR_OPERATOR_OUTPUT_REVIEW_NOT_STUDENT_READY",
        method="coincide",fraction=fraction,selection_seed=1234,pool=spec["pool"],
        requested_K=p.BUDGETS[fraction],actual_K=len(positions),
        record_ids=[ids[position] for position in positions],positions=positions,
        canonical_record_ids_sha256=p.key(ids),selected_record_ids_sha256=p.key([ids[position] for position in positions]),
        operator_result=raw,cluster=p.m.reference(output / "CLUSTER_COMPLETE.json"),
        transfer=p.m.reference(output / "TRANSFER_COMPLETE.json"),
        source_configuration=owner["configuration"],operator_sources=spec["operator_sources"],
        author_floor_not_topped_up=True,
        matched_actual_K_uniform_required=len(positions)!=p.BUDGETS[fraction],
        original_whole_records_not_yet_materialized=True,records_json=None,student_READY=False)
    candidate_ref = p.m.seal(target / "CANDIDATE.json",candidate)
    value = dict(owner=owner,fraction=fraction,requested_K=p.BUDGETS[fraction],actual_K=len(positions),
        candidate=candidate_ref,operator_result=raw,elapsed_seconds=time.monotonic()-started,
        Torch_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        Torch_GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        FAISS_memory_not_in_Torch_counters=True,status="ACTUAL_NATIVE_MMD_CANDIDATE_PENDING_INDEPENDENT_REVIEW")
    validate_budget_receipt(value,spec,ids,output,owner,fraction)
    p.m.seal(accepted,value)
    return value


def run(binding_path):
    p.actor()
    p.m.require(str(Path(binding_path)) == str(p.CONTROL / "registration/BINDING.json"),
                "Exact delegated binding path required")
    binding = p.m.checked(p.m.reference(binding_path))
    p.m.require(binding["owner"]=="Session1" and binding["function_owner"]=="Session7"
        and binding["job_id"]==os.environ.get("SLURM_JOB_ID")
        and not os.environ.get("SLURM_ARRAY_JOB_ID"), "Exact scalar delegated operator process required")
    began=time.monotonic()
    report=p.CONTROL/"registration"/("attempt-"+binding["job_id"])
    p.m.require(not report.exists(),"Immutable attempt identity already used")
    p.m.seal(report/"PREFLIGHT.json",dict(job_id=binding["job_id"],before_DATA_read=True,models=0,forwards=0))
    phase="pre_DATA"
    stopped=[False]
    signal.signal(signal.SIGUSR1,lambda *_:stopped.__setitem__(0,True))
    def terminated(signum,frame):
        raise RuntimeError("Operator received signal "+str(signum))
    signal.signal(signal.SIGTERM,terminated)
    try:
        spec=p.validate(binding["configuration"],binding["review"])
        actual=p.observe(binding,spec,running=True)
        p.m.seal(report/"ACTUAL.json",dict(actual=actual,configuration=binding["configuration"],
            review=binding["review"],storage=binding["storage_reservation"],before_DATA_read=True))
        hours,minutes,seconds=map(int,actual["fields"]["RunTime"].split(":"))
        deadline=time.time()+360*60-(hours*3600+minutes*60+seconds)-180
        stop=lambda:stopped[0] or time.time()>=deadline
        p.m.require(not stop(),"No accepted-boundary reserve remains")
        phase="paid_input_and_runtime"
        observed=runtime(spec)
        from experiments.root_native_baselines_20261023 import pins
        pins.verify(spec["operator_sources"])
        matrix,ids,matrix_descriptor=paid_inputs(spec)
        output=directory(p.OUTPUT)
        output.mkdir(parents=True,exist_ok=True)
        capacity=os.statvfs(output)
        p.m.require(capacity.f_bavail*capacity.f_frsize>=2<<30 and capacity.f_favail>=1000,
                    "Operator filesystem safety floor unavailable; exact reservation cannot be replaced with a silent default")
        lock_path=output/"EXECUTION.lock"
        descriptor=os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        with os.fdopen(descriptor,"r+b") as lock:
            p.m.require(os.fstat(lock.fileno()).st_nlink==1,"Operator lock aliases another inode")
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            directory(output)
            owner=dict(configuration=binding["configuration"],function_identity=p.function_identity(spec),
                matrix=matrix_descriptor,pool=spec["pool"],storage=binding["storage_reservation"])
            if (output/"OWNER.json").exists():
                p.m.require(p.m.checked(p.m.reference(output/"OWNER.json"))==owner,"Foreign native operator output owner")
            else:
                p.m.require({path.name for path in output.iterdir()}=={"EXECUTION.lock"},
                            "Unowned partial operator output preserved")
                p.m.seal(output/"OWNER.json",owner)
            phase="native_FAISS"
            clustered=cluster_stage(spec,matrix,output,owner,stop)
            phase="native_transferability"
            transferred=transfer_stage(clustered,output,owner,stop)
            profile=memory_profile(clustered)
            p.m.seal(report/"ACTUAL_MMD_MEMORY_BEFORE.json",profile)
            lower=profile["necessary_peak_buffer_bytes_lower_bound"]
            p.m.require(lower<profile["actual_GPU_free_bytes"],
                        "A necessary native MMD buffer exceeds current GPU free bytes; preserve clustering, no approximation")
            results=[]
            for fraction in ("0.05","0.10","0.20"):
                phase="native_MMD_"+fraction
                results.append(selection_stage(spec,matrix,ids,clustered,transferred,output,owner,fraction,stop))
            result=dict(status="ACTUAL_FULL_NATIVE_COINCIDE_OPERATOR_CANDIDATES_PENDING_INDEPENDENT_ACTUAL_REVIEW",
                job_id=binding["job_id"],actual=actual,runtime=observed,configuration=binding["configuration"],
                matrix=spec["matrix_receipt"],matrix_bytes_unchanged=p.m.reference(matrix_descriptor["path"])==matrix_descriptor,
                native_cluster=p.m.reference(output/"CLUSTER_COMPLETE.json"),
                native_transfer=p.m.reference(output/"TRANSFER_COMPLETE.json"),MMD_memory_profile=profile,
                outputs=results,models=0,forwards=0,new_features=0,new_downloads=0,
                full_selected_records_materialized=False,student_or_evaluation_READY=False,
                scientific_rounds_completed=0,elapsed_seconds=time.monotonic()-began,
                setup_failed_FAISS_and_MMD_costs_count=True)
            p.m.require(result["matrix_bytes_unchanged"],"Read-only paid matrix changed during native operator")
            reference=p.m.seal(output/"RESULT.json",result)
            p.m.seal(report/"RESULT.json",dict(result=reference,status=result["status"],outputs=results,
                elapsed_seconds=result["elapsed_seconds"],models=0,forwards=0,new_features=0))
            print(reference,flush=True)
        return 0
    except BaseException as error:
        p.m.seal(report/"FAILURE.json",dict(job_id=binding["job_id"],phase=phase,error=repr(error),
            elapsed_seconds=time.monotonic()-began,accepted_geometry_and_partial_outputs_preserved=True,
            no_automatic_successor=True,models=0,forwards=0,new_features=0,all_failed_setup_costs_count=True))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--binding",required=True)
    raise SystemExit(run(parser.parse_args().binding))
