"""HOME/source-only exact packet preparation; no DATA or scheduler operations."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

from experiments.root_native_campaign_v7_20261024.common import seal
from . import component as c

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "collaboration/session7-coincide-actual-output-review-20261028/v2"


def source_closure(initial):
    seen, pending = {}, list(initial)
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        c.require(path.is_file() and path.is_relative_to(ROOT), "Exact source closure path required")
        raw = path.read_bytes()
        seen[path] = hashlib.sha256(raw).hexdigest()
        for parent in path.parents:
            if parent == ROOT:
                break
            init = parent / "__init__.py"
            if init.is_file():
                pending.append(init)
        if path.suffix != ".py":
            continue
        relative = path.relative_to(ROOT).with_suffix("")
        package = list(relative.parts[:-1])
        for node in ast.walk(ast.parse(raw)):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    c.require(node.level <= len(package), "Invalid relative source import")
                    prefix = ".".join(package[:len(package) - node.level + 1])
                    module = prefix + ("." + node.module if node.module else "")
                else:
                    module = node.module or ""
                modules.append(module)
                modules.extend(module + "." + alias.name for alias in node.names if alias.name != "*")
            for name in modules:
                if not name.startswith("experiments."):
                    continue
                target = ROOT.joinpath(*name.split("."))
                if target.with_suffix(".py").is_file():
                    pending.append(target.with_suffix(".py"))
                elif (target / "__init__.py").is_file():
                    pending.append(target / "__init__.py")
    return dict(sorted((str(path.relative_to(ROOT)), digest) for path, digest in seen.items()))


def prepare(output=CONTROL):
    output = Path(output)
    prior = ROOT / "collaboration/session7-coincide-actual-output-review-20261028/v1/REVIEW_COMPONENT_CONFIG_V2.json"
    cfg = deepcopy(json.loads(prior.read_bytes()))
    paths = [ROOT / name for name in cfg["sources"]]
    paths.extend(path for path in Path(__file__).parent.iterdir() if path.suffix == ".py")
    sources = source_closure(paths)
    cfg.update(schema="s7-actual-coincide-output-review-v2", source_config_version=3,
        sources=sources, heavy_metadata_lock=str(c.BASE / "collaboration/root-four-axis-meta-20261028/OWNED10954931_HEAVY_METADATA.lock"),
        execution_control=str(c.BASE / "collaboration/session7-coincide-actual-output-review-20261028/v2/execution"),
        entry_API=c.MODULE + ".run", registered_CPU_executor=c.MODULE.rsplit(".", 1)[0] + ".executor.execute",
        required_Root_review_status="ROOT_PERSONALLY_APPROVED_S7_COINCIDE_OUTPUT_REVIEW_SOURCE_V2",
        mode_policy="UID2708899 singleton regular no-world-write; exact0660/GID500 exception; stable anchored FD/fullSHA; HOME strict unchanged",
        materialization_io_status="IMPLEMENTED: exact protected pool original-record read + anchored immutable sink; Root separate measured1GiB/32inode grant required",
        materialization_output="/data/hf_cache/project_artifacts/DS-MultiModal/coincide_native_records_20261028/v2",
        materialization_bridge_API="v2.bridge.materialize_candidate + actual v2.materialize.materialize",
        no_blanket_mode_relaxation=True, actual_DATA_reads_so_far=0, actual_job=None)
    cfg.pop("new_output_lease", None)
    cfg["not_performed"] = ["operator/FAISS/MMD rerun", "feature acquisition", "model forwards",
        "actual new reader execution", "actual original-record materialization", "independent actual method acceptance",
        "student training/evaluation", "human quality certification"]
    cfg["Root_launch_only"] = "Root-owned10954931 registered numeric CPU step; exact executor command in handoff"
    cfg["payload_invocation"] = "Root reviewed V3 config + detached V2 approval/authority; --stat-only first, then --materialization-grant-ref"
    cfg["source_closure"] = "STATIC_IMPORT_CLOSURE_PLUS_ALL_208_PRESERVED_V1_DEPENDENCIES_AND_V2_SOURCE"
    config = seal(output / "REVIEW_COMPONENT_CONFIG_V3.json", cfg)
    archive = output / "SOURCE_CLOSURE_V3.tar.gz"
    c.require(not archive.exists(), "Do not replace an already frozen source archive")
    with tarfile.open(archive, "w:gz") as sink:
        for name in sources:
            sink.add(ROOT / name, arcname=name, recursive=False)
    raw = archive.read_bytes()
    package = dict(path=str(archive), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    command = ["srun", "--jobid=10954931", "--overlap", "--exact", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=4", "--mem=32G", "--gres=none", "--time=00:20:00",
        "--chdir=" + str(c.BASE), c.PYTHON, "-B", "-s", "-u", "-m",
        c.MODULE.rsplit(".", 1)[0] + ".executor", "--configuration-ref",
        json.dumps(dict(config, path=str(c.BASE / config["path"].split(str(ROOT) + "/", 1)[1])), separators=(",", ":"))]
    return seal(output / "SOURCE_CONFIG_EXECUTOR_HANDOFF_V3.json", dict(
        status="EXACT_V2_SOURCE_AND_V3_CONFIG_FOR_ROOT_AND_INDEPENDENT_REVIEW_NOT_ACTUAL_DATA_PASS",
        configuration=config, source_archive=package, sources=sources, source_count=len(sources),
        own_source_files=[name for name in sources if "copilot7_coincide_actual_review_v2_" in name],
        fixes=["F1 normalized independent principal", "F2 fixed exact protocol/IDs/order/count/whole materialization identity",
               "F3 read actual review bytes matching descriptor", "F4 scoped0660/GID500 paid DATA stableFD",
               "actual original-record read/sink", "existing shared Root heavy-read lock"],
        next_executable_argv_prefix=command,
        required_detached_argv=["--Root-review-ref REAL_ROOT_REVIEW_DESCRIPTOR_JSON",
                                "--Root-authorization-ref REAL_ROOT_AUTHORITY_DESCRIPTOR_JSON", "--stat-only"],
        operation_sequence=["stat-only", "materialize", "independent actual selection review", "finalize"],
        all_actual_IO_requires_current_registered_owned_process=True, DATA_or_science_executed=False,
        no_chmod_no_matrix_rehash_no_FAISS_MMD_rerun=True, old_V1_BLOCK_preserved=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CONTROL)
    print(json.dumps(prepare(parser.parse_args().output), sort_keys=True))
