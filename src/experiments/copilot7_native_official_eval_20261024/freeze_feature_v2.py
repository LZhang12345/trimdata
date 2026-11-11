"""Build a new source-only DT32 packet from the preserved reviewed V1 bundle."""
import argparse
import ast
import copy
import gzip
import hashlib
import io
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

from . import common as c
from .feature_v2 import INPUT_SCOPE


RELATIVE = Path("experiments/copilot7_native_official_eval_20261024")
CONTROL = Path("collaboration/session7-native-official-execution-20261024")
REMOTE = Path("/home/luyangz/data_selection_multimodal")
NEW_FILES = (
    "feature_v2.py", "run_datatailor_v2.sbatch", "test_feature_v1.py",
    "test_feature_v2.py", "freeze_feature_v2.py",
)


def write_exact(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        c.require(path.read_bytes() == body, "Existing frozen source/artifact differs: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(body)


def remote_descriptor(local, remote):
    return dict(c.desc(local), path=str(remote))


def freeze(staging):
    base = c.ROOT / CONTROL
    output = base / "datatailor32-v2"
    old_spec = c.read(base / "datatailor32-v1/SPEC_V1.json")
    old_manifest = c.read(base / "FEATURE_QUALIFIER_SOURCE_MANIFEST_V2.json")
    old_archive = old_manifest["archive"]
    c.require(c.desc(old_archive["path"]) == old_archive
              and old_archive["sha256"] == "a4f006ae46dfcaf23560af1d9c918da2c077964dd8bae229f6185cdf9486032a",
              "Preserved independently reviewed bundle differs")
    expected = {**old_spec["dependencies"], **old_spec["sources"]}
    bodies = {}
    with tarfile.open(old_archive["path"], "r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            c.require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                      and str(path) == member.name and member.name not in bodies,
                      "Unsafe or duplicate source archive member")
            stream = bundle.extractfile(member)
            c.require(stream is not None, "Source archive member unreadable")
            with stream:
                bodies[member.name] = stream.read()
    c.require(set(bodies) == set(expected)
              and all(hashlib.sha256(body).hexdigest() == expected[name] for name, body in bodies.items()),
              "Original197 source closure differs")
    for name in NEW_FILES:
        relative = str(RELATIVE / name)
        c.require(relative not in bodies, "Do not replace any preserved source")
        bodies[relative] = (c.ROOT / relative).read_bytes()
    for relative, body in bodies.items():
        write_exact(staging / relative, body)
        if relative.endswith(".py"):
            ast.parse(body, filename=relative, feature_version=(3, 10))
        elif relative.endswith(".sbatch"):
            subprocess.run(["bash", "-n", str(staging / relative)], check=True)
    own_sources = dict(old_spec["sources"])
    own_sources.update({str(RELATIVE / name): c.sha(staging / RELATIVE / name) for name in NEW_FILES})
    all_sources = {**old_spec["dependencies"], **own_sources}
    c.require(len(all_sources) == 202 and len(old_manifest["exact_native85"]) == 85
              and all(all_sources[key] == value for key, value in old_manifest["exact_native85"].items()),
              "New namespace must preserve the exact native85 and all191 dependencies")
    c.verify_sources(all_sources, staging)
    test_command = [
        sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        str(RELATIVE / "test_feature_v1.py"), str(RELATIVE / "test_feature_v2.py"),
    ]
    test = subprocess.run(test_command, cwd=staging, text=True, capture_output=True, check=False)
    c.require(test.returncode == 0, "Staged source tests failed:\n" + test.stdout + test.stderr)
    packed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=packed, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for relative, body in sorted(bodies.items()):
                info = tarfile.TarInfo(relative)
                info.size, info.mode, info.mtime = len(body), 0o644, 0
                bundle.addfile(info, io.BytesIO(body))
    bundle_path = output / "SOURCE_BUNDLE_V1.tar.gz"
    write_exact(bundle_path, packed.getvalue())
    c.seal(output / "SOURCE_MANIFEST_V1.json", dict(
        own_sources=own_sources, dependencies=old_spec["dependencies"],
        exact_native85=old_manifest["exact_native85"], archive=c.desc(bundle_path),
        predecessor_archive=old_archive, model_or_dataset_files_in_bundle=False,
    ))
    spec = copy.deepcopy(old_spec)
    remote_control = REMOTE / CONTROL / "datatailor32-v2"
    prior_binding = base / "native15-prep-v2/actual-registration/BINDING.json"
    failure = base / "datatailor32-v1/ACTUAL_FAILURE_10910211_V1.json"
    spec.update(
        code_root=str(REMOTE / "session7_native_prerequisites_src_v3"),
        job_name="dsmm-s7-dt32-qual-v2",
        function="native.datatailor32.actual-forward-engineering.v2",
        wrapper=str(RELATIVE / "run_datatailor_v2.sbatch"),
        output="/data/user_data/luyangz/native_campaign_20261023/session7-native-prerequisites/datatailor32-v2",
        sources=own_sources, input_dependency_scope=INPUT_SCOPE,
        failed_predecessor=dict(
            job_id="10910211", bound_sources_preserved=True,
            failure=remote_descriptor(failure, REMOTE / CONTROL / "datatailor32-v1/ACTUAL_FAILURE_10910211_V1.json"),
        ),
        preparation_readout=dict(
            binding=remote_descriptor(prior_binding, REMOTE / CONTROL / "native15-prep-v2/registration/BINDING.json"),
            job_id="10909981",
            output="/data/user_data/luyangz/native_eval_session7_20261024/native15-prep-v2",
            export_root=str(remote_control / "preparation-readout"),
            approved_control_root=str(remote_control),
        ),
    )
    configuration = c.seal(output / "SPEC_V1.json", spec)
    remote_configuration = dict(configuration, path=str(remote_control / "SPEC_V1.json"))
    c.seal(output / "SELF_REVIEW_V1.json", dict(
        status="SELF_REVIEW_SOURCE_ONLY_NOT_RUNTIME", configuration=remote_configuration,
        sources=own_sources, dependencies=spec["dependencies"],
        exact_native85_preserved=True, original197_source_bytes_preserved=True,
        Python310_AST_all_sources=True, shell_wrappers_syntax=True,
        test_command=test_command, test_stdout=test.stdout, test_stderr=test.stderr,
        input_dependency_scope=INPUT_SCOPE,
        changed_numerical_feature_constructor=False, full_pool_or_student_release_authorized=False,
        counterevidence="10910211 failed before model construction; model assets have not yet been requalified.",
    ))
    c.seal(output / "PACKET_V1.json", dict(
        status="AWAITING_INDEPENDENT_EXACT_SOURCE_CONFIG_REVIEW",
        owner=c.OWNER, configuration=remote_configuration,
        source_manifest=c.desc(output / "SOURCE_MANIFEST_V1.json"),
        archive=c.desc(bundle_path), source_count=len(all_sources),
        predecessor_job="10910211", actual_retry_submitted=False,
        requested_review="Engineering1024/model-component physical dependency separation and exact native15 metadata export; unchanged native numerical kernel.",
        no_fullpool_runtime_or_student_PASS=True,
    ))
    return dict(configuration=remote_configuration, source_count=len(all_sources), tests=test.stdout.strip(),
                archive=c.desc(bundle_path), staging=str(staging))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    print(c.canonical(freeze(args.staging)).decode(), end="")
