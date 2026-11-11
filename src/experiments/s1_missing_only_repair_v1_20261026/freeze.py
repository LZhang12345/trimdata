"""Build exact repair inputs from paid local receipts, without reading cluster DATA."""

import json
from pathlib import Path
import shutil

from . import worker as w

COLLAB = w.ROOT / "collaboration/s1-missing-only-repair-20261026"
REMOTE = str(w.HOME / "s1_missing_only_repair_src_v1_20261026")


def descriptor(path):
    data = Path(path).read_bytes()
    return dict(path=str(Path(path).absolute()), bytes=len(data), sha256=w.base.digest(data))


def mirror(path, name, staging):
    ref = descriptor(path)
    shutil.copyfile(path, staging / "inputs" / name)
    return dict(ref, path=REMOTE + "/inputs/" + name)


def build():
    COLLAB.mkdir(exist_ok=True)
    staging = COLLAB / "staging-v1"
    (staging / "inputs").mkdir(parents=True, exist_ok=False)
    packet_path = w.ROOT / "collaboration/root-s1-budget4-campaign-20261025/S7_EXISTING_DT_GLOBAL3_PROVIDER_RECONCILIATION_V1.json"
    reconciliation = mirror(packet_path, "DT3_RECONCILIATION.json", staging)
    old = w.ROOT / "collaboration/s1-parallel-prerequisites-20261026/actual-terminal-v2/dt16-metadata/output"
    exported = json.loads((old / "RESULT.json").read_bytes())
    template = json.loads((w.ROOT / "collaboration/root-dt-union-qualification-v1-20261025/CONFIG_TEMPLATE_V1.json").read_bytes())
    cfg = {key: template[key] for key in ("pool", "eligibility", "original_configuration",
           "actual_parity", "actual_parity_review", "probe_sizes", "relationship_probe_size", "budgets")}
    entries = []
    for row in sorted(exported["providers"], key=lambda value: value["shard_index"]):
        index = row["shard_index"]
        if index == 3:
            entries.append(dict(shard_index=3, kind="composite_v3", provider_type="STRUCTURED_COMPOSITE_V3_ROOT",
                manifest=dict(path=str(w.CONTROL / "dt3-header/output/GLOBAL3_PROGRESS.json"),
                    bytes=5329, sha256="17762f0e9fcb491ed5472a113fb3d6a598ca88796ed06559f4dffbcd3eb0af26")))
            continue
        w.base.require(row["status"] == "COLLECTED_JSON_METADATA", "Reuse exactly fifteen existing successful headers")
        source = old / ("input-%02d.json" % index)
        actual = descriptor(source)
        original_ref = row["metadata"]["original"]
        w.base.require(actual["bytes"] == original_ref["bytes"] and actual["sha256"] == original_ref["sha256"],
                       "Retained header mirror changed")
        ref = mirror(source, "DT-%02d.json" % index, staging)
        value = json.loads(source.read_bytes())
        if row["kind"] == "original":
            entries.append(dict(shard_index=index, kind="original", completion=ref,
                                database=value["database"], identity=value["measurement_identity"]))
        else:
            entries.append(dict(shard_index=index, kind=row["kind"], manifest=ref))
    w.base.require(len(entries) == 16 and {entry["shard_index"] for entry in entries} == set(range(16)),
                   "Complete typed metadata roster required")
    cfg["feature_shards"] = entries
    source_packet = json.loads((w.ROOT / "collaboration/root-dt-union-qualification-v1-20261025/SOURCE_PACKET_V1.json").read_bytes())
    source_map = dict(source_packet["sources"])
    vqa_config_path = w.ROOT / "collaboration/root-babel-start-evaluation-v2-20261025/CONFIG_V2_VQAv2.json"
    vqa_cfg = json.loads(vqa_config_path.read_bytes())
    for relative, sha in {**vqa_cfg["sources"], **w.base.sources()}.items():
        w.base.require(relative not in source_map or source_map[relative] == sha, "Conflicting frozen source closure")
        source_map[relative] = sha
    for name in w.OWN_FILES:
        path = w.HERE / name
        source_map[str(path.relative_to(w.ROOT))] = w.base.digest(path.read_bytes())
    for relative, sha in source_map.items():
        w.base.require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                       "Literal relative execution source required")
        source = w.ROOT / relative
        w.base.require(source.is_file() and not source.is_symlink()
                       and w.base.digest(source.read_bytes()) == sha, "Changed exact source: " + relative)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    plan = dict(schema="s1-missing-only-repair-plan-v1", owner="Session1", units=list(w.UNITS),
                remote_source_root=REMOTE, resources=w.RESOURCES, sources=dict(sorted(source_map.items())),
                models=0, forwards=0, student_release_authorized=False, reconciliation=reconciliation,
                dt_config=cfg,
                vqav2_binding=dict(path=str(w.HOME / "native_campaign_v3_20261023/registry/root-start-evaluation-v2/VQAv2/BINDING.json"),
                    bytes=3840, sha256="e7b980fdb562e358199d68852f2ec8cca6ca7427b638e01729d0e7589d84c1fd"),
                vqav2_terminal=dict(path="/data/user_data/luyangz/native_campaign_20261023/start-evaluation-v2/VQAv2/attempts/10923672-r2/RESULT.json",
                    bytes=3261, sha256="625b4171c293dcebe981e7a87bbb8d7cb2773168e89e5d7bed169f7e2d2cc5fa"),
                ownership="Author-requested S1 missing-only repair; Root running runtime/archive and all paid providers unchanged.",
                limitations=["No new model/features/QA generation", "No student or full native geometry permission",
                             "VQAv2 missing-QA roster must precede a reviewed genuine inference continuation"])
    local = w.base.publish(COLLAB / "PLAN_V1.json", plan)
    shutil.copyfile(local["path"], staging / "PLAN_V1.json")
    print(json.dumps(dict(local=local, remote=dict(local, path=REMOTE + "/PLAN_V1.json"),
                          staged_source_files=len(source_map)), sort_keys=True))


if __name__ == "__main__":
    build()
