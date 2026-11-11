"""Recompute full-source eligibility rather than trusting a pool status string."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import time
import unicodedata

from experiments.root_native_campaign_20261023 import common as c

ORIGINAL_COUNT = 665298
ORIGINAL_BYTES = 1029887963
ORIGINAL_SHA = "ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8"
SOURCES = ("coco", "gqa", "ocr_vqa", "textvqa", "vg", "text_only")
EMPTY_SOLE_ASSISTANT = {
    245723: ("245723:000000178275", "e615ec579e183058658804f6bea94ccace71207593808c2b218fb7c902397f2c"),
    247644: ("247644:000000047952", "f75d1c940c09e0ac31f8f4a72e9772cbe37b511961b6c0d6c9cb4a0c7bab9e11"),
}
NORMALIZATION = "NFKC-whitespace-collapse-casefold-v1"
VALIDITY_POLICY = {
    "schema": "native-validity-policy-v1",
    "empty_sole_assistant": {str(index): {"record_id": value[0], "original_row_sha256": value[1]}
                             for index, value in EMPTY_SOLE_ASSISTANT.items()},
    "other_invalid_records": "FAIL_CLOSED_REQUIRES_NEW_FROZEN_POLICY",
    "selection_unit": "whole_original_conversation",
    "top_up_or_replacement": False,
}
ALLOWED_REASONS = (
    "frozen_empty_sole_assistant_v1", "protected_text_qa", "protected_text_conversation",
    "protected_multimodal_qa", "protected_image_id", "protected_image_rgb",
)


def source_identity():
    return {"relative_path": "experiments/root_native_input_authority_20261023/pool_authority_v1.py",
            "bytes": Path(__file__).stat().st_size, "sha256": c.sha(__file__)}


def row_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def normalize(text):
    c.require(isinstance(text, str), "QA content must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def turns(row):
    values = row.get("conversations")
    c.require(isinstance(values, list) and len(values) >= 2 and len(values) % 2 == 0,
              "Complete paired native conversation required")
    c.require(all(isinstance(value, dict) and value.get("from") == ("human" if index % 2 == 0 else "gpt")
                  and isinstance(value.get("value"), str) for index, value in enumerate(values)),
              "Native role order/content differs")
    return values


def conversation_key(values):
    return row_hash([(value["from"], normalize(value["value"])) for value in turns({"conversations": values})])


def record_id(index, row):
    return f'{index:06d}:{row.get("id", "")}'


def source(row):
    image = row.get("image")
    if image is None:
        return "text_only"
    c.require(isinstance(image, str) and bool(image), "Image must be an explicit nonempty relative locator")
    path = PurePosixPath(image)
    c.require(not path.is_absolute() and ".." not in path.parts and str(path) == image,
              "Unsafe/noncanonical original image locator")
    c.require(path.parts[0] in SOURCES[:-1] and len(path.parts) >= 2, "Unknown original image source")
    return path.parts[0]


def source_image_key(namespace, identifier):
    c.require(isinstance(namespace, str) and namespace and isinstance(identifier, str) and identifier,
              "Explicit image namespace/ID required")
    return namespace, str(int(identifier)) if identifier.isdecimal() else identifier


def raster_key(image):
    c.require(type(image["width"]) is int and type(image["height"]) is int and
              image["width"] > 0 and image["height"] > 0, "Positive exact raster shape required")
    digest = image["rgb_sha256"]
    c.require(isinstance(digest, str) and len(digest) == 64 and
              all(value in "0123456789abcdef" for value in digest), "Exact decoded RGB SHA256 required")
    return image["width"], image["height"], digest


def _unique(values, label):
    c.require(len(values) == len(set(values)), "Duplicate " + label)
    return set(values)


class ProtectedIndex:
    """Input-only exact matching; no benchmark predictions, scores or fitted weights."""

    def __init__(self, document):
        c.require(document["schema"] == "native-protected-input-contract-v1" and
                  document["status"] == "FROZEN_AUTHORIZED_INPUT_IDENTITIES" and
                  document["normalization"] == NORMALIZATION and document["outcome_tuning"] is False,
                  "Explicit frozen input-only protection contract required")
        self.image_ids, self.rasters = set(), set()
        self.text_qa, self.conversations = defaultdict(set), defaultdict(set)
        self.visual_questions = defaultdict(set)
        self.images = {}
        scopes = document["role_scopes"]
        c.require(isinstance(scopes, dict) and bool(scopes), "Exact nonempty role scopes required")
        observed = {role: {"images": set(), "qas": set(), "conversations": set()} for role in scopes}
        for value in document["images"]:
            role, key = value["role"], value["key"]
            c.require(role in scopes and key not in observed[role]["images"], "Foreign/duplicate protected image key")
            c.require(isinstance(key, str) and key, "Nonempty protected image key required")
            identity = source_image_key(value["namespace"], value["image_id"])
            pixels = raster_key(value)
            self.images[(role, key)] = value
            self.image_ids.add(identity)
            self.rasters.add(pixels)
            for alias in value.get("aliases", []):
                self.image_ids.add(source_image_key(alias["namespace"], alias["image_id"]))
            observed[role]["images"].add(key)
        for value in document["qas"]:
            role, key = value["role"], value["key"]
            c.require(role in scopes and scopes[role]["qa_access"] == "authorized" and
                      key not in observed[role]["qas"] and isinstance(key, str) and key,
                      "Foreign/closed/duplicate protected QA key")
            question = normalize(value["question"])
            c.require(bool(question), "Empty protected question")
            if value["image_key"] is None:
                answers = value["answers"]
                c.require(isinstance(answers, list) and answers and
                          all(isinstance(answer, str) and normalize(answer) for answer in answers),
                          "Authorized text QA requires explicit nonempty answer identities")
                for answer in answers:
                    self.text_qa[(question, normalize(answer))].add((role, key))
            else:
                image = self.images[(role, value["image_key"])]
                identity = source_image_key(image["namespace"], image["image_id"])
                self.visual_questions[("id", identity, question)].add((role, key))
                self.visual_questions[("rgb", raster_key(image), question)].add((role, key))
                for alias in image.get("aliases", []):
                    self.visual_questions[("id", source_image_key(alias["namespace"], alias["image_id"]),
                                           question)].add((role, key))
            observed[role]["qas"].add(key)
        for value in document["conversations"]:
            role, key = value["role"], value["key"]
            c.require(role in scopes and scopes[role]["qa_access"] == "authorized" and
                      key not in observed[role]["conversations"] and isinstance(key, str) and key,
                      "Foreign/closed/duplicate protected conversation key")
            self.conversations[conversation_key(value["turns"])].add((role, key))
            observed[role]["conversations"].add(key)
        for role, expected in scopes.items():
            c.require(expected["qa_access"] in ("authorized", "image_only"), "Explicit QA access policy required")
            for name in ("images", "qas", "conversations"):
                wanted = _unique(expected[name], "declared protected " + name)
                c.require(observed[role][name] == wanted, "Exact protected role/key scope differs: " + role + "/" + name)
            if expected["qa_access"] == "image_only":
                c.require(not observed[role]["qas"] and not observed[role]["conversations"],
                          "Image-only/confirmation role cannot expose QA contents")
        self.coverage = {role: {name: len(values) for name, values in actual.items()}
                         for role, actual in observed.items()}

    def reasons(self, row, image=None):
        values = turns(row)
        result = defaultdict(set)
        if source(row) == "text_only":
            for index in range(0, len(values), 2):
                matches = self.text_qa.get((normalize(values[index]["value"]), normalize(values[index + 1]["value"])), ())
                result["protected_text_qa"].update(matches)
            result["protected_text_conversation"].update(self.conversations.get(conversation_key(values), ()))
        else:
            name = source(row)
            identity = source_image_key(name, PurePosixPath(row["image"]).stem)
            if identity in self.image_ids:
                result["protected_image_id"].add(("image_identity", row_hash(identity)))
            for index in range(0, len(values), 2):
                question = normalize(values[index]["value"].replace("<image>", ""))
                result["protected_multimodal_qa"].update(self.visual_questions.get(("id", identity, question), ()))
            if image is not None:
                pixels = raster_key(image)
                if pixels in self.rasters:
                    result["protected_image_rgb"].add(("dimensioned_RGB", row_hash(pixels)))
                for index in range(0, len(values), 2):
                    question = normalize(values[index]["value"].replace("<image>", ""))
                    result["protected_multimodal_qa"].update(self.visual_questions.get(("rgb", pixels, question), ()))
        return [{"reason": name, "matches": [list(value) for value in sorted(matches)]}
                for name, matches in sorted(result.items()) if matches]


def validity_reasons(index, row):
    values = turns(row)
    empty = [value for value in values if value["from"] == "gpt" and not value["value"].strip()]
    if index in EMPTY_SOLE_ASSISTANT:
        expected_id, digest = EMPTY_SOLE_ASSISTANT[index]
        c.require(record_id(index, row) == expected_id and row_hash(row) == digest and
                  len(values) == 2 and len(empty) == 1, "Frozen empty-target original record differs")
        return [{"reason": "frozen_empty_sole_assistant_v1",
                 "matches": [["validity_policy", expected_id]]}]
    c.require(not empty, "Unexpected empty assistant content requires a new frozen validity policy")
    return []


def budgets(count):
    c.require(type(count) is int and count > 0, "Nonempty actual eligible denominator required")
    return {fraction: count * percentage // 100 for fraction, percentage in (("0.05", 5), ("0.10", 10), ("0.20", 20))}


def partition(original, media, protected):
    retained, excluded, needed, checked = [], [], set(), set()
    original_counts, retained_counts, excluded_counts = Counter(), Counter(), Counter()
    for index, row in enumerate(original):
        name = source(row)
        original_counts[name] += 1
        reasons = validity_reasons(index, row)
        if not reasons:
            reasons = protected.reasons(row)
            if row.get("image") and not reasons:
                c.require(row["image"] in media, "Missing media is a readiness failure, never an exclusion: " + row["image"])
                checked.add(row["image"])
                reasons = protected.reasons(row, media[row["image"]])
        if reasons:
            excluded.append({"source_index": index, "record_id": record_id(index, row),
                             "original_row_sha256": row_hash(row), "source": name, "reasons": reasons})
            excluded_counts[name] += 1
        else:
            retained.append(index)
            retained_counts[name] += 1
            if row.get("image"):
                needed.add(row["image"])
    counts = {name: {"original": original_counts[name], "retained": retained_counts[name],
                     "excluded": excluded_counts[name]} for name in SOURCES}
    return {"retained_indices": retained, "excluded": excluded, "source_counts": counts,
            "required_eligible_images": sorted(needed), "checked_images": sorted(checked)}


def verify_partition(original, rows, indices, identifiers, exclusions, expected):
    c.require(all(type(index) is int and 0 <= index < len(original) for index in indices),
              "Original source index out of bounds")
    c.require(indices == sorted(indices) and len(indices) == len(set(indices)) == len(rows) == len(identifiers),
              "Retained counts/order/unique indices differ")
    removed = [value["source_index"] for value in exclusions]
    c.require(all(type(index) is int and 0 <= index < len(original) for index in removed) and
              removed == sorted(removed) and len(removed) == len(set(removed)), "Excluded source indices differ")
    c.require(not set(indices) & set(removed) and set(indices) | set(removed) == set(range(len(original))),
              "Missing or overlapping original retained/excluded partition")
    c.require(indices == expected["retained_indices"] and exclusions == expected["excluded"],
              "Eligibility/exclusion policy or protected evidence differs")
    for row, index, identifier in zip(rows, indices, identifiers):
        c.require(identifier == record_id(index, original[index]) and row_hash(row) == row_hash(original[index]),
                  "Original whole row/content/turns/record identity changed")
    c.require(all(reason["reason"] in ALLOWED_REASONS for value in exclusions for reason in value["reasons"]),
              "Undeclared exclusion reason")


def require_allocation():
    job = os.environ.get("SLURM_JOB_ID", "")
    c.require(job.isdecimal() and not os.environ.get("SLURM_ARRAY_JOB_ID"),
              "Authority DATA work requires an actual owned scalar CPU+GPU allocation")
    raw = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True, timeout=30)
    fields = dict(value.split("=", 1) for value in raw.split() if "=" in value)
    allocation = dict(value.split("=", 1) for value in fields.get("AllocTRES", "").split(",") if "=" in value)
    c.require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING" and
              fields.get("NodeList") == socket.gethostname().split(".")[0] and
              fields.get("UserId", "").split("(")[0] == os.environ.get("USER") and
              int(allocation.get("gres/gpu", "0")) >= 1 and int(allocation.get("cpu", "0")) >= 1,
              "Actual allocated CPU+GPU node required before DATA access")
    return {"job_id": job, "node": fields["NodeList"], "raw": raw}


def load_original(description):
    c.require(description["bytes"] == ORIGINAL_BYTES and description["sha256"] == ORIGINAL_SHA,
              "Only the frozen complete original665298 metadata is admitted")
    original = c.checked(description)
    c.require(isinstance(original, list) and len(original) == ORIGINAL_COUNT, "Complete original source roster differs")
    return original


def verify_protection_authority(document, roles, protocol):
    c.require(protocol["schema"] == "native-authority-protection-protocol-v1" and
              protocol["status"] == "FROZEN_PRE_OUTCOME_INPUT_CONTRACT" and
              protocol["role_manifest"] == document["role_manifest"] and
              protocol["normalization"] == NORMALIZATION and protocol["outcome_tuning"] is False and
              protocol["role_scopes"] == document["role_scopes"],
              "Independent frozen role/QA/image protocol differs")
    for name in ("images", "qas", "conversations"):
        c.require(protocol[name + "_sha256"] == row_hash(document[name]),
                  "Frozen authorized protocol payload differs: " + name)
    for role, bank in {**{key: value for key, value in roles["banks"].items() if key != "train"},
                       "confirmation": roles["confirmation"]}.items():
        c.require(role in document["role_scopes"], "Existing protected native role was dropped: " + role)
        actual_images = {source_image_key(value["namespace"], value["image_id"])
                         for value in document["images"] if value["role"] == role}
        c.require(actual_images == {source_image_key("coco", str(value)) for value in bank["image_ids"]},
                  "Existing native protected image-ID role differs: " + role)
        actual_qas = {value["key"] for value in document["qas"] if value["role"] == role}
        if role == "confirmation":
            c.require(document["role_scopes"][role]["qa_access"] == "image_only" and not actual_qas,
                      "Protected confirmation QA access remains closed")
        else:
            c.require(actual_qas == {str(value) for value in bank["question_ids"]},
                      "Existing native protected QA role differs: " + role)


def load_inputs(original_description, media_description, protected_description):
    original = load_original(original_description)
    media = c.checked(media_description)
    c.require(media["schema"] == "native-authority-media-inventory-v1" and
              media["original_metadata"] == original_description and
              isinstance(media["assets"], dict), "Exact source-bound native media inventory required")
    document = c.checked(protected_description)
    # These descriptors preserve Root's frozen upstream role/protocol contract;
    # this module neither reinterprets a closed bank nor chooses new boundaries.
    roles = c.checked(document["role_manifest"])
    protocol = c.checked(document["evaluation_protocol"])
    verify_protection_authority(document, roles, protocol)
    protected = ProtectedIndex(document)
    return original, media, protected


def logical_media_closure(media, needed):
    root = Path(media["image_root"]).absolute()
    c.require(".." not in root.parts, "Noncanonical logical media root")
    identities = []
    for name in needed:
        expected = media["assets"][name]
        path = Path(media["image_root"]) / name
        c.require(str(path.absolute()) == expected["path"] and path.absolute().is_relative_to(root) and
                  type(expected["bytes"]) is int and expected["bytes"] > 0 and
                  isinstance(expected["sha256"], str) and len(expected["sha256"]) == 64 and
                  all(value in "0123456789abcdef" for value in expected["sha256"]),
                  "Canonical media path/encoded identity differs")
        identities.append((name, expected["bytes"], expected["sha256"], *raster_key(expected)))
    return {"unique_images": len(identities), "dimensioned_encoded_RGB_identity_sha256": row_hash(identities),
            "missing_eligible_media": 0}


def verify_images(media, needed):
    from PIL import Image
    closure = logical_media_closure(media, needed)
    root = Path(media["image_root"]).resolve()
    for name in needed:
        expected = media["assets"][name]
        path = Path(media["image_root"]) / name
        c.require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Physical image alias escaped inventory")
        c.require(c.desc(path) == {key: expected[key] for key in ("path", "bytes", "sha256")},
                  "Encoded eligible image changed")
        with Image.open(path) as opened:
            rgb = opened.convert("RGB")
            try:
                actual = {"width": rgb.width, "height": rgb.height,
                          "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
                c.require(raster_key(actual) == raster_key(expected), "Decoded eligible image identity changed")
            finally:
                rgb.close()
    return {**closure, "actual_encoded_and_decoded_RGB_checks": True}


def produce(original_description, media_description, protected_description, output):
    actual = require_allocation()
    original, media, protected = load_inputs(original_description, media_description, protected_description)
    expected = partition(original, media["assets"], protected)
    indices = expected["retained_indices"]
    rows = [original[index] for index in indices]
    identifiers = [record_id(index, original[index]) for index in indices]
    verify_partition(original, rows, indices, identifiers, expected["excluded"], expected)
    closure = verify_images(media, expected["checked_images"])
    output = Path(output)
    records = c.seal(output / "records.json", rows)
    images = c.seal(output / "images.json", {"assets": {
        name: media["assets"][name] for name in expected["required_eligible_images"]}})
    exclusion = c.seal(output / "exclusions.json", {
        "schema": "native-complete-original-partition-v1", "original_count": len(original),
        "retained_count": len(indices), "excluded": expected["excluded"],
        "source_counts": expected["source_counts"], "validity_policy": VALIDITY_POLICY,
        "retained_indices_sha256": row_hash(indices), "protected_contract": protected_description})
    authority = {
        "schema": "native-full-source-pool-authority-v1", "status": "PASS_EXACT_ORIGINAL_PARTITION_AND_ELIGIBLE_MEDIA",
        "original_metadata": original_description, "media_inventory": media_description,
        "protected_contract": protected_description, "role_coverage": protected.coverage,
        "records_json": records, "image_manifest": images, "exclusions": exclusion,
        "source_counts": expected["source_counts"], "eligible_N": len(indices),
        "K": budgets(len(indices)), "validity_policy": VALIDITY_POLICY, "media_closure": closure,
        "eligible_unique_images": len(expected["required_eligible_images"]),
        "protected_audit_counts": dict(Counter(reason["reason"] for value in expected["excluded"]
                                               for reason in value["reasons"])),
        "source": source_identity(), "actual": actual, "outcome_tuning": False,
        "top_up_or_replacement": False, "trainer_or_signal_qualified": False,
        "actual_tokenizer_target_survival_qualified": False,
        "physical_media_readiness": True, "physical_media_readiness_scope": "producer_snapshot_only",
        "runtime_transport_required": False}
    authority_path = output / "AUTHORITY.json"
    if authority_path.exists():
        previous = c.read(authority_path)
        c.require({key: value for key, value in previous.items() if key != "actual"} ==
                  {key: value for key, value in authority.items() if key != "actual"},
                  "Accepted authority input/output/policy changed")
        receipt = c.desc(authority_path)
    else:
        receipt = c.seal(authority_path, authority)
    pool = {"schema": "published-llava-pool-v1", "status": "PASS_POOL",
            "pool_kind": "all_source_protected_bridge", "source_indices": indices,
            "record_ids": identifiers, "count": len(indices), "records_json": records,
            "source_json": original_description, "image_root": media["image_root"],
            "image_manifest": images, "protected_exclusions": exclusion, "input_authority": receipt}
    return c.seal(output / "POOL.json", pool)


def validate_pool(description, eligibility_description, *, full_media_revalidation=False, audit_output=None):
    """Logical consumer by default; physical replay is an explicitly costed audit."""
    started = time.monotonic()
    actual = require_allocation()
    eligibility = c.checked(eligibility_description)
    c.require(eligibility["status"] == "PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY" and
              eligibility["pool"] == description and eligibility["protected_QA_image_RGB_overlap_count"] == 0 and
              eligibility["missing_media_count"] == 0 and eligibility["outcome_tuning"] is False,
              "Independent frozen eligibility pin is required, not self-referential pool metadata")
    expected_protected_description = eligibility["protected_contract"]
    pool = c.checked(description)
    c.require(pool["schema"] == "published-llava-pool-v1" and pool["status"] == "PASS_POOL" and
              pool["pool_kind"] == "all_source_protected_bridge", "Exact all-source protected pool required")
    authority = c.checked(pool["input_authority"])
    c.require(authority["schema"] == "native-full-source-pool-authority-v1" and
              authority["status"] == "PASS_EXACT_ORIGINAL_PARTITION_AND_ELIGIBLE_MEDIA" and
              eligibility["input_authority"] == pool["input_authority"] and
              authority["protected_contract"] == expected_protected_description and
              authority["original_metadata"] == pool["source_json"] and
              authority["source"] == source_identity() and authority["outcome_tuning"] is False and
              authority["top_up_or_replacement"] is False and authority["trainer_or_signal_qualified"] is False,
              "Authority/source/protection identity differs")
    original, media, protected = load_inputs(pool["source_json"], authority["media_inventory"],
                                            expected_protected_description)
    protected_document = c.checked(expected_protected_description)
    c.require(protected_document["evaluation_protocol"] == eligibility["evaluation_protocol"],
              "Frozen eligibility evaluation protocol differs from protected input contract")
    expected = partition(original, media["assets"], protected)
    rows = c.checked(pool["records_json"])
    exclusions = c.checked(pool["protected_exclusions"])
    verify_partition(original, rows, pool["source_indices"], pool["record_ids"], exclusions["excluded"], expected)
    c.require(pool["count"] == len(rows) == authority["eligible_N"] and
              authority["K"] == budgets(len(rows)) and authority["validity_policy"] == VALIDITY_POLICY and
              exclusions["validity_policy"] == VALIDITY_POLICY and
              exclusions["original_count"] == ORIGINAL_COUNT and exclusions["retained_count"] == len(rows) and
              exclusions["retained_indices_sha256"] == row_hash(pool["source_indices"]) and
              exclusions["protected_contract"] == expected_protected_description and
              exclusions["source_counts"] == authority["source_counts"] == expected["source_counts"],
              "Complete partition/source counts/validity/K denominator differs")
    c.require(authority["protected_audit_counts"] == dict(Counter(
        reason["reason"] for value in expected["excluded"] for reason in value["reasons"])),
        "Protected QA/image audit counts differ")
    c.require(authority["physical_media_readiness"] is True and authority["runtime_transport_required"] is False,
              "Cross-site/archive transport requires a separate independently qualified rebind, not this local authority")
    c.require(authority["records_json"] == pool["records_json"] and
              authority["image_manifest"] == pool["image_manifest"] and
              authority["exclusions"] == pool["protected_exclusions"] and
              authority["role_coverage"] == protected.coverage and pool["image_root"] == media["image_root"],
              "Authority outputs/roles/media root differ")
    images = c.checked(pool["image_manifest"])["assets"]
    c.require(set(images) == set(expected["required_eligible_images"]) and
              all(images[name] == media["assets"][name] for name in images), "Exact eligible image closure differs")
    logical = logical_media_closure(media, expected["checked_images"])
    c.require(len(expected["required_eligible_images"]) == authority["eligible_unique_images"] and
              authority["media_closure"] == {**logical, "actual_encoded_and_decoded_RGB_checks": True},
              "Canonical media identity differs from the exact producer's physical receipt")
    if full_media_revalidation:
        c.require(audit_output is not None, "Full media replay requires an explicit registered audit output")
        c.require(verify_images(media, expected["checked_images"]) == authority["media_closure"],
                  "Physical media replay differs from producer evidence")
        c.seal(audit_output, {"schema": "native-authority-explicit-media-reaudit-v1",
                             "status": "PASS_ACTUAL_FULL_MEDIA_REVALIDATION",
                             "pool": description, "eligibility": eligibility_description,
                             "input_authority": pool["input_authority"], "actual": actual,
                             "media_closure": authority["media_closure"],
                             "elapsed_seconds": time.monotonic() - started,
                             "cost_scope": "Actual registered allocation/step; Root also accounts all attempts"})
    else:
        c.require(audit_output is None, "Logical consumer must not publish a physical audit PASS")
    return pool, rows, pool["record_ids"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="JSON descriptor of the pinned original metadata")
    parser.add_argument("--media", required=True, help="JSON descriptor of source-bound media inventory")
    parser.add_argument("--protected", required=True, help="JSON descriptor of the frozen authorized protection contract")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(c.canonical(produce(c.read(args.original), c.read(args.media), c.read(args.protected), args.output)).decode())


if __name__ == "__main__":
    main()
