"""Narrow source-format parsers; no predictions, judges, models or downloads.

An author question file is NOT self-proving complete. The caller must compare
its entire ordered (component, QA ID, image reference) support with a separately
frozen exact support projection before accepting the bank. Missing/ambiguous
formats fail instead of substituting a similarly named edition.
"""
import csv
import base64
import hashlib
import io
import json
import io as binary_io
from pathlib import PurePosixPath


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(value):
    require(type(value) in (int, str), "Native ID must be string or integer")
    value = str(value)
    require(value and value == value.strip(), "Empty or noncanonical native ID")
    return value


def image_ref(value):
    require(isinstance(value, str) and value and value == value.strip(), "Explicit original image reference required")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value,
            "Image reference must remain relative and inside its official source")
    return value


def question(value):
    require(isinstance(value, str) and value.strip(), "Authorized question text required")
    return value


def json_load(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda unused: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant")))


def author_jsonl(raw, options):
    """Pinned c121 model_vqa_loader/model_vqa question fields, all source rows."""
    require(options.get("qa_access") == "authorized", "Author JSONL contains QA text; access must be explicit")
    result = []
    for line in raw.splitlines():
        require(line.strip(), "Blank row in exact author question file")
        row = json_load(line)
        require(isinstance(row, dict), "Author question row must be an object")
        text_key = options.get("text_field", "text")
        require(text_key in ("text", "question"), "Unsupported author question field")
        result.append({"id": identity(row["question_id"]), "image": image_ref(row["image"]),
                       "question": question(row[text_key])})
    return result


def scienceqa_image(raw, options):
    """c121 model_vqa_science: id + conversations[0].value; image-only support."""
    require(options.get("qa_access") == "authorized", "ScienceQA input access must be authorized")
    rows = json_load(raw)
    require(isinstance(rows, list), "Exact llava_test_CQM-A JSON array required")
    result = []
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("conversations"), list)
                and row["conversations"] and row["conversations"][0]["from"] == "human",
                "ScienceQA native question turn differs")
        if "image" not in row:
            continue  # The declared metric is ScienceQA_image, never the text denominator.
        result.append({"id": identity(row["id"]), "image": image_ref(row["image"]),
                       "question": question(row["conversations"][0]["value"].replace("<image>", "").strip())})
    return result


def _none(value):
    return value is None or (isinstance(value, str) and value.lower() in ("", "nan", "none"))


def mmbench_tsv(raw, options):
    """c121 hint + question + contiguous A..D options, original TSV order.

    Image references are QA indices, NOT guessed COCO IDs. Encoded image bytes
    remain in the exact official TSV; media provider must resolve each index to
    verified encoded/RGB identity. No implicit short-string image-ID pointers.
    """
    require(options.get("qa_access") == "authorized", "MMBench input access must be authorized")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t")
    require(reader.fieldnames and {"index", "question", "image", "hint", "A", "B", "C", "D"}
            <= set(reader.fieldnames), "Exact intended MMBench TSV fields required")
    require(len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate TSV column")
    result = []
    for row in reader:
        require(None not in row and None not in row.values(), "Malformed TSV record")
        key = identity(row["index"])
        text = question(row["question"])
        if not _none(row["hint"]):
            text = row["hint"] + "\n" + text
        ended = False
        for option in "ABCD":
            value = row[option]
            if _none(value):
                ended = True
            else:
                require(not ended, "Noncontiguous MMBench options differ from author input")
                text += "\n" + option + ". " + value
        require(row["image"] and len(row["image"]) > 64,
                "Unresolved MMBench image alias; requires explicit reviewed source adaptation")
        encoded = base64.b64decode(row["image"], validate=True)
        require(encoded, "Empty official inline encoded image")
        result.append({"id": key, "image": "tsv-index/" + key,
                       "question": text, "inline_encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                       "inline_encoded_bytes": len(encoded)})
    return result


def reserved_id_projection(raw, options):
    """Consume only an already authorized id/imageId projection, never raw GQA QAs."""
    require(options.get("qa_access") == "image_only", "Reserved projection must remain image-only")
    from .roles_projection import Reader
    reader = Reader(binary_io.BytesIO(raw), len(raw))
    result = []
    reader.expect("[")
    reader.ws()
    if reader.peek() == 93:
        reader.take()
        return result
    while True:
        row = {}
        def field(name):
            require(name in ("id", "imageId"), "Reserved projection must contain only ID fields")
            reader.ws()
            row[name] = reader.string(True) if reader.peek() == 34 else reader.atom(True)
        reader.object(field)
        require(set(row) == {"id", "imageId"}, "Both exact ID fields required")
        result.append({"id": identity(row["id"]), "image": identity(row["imageId"])})
        reader.ws()
        token = reader.take()
        if token == 93:
            break
        require(token == 44, "Malformed reserved projection array")
    reader.ws()
    require(reader.peek() is None, "Trailing reserved projection payload")
    return result


def nocaps_images(raw, options):
    """Official NoCaps validation images membership; captions are not consumed."""
    require(options.get("qa_access") == "image_only", "NoCaps protection is image-only")
    values = json_load(raw)
    require(isinstance(values, dict) and isinstance(values.get("images"), list),
            "Exact intended NoCaps validation images JSON required")
    result = []
    for row in values["images"]:
        require(isinstance(row, dict), "Malformed NoCaps image record")
        key = identity(row["id"])
        # URLs/filenames are not interpreted as another dataset's source ID.
        result.append({"id": key, "image": key})
    return result


PARSERS = {
    "c121-author-jsonl-v1": author_jsonl,
    "c121-scienceqa-image-v1": scienceqa_image,
    "c121-mmbench-tsv-v1": mmbench_tsv,
    "reserved-id-image-projection-v1": reserved_id_projection,
    "nocaps-validation-images-v1": nocaps_images,
}


def parse(raw, kind, options):
    require(kind in PARSERS, "Source adapter unavailable; exact raw format must be reviewed before use")
    rows = PARSERS[kind](raw, options)
    require(rows, "Empty official support is not a complete bank")
    seen = set()
    for row in rows:
        require(row["id"] not in seen, "Duplicate QA/image key in source component")
        seen.add(row["id"])
    return rows
