import hashlib
import io
from pathlib import Path

from . import common as c


def load(config):
    metadata = c.checked(config["metadata_result"])
    c.require(metadata["status"] == "COMPLETE_FULL_TEXTVQA_QA_METADATA_GT_ALIGNMENT_ONLY"
              and metadata["questions"] == c.QUESTIONS and metadata["raw_predictions"] == 0,
              "Actual complete metadata provider required")
    roster = c.checked(metadata["full_QA_roster"])
    media = c.checked(config["media"])
    scope = c.checked(config["frozen_scope"])["banks"]["TextVQA"]
    c.require(config["frozen_scope"]["sha256"] ==
              "63d135d8005ccf899cda7aa385501d55e3238376b7e2313aa7c6d45d09659efb"
              and scope["qa_access"] == "authorized" and scope["metric"] == "TextVQA"
              and scope["components"][0]["source"] == roster["source"]
              and scope["components"][0]["support"] == roster["support"]
              and scope["media"]["sha256"] == config["media"]["sha256"],
              "Exact authorized full-bank scope; no input-only permission expansion")
    c.require(roster["bank"] == "TextVQA" and roster["expected_questions"] == c.QUESTIONS
              and len(roster["records"]) == c.QUESTIONS and roster["full_declared_scope"] is True
              and roster["inference_answers_excluded"] is True,
              "No engineering subset or answer-bearing inference roster")
    source = roster["source"]["sha256"]
    c.require(source == "9f60db6fb89e369263ed676dc26b48d00d4a462e73f0930c09d23394e9dcf3a3"
              and media["metric"] == "TextVQA" and media["edition"] == roster["edition"]
              and media["split"] == roster["split"], "Foreign original QA/media edition")
    keys = set()
    for row in roster["records"]:
        c.require(row["bank"] == "TextVQA" and row["source_sha256"] == source
                  and set(row["native_input"]) == {"question_id", "image", "text"}
                  and row["native_input"]["question_id"] == row["native_id"]
                  and type(row["native_input"]["question_id"]) is type(row["native_id"]),
                  "Original native QA identity/answer exclusion differs")
        c.require(row["semantic_key"] not in keys, "Duplicate semantic QA")
        keys.add(row["semantic_key"])
    used = {row["image_reference"] for row in roster["records"]}
    c.require(used == set(media["images"]) and len(used) == c.IMAGES
              and sum(row["encoded_bytes"] for row in media["images"].values()) == c.ENCODED_BYTES,
              "Exact complete3166-image/916425313-byte bank required")
    return metadata, roster, media


def image(row, media, image_root):
    from PIL import Image
    expected = media["images"][row["image_reference"]]
    relative = Path(row["native_input"]["image"])
    root = Path(image_root).resolve()
    path = root / relative
    c.require(not relative.is_absolute() and ".." not in relative.parts
              and path.resolve().is_relative_to(root), "Unsafe/foreign native image path")
    payload = path.read_bytes()
    c.require(len(payload) == expected["encoded_bytes"]
              and hashlib.sha256(payload).hexdigest() == expected["encoded_sha256"],
              "Actual official encoded image differs: " + str(relative))
    with Image.open(io.BytesIO(payload)) as opened:
        rgb = opened.convert("RGB")
    c.require(rgb.size == (expected["width"], expected["height"])
              and hashlib.sha256(rgb.tobytes()).hexdigest() == expected["rgb_sha256"],
              "Actual official RGB pixels/dimensions differ: " + str(relative))
    return rgb


def verify_all_media(roster, media, image_root):
    seen = set()
    for row in roster["records"]:
        if row["image_reference"] not in seen:
            image(row, media, image_root).close()
            seen.add(row["image_reference"])
    c.require(len(seen) == c.IMAGES, "Incomplete physical full-bank media verification")
    return dict(images=len(seen), encoded_bytes=c.ENCODED_BYTES,
                encoded_and_RGB_verified=True, image_inventory_sha256=c.key(media["images"]))
