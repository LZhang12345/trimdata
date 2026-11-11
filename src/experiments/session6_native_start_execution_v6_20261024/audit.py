from contextlib import contextmanager
import hashlib

from . import common as c
from . import inputs
from experiments.session6_official_eval_v2_20261024.journal import Journal


def tensor_digest(value):
    import torch
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(c.canonical([str(value.dtype), list(value.shape)]))
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def expected_inputs(row, loaded, media, image_root):
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    tokenizer, model, processor, _ = loaded
    c.require(not model.config.mm_use_im_start_end and not model.config.mm_use_im_patch_token,
              "Frozen START image-token flags differ")
    conversation = conv_templates["vicuna_v1"].copy()
    conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + row["native_input"]["text"])
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    tokens = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    rgb = inputs.image(row, media, image_root)
    try:
        pixels = process_images([rgb], processor, model.config)[0].unsqueeze(0).to(torch.float16)
        return dict(prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    input_ids_sha256=tensor_digest(tokens), input_token_count=tokens.shape[-1],
                    pixel_values_sha256=tensor_digest(pixels), image_size=list(rgb.size))
    finally:
        rgb.close()


class Recorder:
    def __init__(self, rows, loaded, media, image_root, function, context):
        self.rows, self.loaded, self.media = rows, loaded, media
        self.image_root, self.function, self.context = image_root, function, context
        self.position, self.pending = 0, None

    def take(self):
        c.require(self.pending is not None, "No actual generation to commit")
        value, self.pending = self.pending, None
        self.position += 1
        return value

    @contextmanager
    def installed(self):
        model = self.loaded[1]
        original = model.generate

        def generate(*args, **kwargs):
            c.require(self.pending is None and self.position < len(self.rows) and len(args) == 1,
                      "Native generate/semantic-row order differs")
            required = {"images", "image_sizes", "do_sample", "temperature", "top_p",
                        "num_beams", "max_new_tokens", "use_cache"}
            c.require(set(kwargs) == required and kwargs["do_sample"] is False
                      and kwargs["temperature"] == 0 and kwargs["top_p"] is None
                      and kwargs["num_beams"] == 1 and kwargs["max_new_tokens"] == 128
                      and kwargs["use_cache"] is True, "Actual native loader generation kwargs differ")
            row = self.rows[self.position]
            proof = expected_inputs(row, self.loaded, self.media, self.image_root)
            c.require(tensor_digest(args[0]) == proof["input_ids_sha256"]
                      and tensor_digest(kwargs["images"]) == proof["pixel_values_sha256"]
                      and [list(size) for size in kwargs["image_sizes"]] == [proof["image_size"]],
                      "Actual native token/processed-pixel/image-size identity differs")
            output = original(*args, **kwargs)
            token_ids = output.detach().cpu().tolist()
            c.require(len(token_ids) == 1 and token_ids[0], "Native generation returned no token sequence")
            eos = model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            ended_eos = token_ids[0][-1] in eos_ids
            c.require(ended_eos or len(token_ids[0]) >= kwargs["max_new_tokens"],
                      "Native generation stopped for an unqualified reason")
            self.pending = dict(function_sha256=c.key(self.function), actual=self.context,
                native_generation=dict(semantic_key=row["semantic_key"], **proof,
                    output_ids=token_ids[0], raw_output_token_count=len(token_ids[0]),
                    output_count_is_raw_native_sequence_not_assumed_new_token_count=True,
                    stop_reason="EOS" if ended_eos else "LENGTH",
                    decoded=self.loaded[0].batch_decode(output, skip_special_tokens=True)[0].strip(),
                    explicit_kwargs={k: v for k, v in kwargs.items() if k not in ("images", "image_sizes")},
                    image_reference=row["image_reference"]))
            return output

        model.generate = generate
        try:
            yield
            c.require(self.pending is None, "Native generation was not durably committed")
        finally:
            model.generate = original


class AuditedJournal(Journal):
    def append(self, semantic_id, raw, audit):
        generation = audit["native_generation"]
        c.require(generation["semantic_key"] == semantic_id and raw["text"] == generation["decoded"],
                  "Native raw answer differs from audited actual generated IDs")
        super().append(semantic_id, raw, audit)


def validate_entry(entry, row, function):
    raw, record = entry["raw"], entry["audit"]
    generation = record["native_generation"]
    key = row["semantic_key"]
    c.require(entry["semantic_key"] == record["semantic_key"] == generation["semantic_key"] == key
              and record["function_sha256"] == c.key(function)
              and type(raw["question_id"]) is type(row["native_id"])
              and raw["question_id"] == row["native_id"]
              and raw["prompt"] == row["native_input"]["text"]
              and isinstance(raw["text"], str) and raw["text"] == generation["decoded"]
              and generation["image_reference"] == row["image_reference"],
              "Accepted native raw/audit/model/semantic/image identity differs")
    binding = record["generation_config_binding"]
    c.require(binding == dict(model_generation_config_sha256=c.key(function["model_generation_config"]),
                             native_num_beams_keyword_present=True, effective_num_beams=1),
              "Accepted answer has a different actual inherited generation configuration")
    c.require(type(generation["input_token_count"]) is int and generation["input_token_count"] > 0
              and all(isinstance(generation[name], str) and len(generation[name]) == 64
                      and all(ch in "0123456789abcdef" for ch in generation[name])
                      for name in ("prompt_utf8_sha256", "input_ids_sha256", "pixel_values_sha256")),
              "Accepted prompt/token/processed-pixel commitments are missing or invalid")
    output = generation["output_ids"]
    c.require(isinstance(output, list) and output and all(type(value) is int for value in output)
              and generation["raw_output_token_count"] == len(output),
              "Accepted raw native generated IDs differ")
    eos = function["model_generation_config"]["eos_token_id"]
    eos_ids = eos if isinstance(eos, list) else [eos]
    expected_stop = "EOS" if output[-1] in eos_ids else "LENGTH"
    c.require(generation["stop_reason"] == expected_stop
              and (expected_stop == "EOS" or len(output) >= 128)
              and generation["explicit_kwargs"] == dict(do_sample=False, temperature=0,
                    top_p=None, num_beams=1, max_new_tokens=128, use_cache=True),
              "Accepted native decoding policy or stop accounting differs")
