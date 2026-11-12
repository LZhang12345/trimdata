"""BENCH4 bank protocols for Tool1 (stdlib only; nothing here opens or carries labels).

Banks (one GPU job per model runs them in ORDER, each with its own resume-stable per-QA journal):
  mmmu-val        MMMU validation, 900 Q / 30 subjects (HF MMMU/MMMU@876ce5cb). Tool1 author-delegated
                  protocol: official MMMU repository construct_prompt with configs/llava1.5.yaml, official
                  process_single_sample image rule (image_1; no question in this revision has more than one
                  image inside its options), LLaVA c121 vicuna_v1 conversation and image tokenization, greedy
                  decoding with the repository's 128-token horizon; answers parsed by the official
                  parse_multi_choice_response / parse_open_response in the scoring step (score4.py).
  llava-wild      LLaVA-Bench-in-the-Wild, 60 Q: LLaVA c121 model_vqa (vicuna_v1, temperature 0, 1024 tokens).
                  Scored only by the separate registered GPT judge step (judge4.py / score_step4.py --kind judge).
  mm-vet          MM-Vet v1, 218 Q: LLaVA c121 model_vqa exactly as scripts/v1_5/eval/mmvet.sh. Scored only by the
                  separate registered GPT judge step with the official MM-Vet grader prompt.
  mmbench-en-dev  MMBench dev 20230712 (EN), LLaVA c121 model_vqa_mmbench (single-pred prompt, vicuna_v1,
                  temperature 0, 1024 tokens). The generation roster is the union of every TSV row (the official
                  circular variants, index mod 1e6 groups) and the explicit LLaVA --all-rounds left rotations of each
                  base row (index < 1e6), deduplicated on exact (image bytes, prompt); scoring reports official
                  dev-TSV CircularEval, explicit all-rotation CircularEval and single-pass base-row accuracy.

Questions/images come only from the on-node BENCH4 input inventory (label-free QUESTIONS.jsonl + content-addressed
image files, cross-checked against Root's frozen supports and RGB proofs). Labels live in separate LABELS files
that only the scoring step opens.
"""

import ast
import copy
import hashlib
import json
from pathlib import PurePosixPath
import re


HOME = "/home/luyangz/data_selection_multimodal"
SRC = HOME + "/native_campaign_v3_20261023/src"
OFF = "/data/hf_cache/datasets/dsmm_native_official_inputs_v1"
UD = "/data/user_data/luyangz"
INPUT_ROOT = "/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-bench4-inputs-v1-20261104"
LLAVA_REVISION = "c121f0432da27facab705978f83c4ada465e46fd"
MMMU_REVISION = "876ce5cb130f7f7e290ce4d9984357737d4db5cf"
MMMU_CODE_COMMIT = "268471d0d488258990025331c7528359c324aa25"
ROLE = "TOOL1_BENCH4_EXISTING_STUDENT_EVALUATION"
ORDER = ("mmmu-val", "llava-wild", "mm-vet", "mmbench-en-dev")
FORBIDDEN_FIELDS = ("answer", "answers", "label", "labels", "gt", "ground_truth", "gt_content", "correct_choice",
                    "explanation", "multiple_choice_answer")
IMAGE_TOKEN = "<image>"
MMBENCH_SUFFIX = "Answer with the option's letter from the given choices directly."
MMBENCH_OPTIONS = ("A", "B", "C", "D")
MMBENCH_CIRCULAR_BASE = 1000000
# PyYAML parse of the pinned MMMU configs/llava1.5.yaml (double-quoted scalars fold blank lines to one newline).
MMMU_CONFIG = {
    "task_instructions": "",
    "multi_choice_example_format": "{}\n{}\nAnswer with the option's letter from the given choices directly.",
    "short_ans_example_format": "{}\nAnswer the question using a single word or phrase.",
    "temperature": 0,
}
MMMU_SUBJECTS = (
    ("Accounting", 1535386, "f1bedfc8ad70d24f8c1daa50a309d9b3b88ea14caea60299b648c69049fa870f"),
    ("Agriculture", 119223107, "f95765c549ebaeedd89c14bfe39d1deefcb9ec6b8b082f5945ffc547f2c4e20c"),
    ("Architecture_and_Engineering", 726950, "09792c72c949565ba9378366562818d9687eba5b4f6fc3e84482a8862cf4dd9c"),
    ("Art", 29939738, "d86be1080d7a33a76e9b98f0df0d7ebacbd37b1c29927bbf592489cc4b996c57"),
    ("Art_Theory", 29783856, "ba4e709ace78695df3b2ed8859e7cfb36e3cd9c60abd5a0fade9800653b695d6"),
    ("Basic_Medical_Science", 4131430, "577102f6f742471a947a82579df0ffa8b7bfd1b10dc2eadbf90f9ea5ff0cddd4"),
    ("Biology", 8494798, "644e5b10eb90689a0e41a7c738526113551f59ab5e66c44a870e6bca52d39e08"),
    ("Chemistry", 1522870, "ff62af1b99c6f60d2b2228bf438152d9e44c06e42455d5eb5e229954aa95b00b"),
    ("Clinical_Medicine", 10888251, "2c3d29a6c838e279974731ea92fa459e15107f9213970a9a4f9312c1f102bd58"),
    ("Computer_Science", 2078701, "73c8ef8ba23a050b85c524363614801cdc73db36ad7a7348643006766da037f5"),
    ("Design", 16227878, "c69b48ecde9ad80bb8b530dec4da99ae051b17f90abefd2b98230313580c5b7f"),
    ("Diagnostics_and_Laboratory_Medicine", 37089865,
     "cefb7363510a05284dd8a0b12cb7cc7f20843c65d6b7101eefe3929d4061a592"),
    ("Economics", 1424097, "1a6b0ef6edf3eb76f2fae37cbff28d6723eec4bd94d4b3d6e676aa4ea924d95b"),
    ("Electronics", 644538, "558d6970c95c9961c63da8b1ebd7a885524059a83838fa67e6be8c6522a536ba"),
    ("Energy_and_Power", 1646107, "46f8d320be746b990145c4e2bf6b88fc493f0e8e322759ee2a9e2fcc8ee404b5"),
    ("Finance", 1073434, "75c773f16c39b60ff7ff2ad8dd9fc82986418cc4bee9864c10ec4098adefc555"),
    ("Geography", 6677774, "536505f8dc6199713effcd7cf9b8ae2da2325e51ec17d0fcfabc75956965701c"),
    ("History", 8430352, "fe6ba5fa13133d43afbd63abfe75b49ac45c5de566ac896fbc370cb1492c8f82"),
    ("Literature", 14246788, "3ad02815b3af320c574b254c7a2134f31288a360c0168035ee051c8946cb8b1b"),
    ("Manage", 3139783, "838ecee5c3c9708b3fb70e414b83bfee35725814290908a853b24d06bbd832fe"),
    ("Marketing", 1361340, "0b4854d8a509e8e0f401255d5788fd83e489e4d5d5207826856dcffa1157feb2"),
    ("Materials", 2309857, "6881364f70c6911f0610005ad5aaf10f4394a7f578ead566e528b574da534e54"),
    ("Math", 1447995, "ebe73665a3b1db2bfd33ae67a00efcf86b6d817463a12d995e30716d69aae8e7"),
    ("Mechanical_Engineering", 876772, "8381e04c04dc8e0b3a21404e251a1fc2deb52d3f5b2485e5e8bc99e4907d21d7"),
    ("Music", 9363650, "132f8e3965276788f1b5388c39565d5475b8eb61af3142522be446e9a131af8f"),
    ("Pharmacy", 1551086, "93bc3de041d8f69b9e7cd29886ad68f26ac5b11bdf221cabb5f84b8b8d8ca6a9"),
    ("Physics", 1117086, "649a7c85ae9f1aa0762059eca0c082e3862f669f07a1283c36ada31def2c391d"),
    ("Psychology", 4311520, "16101fe448e07ce654d95b5ef5793988fb6e7f5fe42b94575c33c9ffbd4ad919"),
    ("Public_Health", 1510182, "c8525f7af4a47ffd3c2959816147911203782c8e1116267f4e6e54026ef0f923"),
    ("Sociology", 18459557, "a38e31036b6cfb6df82c069f2715b8df0158969405e86a08c1732c334638ac32"),
)


class BankError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BankError(message)


def ref(path, size, digest):
    return {"path": path, "bytes": size, "sha256": digest}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def key(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def hex64(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


FROZEN_SCOPE = ref(SRC + "/collaboration/root-input-contract-assembly-v1-20261024/ROOT_FROZEN_SCOPE_V1.json",
                   29471, "63d135d8005ccf899cda7aa385501d55e3238376b7e2313aa7c6d45d09659efb")
NATIVE_ASSETS = ref(HOME + "/experiments/published_foundation_20261022/runtime-v1/outputs/assets/COMPLETE.json",
                    10819, "b6fe9f87e7fc20809f6176ad75014469e16fdfbc2120fa83cd1c512850bed027")
START_TEMPLATE = ref(SRC + "/collaboration/root-babel-start-evaluation-v2-20261025/CONFIG_V2_VizWiz.json",
                     28308, "e507f51fee6f78c69336e601d66a6df7163243086d0836801c91cc6f5b9b1851")
QUALIFIED = UD + "/official_input_qualification_20261024/v1/"
MMMU_QUALIFIED = UD + "/mmmu_input_qualification_20261024/v1/"
START_DECODING = {"conv_mode": "vicuna_v1", "do_sample": False, "max_new_tokens": 128, "num_beams": 1,
                  "temperature": 0, "top_p": None, "use_cache": True}


def decoding(max_new_tokens):
    return {"conv_mode": "vicuna_v1", "do_sample": False, "temperature": 0, "top_p": None, "num_beams": 1,
            "max_new_tokens": max_new_tokens, "use_cache": True, "image_sizes_passed": True}


PROTOCOLS = {
    "mmmu-val": {
        "bank": "MMMU_validation", "kind": "mmmu", "edition": "MMMU_validation@" + MMMU_REVISION,
        "split": "validation", "items": 900, "images": None, "subjects": 30, "decoding": decoding(128),
        "suffix": None, "image_mode": "convert_rgb",
        "prompt_policy": "OFFICIAL_MMMU_CONSTRUCT_PROMPT_LLAVA15_YAML_IMAGE1_PROCESS_SINGLE_SAMPLE_C121_VICUNA_V1",
        "sources": {"parquets": [ref(OFF + "/files/mmmu/%s/validation-00000-of-00001.parquet" % s, b, h)
                                 for s, b, h in MMMU_SUBJECTS]},
        "root_support": ref(MMMU_QUALIFIED + "FROZEN_SUPPORT.json", 626966,
                            "698f7a59e69d043ccaac0a52452ec1025b67cc2742546948d4986e235e8cf339"),
        "root_provider": ref(MMMU_QUALIFIED + "COMPLETE.json", 4048,
                             "b7e316b6c8a2c4040047b1fd1b9e50327f4e4c0e512f36e019dbfac0da775873"),
        "root_rgb_proof": ref(MMMU_QUALIFIED + "RGB_PROOF.json", 370936,
                              "3fbb2eaa8bed7937bd6fa42789080e50c791e5197200786698722bf2a9721702"),
        "output_envelope": 16 << 20,
        "scoring": "OFFICIAL_MMMU_PARSE_AND_EVAL_ONLY_PER_SUBJECT_DOMAIN_OVERALL_IN_REGISTERED_SCORING_STEP",
        "judge_required": False,
        "disclosures": [
            "Decoding is greedy (do_sample False, num_beams 1, max_new_tokens 128); the official run_llava.py "
            "uses do_sample True, temperature 1, num_beams 5 although configs/llava1.5.yaml sets temperature 0.",
            "Images follow LLaVA c121 (PIL convert('RGB') + process_images pad); run_llava.py calls "
            "image_processor.preprocess on the raw image (center crop, no pad).",
            "Responses are LLaVA c121 decodes of the new tokens only (c121 generate returns only new tokens; the "
            "official engine slices off input_token_len for older LLaVA).",
            "options are parsed with ast.literal_eval (official construct_prompt uses eval) of the same string.",
            "Every validation question has image_1 and no question has more than one image inside its options in "
            "this revision, so the official random-choice branch (whose source has a `from random import random` "
            "bug) never fires; 43 questions reference several images and, as in the official engine, only "
            "image_1 is fed while '<image k>' placeholders stay in the text.",
            "Parse fallbacks consume random.Random(42) in official dataset order (eval_utils/run_llava seed 42), "
            "replayed deterministically by the scoring step.",
        ],
    },
    "llava-wild": {
        "bank": "LLaVA_Bench_in_the_Wild", "kind": "freeform",
        "edition": "LLaVA-Bench-in-the-Wild@9773b6fd2be88d4ced480dcd9bfd4ce76e47f79b", "split": "questions",
        "items": 60, "images": 24, "decoding": decoding(1024), "suffix": None, "image_mode": "convert_rgb",
        "prompt_policy": "VERBATIM_LLAVA_C121_MODEL_VQA_LLAVABENCH_SH",
        "sources": {"questions": ref(OFF + "/files/llava-bench-in-the-wild/questions.jsonl", 8246,
                                     "a80b369a4b629f79f8c0f9a2aa9b565f0cfe34734d3bcdd30cb4e38712c07e21"),
                    "image_dir": OFF + "/files/llava-bench-in-the-wild/images"},
        "root_support": ref(QUALIFIED + "LLaVA_Bench_in_the_Wild/FROZEN_SUPPORT.json", 14440,
                            "aa21c552ca08e2d91714c60fb5bd0e976ce324212ba66962c52d6afdc9349544"),
        "root_provider": ref(QUALIFIED + "LLaVA_Bench_in_the_Wild/COMPLETE.json", 978,
                             "78b4c6855affc869512ba42f4d379ea6e9de57edee35c3f4f976bfc698b0c6c9"),
        "root_media": ref(QUALIFIED + "LLaVA_Bench_in_the_Wild/MEDIA.json", 16265,
                          "5560282617190f5e5e050888cfd84fa2c46cfca3d79d3344a57e17899f35588e"),
        "media_component": "official_pinned", "output_envelope": 8 << 20,
        "scoring": "LLAVA_C121_EVAL_GPT_REVIEW_BENCH_RELATIVE_SCORE_IN_REGISTERED_JUDGE_STEP",
        "judge_required": True, "disclosures": ["Raw answers omit LLaVA's random answer_id/model_id fields."],
    },
    "mm-vet": {
        "bank": "MM_Vet", "kind": "freeform", "edition": "MM-Vet_v1", "split": "v1", "items": 218, "images": 200,
        "decoding": decoding(1024), "suffix": None, "image_mode": "convert_rgb",
        "prompt_policy": "VERBATIM_LLAVA_C121_MODEL_VQA_MMVET_SH",
        "sources": {"questions": ref(OFF + "/input_members_v2/mm-vet/llava-mm-vet.jsonl", 29575,
                                     "ac90c438b58fa867f3e9ff5d1cb0c27be4dd35f167ea129ccceb246e913a65c6"),
                    "archive": ref(OFF + "/files/mmvet-v1/mm-vet.zip", 66540341,
                                   "ff85408b7f382be146195b10ff5c7f995fbf297e92ea1132b61e438c8967116a"),
                    "archive_image_prefix": "mm-vet/images/"},
        "root_support": ref(QUALIFIED + "MM_Vet/FROZEN_SUPPORT.json", 48532,
                            "9fe4cd7ddc12a841fb1a21c2558f8d90a81f2cd10cc5597a6efe7c19bef9efea"),
        "root_provider": ref(QUALIFIED + "MM_Vet/COMPLETE.json", 913,
                             "ae1c5f83607297152774c8c2746e64c56041fb9c4e07c21a6af45ed12d4e00b9"),
        "root_media": ref(QUALIFIED + "MM_Vet/MEDIA.json", 128217,
                          "2ee605e3c0ab8a0c8228c2f4c0459c59e770e7442f87060c620704b0ee69baeb"),
        "media_component": "native_c121", "output_envelope": 16 << 20,
        "label_member": {"name": "mm-vet/mm-vet.json", "bytes": 136500,
                         "sha256": "6eced3121b9865098ea72aca233ca5ff7fd7b81374494d878815b9f856442d8e"},
        "scoring": "OFFICIAL_MMVET_GRADER_CAPABILITY_AND_TOTAL_IN_REGISTERED_JUDGE_STEP",
        "judge_required": True, "disclosures": ["Raw answers omit LLaVA's random answer_id/model_id fields."],
    },
    "mmbench-en-dev": {
        "bank": "MMBench_EN", "kind": "mmbench", "edition": "MMBench_dev_20230712", "split": "dev",
        "tsv_rows": 4377, "groups": 1176, "items": None, "images": None, "decoding": decoding(1024),
        "suffix": MMBENCH_SUFFIX, "image_mode": "native_base64_no_convert",
        "prompt_policy": "LLAVA_C121_MODEL_VQA_MMBENCH_SINGLE_PRED_TSV_ROWS_UNION_EXPLICIT_ALL_ROUNDS",
        "sources": {"tsv": ref(OFF + "/files/mmbench/mmbench_dev_20230712.tsv", 137646313,
                               "52d45f29876e56138d3d4feb169dc9a7323f6ed2ee3f439ae7e4af94192b3324")},
        "root_support": ref(QUALIFIED + "MMBench_EN/FROZEN_SUPPORT.json", 1008847,
                            "ed7605e1809ed04e0c95fdb3de47c3d10bad419e67b3cae8ac86eb50c40278d2"),
        "root_provider": ref(QUALIFIED + "MMBench_EN/COMPLETE.json", 936,
                             "5c29b089b8741d75c3e129b3749c08548123cb99eb51b94331979d3e2a2a6e65"),
        "root_media": ref(QUALIFIED + "MMBench_EN/MEDIA.json", 2910880,
                          "e999cb0d739ebe61bd71b4b88f94d94da8a7a3c3a6671e14571d2c9dc81f32b5"),
        "media_component": "native_c121", "output_envelope": 64 << 20,
        "scoring": "LOCAL_DEV_CIRCULAR_EVAL_TSV_GROUPS_AND_EXPLICIT_ROTATIONS_PLUS_PER_ABILITY_IN_REGISTERED_STEP",
        "judge_required": False,
        "disclosures": [
            "Dev split scored locally (labels in the dev TSV); not the MMBench test server.",
            "TSV parsed with pandas.read_table exactly as model_vqa_mmbench (NaN/'nan'/'none' options end the "
            "option list); images decoded as load_image_from_base64 (no RGB conversion), as c121 does.",
            "Option-letter extraction uses the deterministic VLMEvalKit can_infer stage only (no ChatGPT stage); "
            "unparsable answers count as wrong and are reported.",
            "Raw answers omit LLaVA's random answer_id/model_id fields.",
        ],
    },
}

for _name, _value in PROTOCOLS.items():
    _value.update(name=_name, frozen_scope=copy.deepcopy(FROZEN_SCOPE), local_scoring=not _value["judge_required"],
                  llava_revision=LLAVA_REVISION)


def protocol(name):
    require(isinstance(name, str) and name in PROTOCOLS, "Unknown BENCH4 bank: %r" % (name,))
    return copy.deepcopy(PROTOCOLS[name])


def check_bank_list(names):
    require(isinstance(names, (list, tuple)) and list(names) == list(ORDER),
            "Exact reviewed BENCH4 bank order required: %s" % (ORDER,))
    return list(names)


# ----------------------------------------------------------------------------------------------- prompt logic
def pandas_none(value):
    """LLaVA c121 model_vqa_mmbench.is_none on a pandas-read cell (NaN floats included)."""
    if value is None:
        return True
    if type(value) is float and value != value:
        return True
    if type(value) is str and value.lower() in ("nan", "none"):
        return True
    return False


def mmbench_options(row):
    """c121 get_options(row, all_options): stops at the first missing option."""
    options = []
    for letter in MMBENCH_OPTIONS:
        value = row[letter]
        if pandas_none(value):
            break
        options.append(value)
    return options


def mmbench_cur_prompt(question, hint, options):
    """c121: hint + '\\n' + question when hint is present, then '\\n<letter>. <option>' per option."""
    require(isinstance(question, str) and isinstance(options, list) and 1 <= len(options) <= 4
            and all(isinstance(o, str) for o in options), "MMBench question/options must be strings")
    text = question
    if not pandas_none(hint):
        require(isinstance(hint, str), "MMBench hint must be a string")
        text = hint + "\n" + text
    for letter, option in zip(MMBENCH_OPTIONS[:len(options)], options):
        text = text + "\n" + letter + ". " + option
    return text


def mmbench_rounds(options):
    """c121 --all-rounds: round r shows options left-rotated by r and cur_option_char rotated the same way."""
    chars = list(MMBENCH_OPTIONS[:len(options)])
    rounds, current, current_chars = [], list(options), list(chars)
    for index in range(len(options)):
        rounds.append((index, list(current), list(current_chars)))
        current = current[1:] + current[:1]
        current_chars = current_chars[1:] + current_chars[:1]
    return rounds


def mmmu_options(text):
    """Official construct_prompt evaluates sample['options']; literal_eval of a list literal is identical."""
    value = ast.literal_eval(text)
    require(isinstance(value, list) and all(isinstance(o, str) for o in value), "MMMU options must be a str list")
    return value


def mmmu_prompt(question, options_text, question_type, config=MMMU_CONFIG):
    """Official MMMU utils/data_utils.construct_prompt (label-free part) with the llava1.5.yaml configuration."""
    require(isinstance(question, str) and isinstance(question_type, str), "MMMU question/type must be strings")
    options = mmmu_options(options_text)
    if question_type == "multiple-choice":
        start_chr, prediction_range, index2ans, example = "A", [], {}, ""
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt = config["multi_choice_example_format"].format(question, example)
    else:
        prediction_range, index2ans = None, None
        empty_prompt = config["short_ans_example_format"].format(question)
    if config["task_instructions"]:
        final = config["task_instructions"].strip() + "\n\n" + empty_prompt
    else:
        final = empty_prompt
    return final, prediction_range, index2ans


def mmmu_option_images(options_text):
    """Official process_single_sample: count <img='...'> references inside the options."""
    return sum(len(re.findall("<img='(.*?)'>", option)) for option in mmmu_options(options_text))


def llava_query(cur_prompt, mm_use_im_start_end, suffix=None):
    """c121 eval scripts: image token line, then the prompt (and the single-pred suffix for MMBench)."""
    require(mm_use_im_start_end is False, "LLaVA-1.5 uses no image start/end tokens")
    qs = IMAGE_TOKEN + "\n" + cur_prompt
    if suffix is not None:
        qs = qs + "\n" + suffix
    return qs


# --------------------------------------------------------------------------------------------- roster checks
def _image(value):
    require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256", "rgb_sha256", "width", "height"},
            "Exact image descriptor required")
    path = value["path"]
    require(isinstance(path, str) and path.startswith("/") and "\0" not in path and "\\" not in path
            and all(part not in ("", ".", "..") for part in path[1:].split("/")), "Literal absolute image path")
    require(path.startswith(INPUT_ROOT + "/") or path.startswith(OFF + "/files/llava-bench-in-the-wild/images/"),
            "Images come only from the BENCH4 input inventory or the pinned Wild image directory")
    require(type(value["bytes"]) is int and value["bytes"] > 0 and hex64(value["sha256"])
            and hex64(value["rgb_sha256"]) and type(value["width"]) is int and type(value["height"]) is int
            and value["width"] > 0 and value["height"] > 0, "Image identity fields required")
    return value


def _no_labels(value):
    if isinstance(value, dict):
        require(not any(field in value for field in FORBIDDEN_FIELDS), "Label fields must never enter generation")
        for item in value.values():
            _no_labels(item)
    elif isinstance(value, list):
        for item in value:
            _no_labels(item)


def validate_questions(name, rows, expected_items=None):
    """Exact label-free generation roster for one bank; returns order-sensitive commitments."""
    value = protocol(name)
    count = value["items"] if value["items"] is not None else expected_items
    require(type(count) is int and count > 0 and isinstance(rows, list) and len(rows) == count,
            "Exact full generation roster count required")
    keys, images, names = [], set(), set()
    for row in rows:
        require(isinstance(row, dict) and row.get("bank") == value["bank"] and row.get("protocol") == name,
                "Roster row bank/protocol differs")
        _no_labels(row)
        require(hex64(row.get("semantic_key")) and hex64(row.get("source_sha256"))
                and type(row.get("original_row_index")) is int and row["original_row_index"] >= 0
                and type(row.get("round_id")) is int and row["round_id"] >= 0, "Semantic identity fields required")
        require(row["semantic_key"] == semantic_key(value["bank"], row["source_sha256"], row["original_row_index"],
                                                    row.get("native_id"), row["round_id"]), "Forged semantic key")
        native = row.get("native_input")
        require(isinstance(native, dict), "Native input required")
        image = _image(row.get("image"))
        images.add(image["sha256"])
        kind = value["kind"]
        if kind == "freeform":
            names.add(native.get("image"))
            require(set(native) == {"question_id", "image", "text"} and isinstance(native["text"], str)
                    and native["text"].strip() and type(row.get("native_id")) is type(native["question_id"])
                    and row["native_id"] == native["question_id"] and row["round_id"] == 0,
                    "model_vqa native input must be exactly question_id/image/text")
            require(row["cur_prompt"] == native["text"], "model_vqa cur_prompt is the question text")
        elif kind == "mmmu":
            require(set(native) == {"id", "subject", "question_type", "question", "options", "num_images"}
                    and row["native_id"] == native["id"] and row["round_id"] == 0, "MMMU native input fields")
            final, choices, index2ans = mmmu_prompt(native["question"], native["options"], native["question_type"])
            require(row["cur_prompt"] == final and row.get("all_choices") == choices
                    and row.get("index2ans") == index2ans, "MMMU prompt must equal official construct_prompt")
            require(mmmu_option_images(native["options"]) <= 1, "Option-image questions need the random branch")
        else:
            require(set(native) == {"index", "question", "hint", "options", "option_char", "round_id"}
                    and type(native["index"]) is int and row["native_id"] == native["index"]
                    and native["round_id"] == row["round_id"], "MMBench native input fields")
            require(row["cur_prompt"] == mmbench_cur_prompt(native["question"], native["hint"], native["options"])
                    and native["option_char"] == list(MMBENCH_OPTIONS[:len(native["options"])])[
                        native["round_id"]:] + list(MMBENCH_OPTIONS[:len(native["options"])])[:native["round_id"]],
                    "MMBench prompt must equal c121 model_vqa_mmbench construction")
            serves = row.get("serves")
            require(isinstance(serves, dict) and set(serves) == {"tsv_rows", "explicit_rounds"}
                    and (serves["tsv_rows"] or serves["explicit_rounds"]), "MMBench item must serve a scored unit")
        keys.append(row["semantic_key"])
    require(len(set(keys)) == len(keys), "Duplicate semantic QA key")
    if value["images"] is not None:
        require(len(names) == value["images"], "Exact unique native image count required: %d" % value["images"])
    return {"protocol": name, "bank": value["bank"], "items": len(rows), "unique_images": len(images),
            "ordered_semantic_keys_sha256": key(keys),
            "ordered_native_inputs_sha256": key([[r["native_input"], r["image"]["sha256"]] for r in rows]),
            "answer_or_label_fields_present": False}


def semantic_key(bank, source_sha256, original_row_index, native_id, round_id=0):
    """Byte-identical to session6_official_eval_v2 journal.semantic_key (the frozen Journal re-derives it)."""
    require(hex64(source_sha256) and type(original_row_index) is int and original_row_index >= 0
            and type(native_id) in (str, int) and type(round_id) is int and round_id >= 0, "Semantic identity")
    return key([bank, source_sha256, original_row_index, type(native_id).__name__, native_id, round_id])


def journal_function(name, cell, model, roster_summary, sources_sha256, inputs):
    """Resume-stable per-bank journal identity: no job, node, attempt or route fields."""
    value = protocol(name)
    require(isinstance(model, dict) and hex64(model.get("parameter_sha256"))
            and hex64(model.get("auxiliary_identity_sha256")) and isinstance(model.get("generation_config"), dict)
            and type(model.get("adapter_active")) is bool, "Exact loaded model identity required")
    require(isinstance(roster_summary, dict) and roster_summary.get("protocol") == name, "Roster summary mismatch")
    require(hex64(sources_sha256) and isinstance(inputs, dict) and hex64(inputs.get("sha256")),
            "Bound source closure digest and input manifest required")
    return {
        "schema": "tool1-bench4-bank-function-v1", "role": ROLE, "protocol": name, "bank": value["bank"],
        "split": value["split"], "edition": value["edition"], "cell": cell,
        "model_identity_sha256": model["parameter_sha256"],
        "auxiliary_identity_sha256": model["auxiliary_identity_sha256"],
        "model_generation_config": copy.deepcopy(model["generation_config"]),
        "adapter_active": model["adapter_active"], "adapter_merged": False,
        "full_roster_sha256": roster_summary["ordered_semantic_keys_sha256"],
        "native_inputs_sha256": roster_summary["ordered_native_inputs_sha256"],
        "questions_sha256": inputs["sha256"], "decoding": copy.deepcopy(value["decoding"]),
        "suffix": value["suffix"], "image_mode": value["image_mode"], "prompt_policy": value["prompt_policy"],
        "sources_sha256": sources_sha256, "llava_revision": LLAVA_REVISION,
    }


def raw_answer(name, row, text):
    """Native LLaVA answer row (minus the random answer_id/model_id fields)."""
    value = protocol(name)
    require(isinstance(text, str), "Generated text must be a string")
    if value["kind"] == "mmbench":
        native = row["native_input"]
        return {"question_id": native["index"], "round_id": native["round_id"], "prompt": row["cur_prompt"],
                "text": text, "options": native["options"], "option_char": native["option_char"], "metadata": {}}
    if value["kind"] == "mmmu":
        return {"question_id": row["native_id"], "prompt": row["cur_prompt"], "text": text,
                "question_type": row["native_input"]["question_type"], "metadata": {}}
    return {"question_id": row["native_id"], "prompt": row["cur_prompt"], "text": text, "metadata": {}}


def completion(name, accepted, missing, items):
    require(type(accepted) is int and type(missing) is int and missing >= 0 and accepted >= 0
            and accepted + missing == items, "Accepted plus missing must equal the full bank")
    return {"protocol": name, "items": items, "accepted": accepted, "missing": missing,
            "complete": missing == 0 and accepted == items}


def parse_predictions(name, raw, rows):
    """Exported per-QA answers aligned one-to-one with the roster order."""
    require(isinstance(raw, bytes) and raw.endswith(b"\n"), "Complete newline-terminated JSONL required")
    lines = raw[:-1].split(b"\n")
    require(len(lines) == len(rows), "Prediction count differs from the full roster")
    result = []
    for row, line in zip(rows, lines):
        value = json.loads(line)
        require(isinstance(value, dict) and type(value.get("question_id")) is type(row["native_id"])
                and value["question_id"] == row["native_id"] and value.get("prompt") == row["cur_prompt"]
                and isinstance(value.get("text"), str), "Prediction order, ID or prompt differs")
        require(not any(field in value for field in FORBIDDEN_FIELDS), "Predictions must not carry label fields")
        if protocol(name)["kind"] == "mmbench":
            require(value.get("round_id") == row["round_id"], "MMBench round differs")
        result.append(value)
    return result


def image_path_kind(path):
    return "wild_shared" if path.startswith(OFF + "/") else "inventory"


def relative_image_name(digest, suffix):
    require(hex64(digest) and suffix in (".jpg", ".png", ".gif", ".bmp", ".webp", ".bin"), "Image name parts")
    return str(PurePosixPath(digest[:2]) / (digest + suffix))
